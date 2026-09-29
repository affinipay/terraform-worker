"""GitHub handler for reporting plan status to a pull request.

Authenticates as a GitHub App and, for plan runs, maintains:

- a rollup check run on the PR head commit whose markdown output carries the
  per-definition job summary, plus one check run per definition (created
  queued, started at plan pre, and concluded from the plan result)
- a live status comment on the pull request (when one is configured),
  updated in place as each definition plans; per-definition detail is
  included only when comment_details is enabled, split across additional
  comments when the body exceeds GitHub's size limit

When the openai handler is also configured, its plan summary is embedded in
the per-definition details; otherwise a trimmed copy of the plan output is
used.

Example configuration (all credential fields fall back to environment
variables, so an empty mapping works in GitHub Actions with a configured app)::

    handlers:
      github:
        repository: "myorg/myrepo"          # or GITHUB_REPOSITORY
        pull_request: "{{ env.PR_NUMBER }}" # or GITHUB_PULL_REQUEST / PULL_REQUEST
        app_id: "12345"                     # or GITHUB_APP_ID
        private_key_file: /secrets/app.pem  # or private_key / GITHUB_APP_PRIVATE_KEY(_FILE)
"""

import os
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping

import click
from pydantic import BaseModel, Field, field_validator, model_validator

import tfworker.util.log as log
from tfworker.custom_types.terraform import TerraformAction, TerraformStage
from tfworker.exceptions import HandlerError
from tfworker.util.system import strip_ansi

from .base import BaseHandler
from .registry import HandlerRegistry
from .results import BaseHandlerResult

if TYPE_CHECKING:  # pragma: no cover
    from github.CheckRun import CheckRun
    from github.Issue import Issue
    from github.IssueComment import IssueComment
    from github.PullRequest import PullRequest
    from github.Repository import Repository

    from tfworker.commands.terraform import TerraformResult
    from tfworker.definitions.collection import DefinitionsCollection
    from tfworker.definitions.model import Definition

# GitHub caps comment and check run bodies at 65536 characters; this leaves headroom for markers
BODY_BUDGET = 60000
# room for a detail chunk after a continuation comment's marker, heading and <details> wrapper
DETAIL_CHUNK_LIMIT = BODY_BUDGET - 1000
# consecutive API failures before the handler disables itself
MAX_API_FAILURES = 3


def _hard_wrap(text: str, width: int) -> str:
    """Hard-wrap long lines, indenting continuations; unbreakable tokens stay intact."""
    if width <= 0:
        return text
    out: list[str] = []
    for line in text.splitlines():
        if len(line) <= width:
            out.append(line)
            continue
        indent = line[: len(line) - len(line.lstrip())] + "    "
        out.extend(
            textwrap.wrap(
                line,
                width=width,
                subsequent_indent=indent,
                break_long_words=False,
                break_on_hyphens=False,
            )
            or [line]
        )
    return "\n".join(out)


OUTPUT_CHANGES_HEADING = "Changes to Outputs:"

# where the changes start: resource actions, or outputs for a plan that only changes outputs
PLAN_SECTION_HEADINGS = (
    "Terraform will perform the following actions:",
    OUTPUT_CHANGES_HEADING,
)

# terraform's trailing footer, which follows the changes
PLAN_FOOTER_PREFIXES = (
    "Saved the plan to:",
    "To perform exactly these actions",
    "You can apply this plan to save these new output values",
    "Note: You didn't use the -out option",
)


def _is_plan_footer(line: str) -> bool:
    """Whether a line begins terraform's post-plan footer or its box-drawing rule."""
    if line.startswith(PLAN_FOOTER_PREFIXES):
        return True
    stripped = line.strip()
    return bool(stripped) and set(stripped) == {"─"}


# terraform plan change markers, optionally indented: +, -, ~, -/+, +/-
_PLAN_MARKER = re.compile(r"^( +)([+~-]|[+-]/[+-]) ", re.MULTILINE)


def _diff_format(text: str) -> str:
    """Hoist change markers to column 0 (update/replace become ``!``) for ```diff highlighting."""

    def _sub(m: "re.Match") -> str:
        marker = m.group(2)
        if marker not in ("+", "-"):
            marker = "!"
        return f"{marker}{m.group(1)} "

    return _PLAN_MARKER.sub(_sub, text)


def _fence_safe_truncate(text: str, limit: int) -> str:
    """Truncate markdown on a line boundary, closing any open code fence."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if "\n" in cut:
        cut = cut[: cut.rfind("\n") + 1]
    if cut.count("```") % 2 == 1:
        cut += "```\n"
    return cut + "\n_… truncated_"


def _chunk_markdown(text: str, limit: int) -> list[str]:
    """Split markdown on line boundaries, closing and reopening code fences across chunks."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    open_fence: str | None = None
    for line in text.splitlines(keepends=True):
        if current and size + len(line) > limit:
            if open_fence:
                current.append("```\n")
            chunks.append("".join(current).rstrip("\n"))
            current = [f"{open_fence}\n"] if open_fence else []
            size = sum(len(part) for part in current)
        current.append(line)
        size += len(line)
        if line.strip().startswith("```"):
            # the opening fence line, with any language tag, reopens the next chunk
            open_fence = None if open_fence else line.strip()
    if current:
        chunks.append("".join(current).rstrip("\n"))
    return chunks


def _env_int(label: str, *names: str) -> int | None:
    """The first set environment variable of names as an int; None when unset or non-numeric."""
    raw = next((os.environ[name] for name in names if os.environ.get(name)), None)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        log.warn(f"github handler: ignoring non-numeric {label} {raw!r}")
        return None


class GithubConfig(BaseModel):
    repository: str | None = Field(
        default=None,
        description="Repository in 'owner/repo' form; falls back to GITHUB_REPOSITORY.",
    )
    pull_request: int | None = Field(
        default=None,
        description="Pull request number to comment on; falls back to GITHUB_PULL_REQUEST or PULL_REQUEST. When unset, only the check run is maintained.",
    )
    commit_sha: str | None = Field(
        default=None,
        description="Commit sha for the check runs; defaults to the PR head sha, then GITHUB_SHA.",
    )
    app_id: str | None = Field(
        default=None,
        description="GitHub App id; falls back to GITHUB_APP_ID.",
    )
    private_key: str | None = Field(
        default=None,
        repr=False,
        description="GitHub App private key (PEM content); falls back to GITHUB_APP_PRIVATE_KEY.",
    )
    private_key_file: str | None = Field(
        default=None,
        description="Path to the GitHub App private key; falls back to GITHUB_APP_PRIVATE_KEY_FILE.",
    )
    installation_id: int | None = Field(
        default=None,
        description="App installation id; discovered from the repository when unset. Falls back to GITHUB_APP_INSTALLATION_ID.",
    )
    check_run_name: str | None = Field(
        default=None,
        description="Name of the check run; defaults to 'tfworker/<deployment>/plan'.",
    )
    comment_marker: str = "tfworker-status"
    comment_details: bool = Field(
        default=False,
        description="Include per-definition detail (plan output or AI summary) in the PR comment, split across additional comments when needed. When false the comment carries only the status table; detail lives in the check run output.",
    )
    max_detail_chars: int = Field(
        default=8000,
        description="Maximum characters of per-definition detail included in check run output, which cannot be split. PR comments always carry the full detail, split across comments as needed.",
    )
    wrap_width: int = Field(
        default=120,
        description="Hard-wrap fenced plan and error output at this column so check run pages do not scroll horizontally. 0 disables wrapping.",
    )
    required: bool = False

    model_config = {"extra": "forbid"}

    @field_validator(
        "repository",
        "pull_request",
        "commit_sha",
        "app_id",
        "private_key",
        "private_key_file",
        "installation_id",
        "check_run_name",
        mode="before",
    )
    @classmethod
    def _empty_to_none(cls, v):
        # tolerate empty Jinja renders like pull_request: "{{ env.PR_NUMBER }}"
        if isinstance(v, str) and v.strip().lower() in ("", "null", "none"):
            return None
        return v

    @model_validator(mode="after")
    def _env_fallbacks(self) -> "GithubConfig":
        self.repository = self.repository or os.environ.get("GITHUB_REPOSITORY") or None
        self.app_id = self.app_id or os.environ.get("GITHUB_APP_ID") or None
        self.private_key = (
            self.private_key or os.environ.get("GITHUB_APP_PRIVATE_KEY") or None
        )
        self.private_key_file = (
            self.private_key_file
            or os.environ.get("GITHUB_APP_PRIVATE_KEY_FILE")
            or None
        )
        if self.pull_request is None:
            self.pull_request = _env_int(
                "PR number", "GITHUB_PULL_REQUEST", "PULL_REQUEST"
            )
        if self.installation_id is None:
            self.installation_id = _env_int(
                "installation id", "GITHUB_APP_INSTALLATION_ID"
            )
        return self

    def settings_errors(self) -> list[str]:
        errors = []
        if not self.repository:
            errors.append("missing repository")
        if not self.app_id:
            errors.append("missing app_id")
        if not self.private_key and not self.private_key_file:
            errors.append("missing private_key or private_key_file")
        if self.private_key and self.private_key_file:
            errors.append("private_key and private_key_file are mutually exclusive")
        return errors


class GithubResult(BaseHandlerResult):
    definition: str
    check_run_url: str | None = None
    definition_check_url: str | None = None
    comment_url: str | None = None


PlanStatus = Literal["pending", "running", "no_changes", "changes", "failed", "skipped"]

STATUS_DISPLAY: dict[PlanStatus, str] = {
    "pending": "⏳ pending",
    "running": "🔄 planning",
    "no_changes": "✅ no changes",
    "changes": "📝 changes",
    "failed": "❌ failed",
    "skipped": "⏭️ skipped",
}


@dataclass
class DefinitionStatus:
    """One definition's row in the status report."""

    status: PlanStatus = "pending"
    plan_line: str = ""
    detail: str = ""
    url: str = ""


class GithubStatusReport:
    """State and markdown rendering for the status comments and check run; makes no API calls."""

    def __init__(
        self,
        deployment: str,
        marker: str,
        max_detail_chars: int,
        include_details: bool = False,
        run_id: str | None = None,
    ) -> None:
        self.deployment = deployment
        self.marker = marker
        self.max_detail_chars = max_detail_chars
        self.include_details = include_details
        self.run_id = run_id
        self.check_url: str | None = None
        self._rows: dict[str, DefinitionStatus] = {}

    @property
    def rows(self) -> Mapping[str, DefinitionStatus]:
        return self._rows

    @property
    def primary_marker(self) -> str:
        return f"<!-- {self.marker}: {self.deployment} -->"

    def part_marker(self, part: int) -> str:
        return f"<!-- {self.marker}: {self.deployment} part={part} -->"

    def marker_prefix(self) -> str:
        return f"<!-- {self.marker}: {self.deployment}"

    def ensure(self, name: str) -> DefinitionStatus:
        return self._rows.setdefault(name, DefinitionStatus())

    def set_url(self, name: str, url: str) -> None:
        self.ensure(name).url = url

    def mark(
        self, name: str, status: PlanStatus, plan_line: str = "", detail: str = ""
    ) -> None:
        row = self.ensure(name)
        row.status = status
        if plan_line:
            row.plan_line = plan_line
        if detail:
            row.detail = detail

    def finalize(self) -> None:
        """Mark anything that never ran as skipped (aborted or filtered runs)."""
        for row in self._rows.values():
            if row.status in ("pending", "running"):
                row.status = "skipped"

    def conclusion(self) -> str:
        if any(r.status == "failed" for r in self._rows.values()):
            return "failure"
        return "success"

    def summary_line(self) -> str:
        counts: dict[PlanStatus, int] = {}
        for row in self._rows.values():
            counts[row.status] = counts.get(row.status, 0) + 1
        parts = [
            f"{counts[s]} {STATUS_DISPLAY[s]}" for s in STATUS_DISPLAY if s in counts
        ]
        return ", ".join(parts) if parts else "no definitions"

    def _header(self) -> str:
        lines = [f"## Terraform plan status: `{self.deployment}`", ""]
        if self.run_id:
            # the id the stored plans are keyed by, so an apply can request exactly these plans
            lines.append(f"Run `{self.run_id}`")
            lines.append("")
        if self.check_url:
            lines.append(f"[View check run]({self.check_url})")
            lines.append("")
        return "\n".join(lines)

    def _table(self) -> str:
        lines = ["| Definition | Status | Plan |", "| --- | --- | --- |"]
        for name, row in self._rows.items():
            label = f"[`{name}`]({row.url})" if row.url else f"`{name}`"
            lines.append(
                f"| {label} | {STATUS_DISPLAY[row.status]} | {row.plan_line} |"
            )
        return "\n".join(lines)

    def _wrap_details(
        self, name: str, row: DefinitionStatus, body: str, part: str = ""
    ) -> str:
        plan_line = f" — {row.plan_line}" if row.plan_line else ""
        return (
            "<details>\n"
            f"<summary><code>{name}</code> — {STATUS_DISPLAY[row.status]}"
            f"{plan_line}{part}</summary>\n\n"
            f"{body}\n\n"
            "</details>"
        )

    def _detail_blocks(self) -> list[str]:
        """Comment detail blocks; a detail too large for one comment is chunked across several."""
        blocks = []
        for name, row in self._rows.items():
            if not row.detail:
                continue
            chunks = _chunk_markdown(row.detail, DETAIL_CHUNK_LIMIT)
            for i, chunk in enumerate(chunks):
                part = f" (part {i + 1}/{len(chunks)})" if len(chunks) > 1 else ""
                blocks.append(self._wrap_details(name, row, chunk, part))
        return blocks

    def _check_detail_blocks(self) -> list[str]:
        """Check run detail blocks, each capped at max_detail_chars as check output cannot split."""
        blocks = []
        for name, row in self._rows.items():
            if not row.detail:
                continue
            body = _fence_safe_truncate(row.detail, self.max_detail_chars)
            blocks.append(self._wrap_details(name, row, body))
        return blocks

    def render_comment_bodies(self) -> list[str]:
        """The primary comment with the table, plus continuation comments for detail that does not fit."""
        primary = "\n".join([self.primary_marker, self._header(), self._table(), ""])
        bodies = [primary]
        if not self.include_details:
            return bodies
        for block in self._detail_blocks():
            if len(bodies[-1]) + len(block) + 2 <= BODY_BUDGET:
                bodies[-1] = f"{bodies[-1]}\n{block}"
            else:
                part = len(bodies) + 1
                bodies.append(
                    "\n".join(
                        [
                            self.part_marker(part),
                            f"_Terraform plan status for `{self.deployment}` "
                            f"(continued, part {part})_",
                            "",
                            block,
                        ]
                    )
                )
        return bodies

    def render_check_summary(self) -> str:
        """The rollup check run output markdown."""
        body = "\n".join(
            [self._header(), self._table(), ""] + self._check_detail_blocks()
        )
        if len(body) > BODY_BUDGET:
            body = "\n".join(
                [
                    self._header(),
                    self._table(),
                    "",
                    "_Detail sections omitted; body exceeded GitHub's size limit._",
                ]
            )
        return body


StageFunction = Callable[..., "GithubResult | None"]


@HandlerRegistry.register("github")
class GithubHandler(BaseHandler):
    """Report plan progress and results to GitHub."""

    actions = [TerraformAction.PLAN]
    config_model = GithubConfig
    _ready = False
    default_priority = {
        TerraformAction.PLAN: 90,
    }
    # run after openai, when configured, so its plan summary is in the shared results
    dependencies = {
        TerraformAction.PLAN: {TerraformStage.POST: ["openai"]},
    }

    def __init__(self, config: GithubConfig) -> None:
        self.config = config
        self._app_state = None
        self._repo: "Repository | None" = None
        self._pr: "PullRequest | None" = None
        self._issue: "Issue | None" = None
        self._check: "CheckRun | None" = None
        self._def_checks: dict[str, "CheckRun"] = {}
        self._def_concluded: set[str] = set()
        self._report: GithubStatusReport | None = None
        self._comments: list["IssueComment"] = []
        self._api_failures = 0
        self.execution_functions: dict[
            TerraformAction, dict[TerraformStage, StageFunction]
        ] = {
            TerraformAction.PLAN: {
                TerraformStage.PRE: self._pre_plan,
                TerraformStage.POST: self._post_plan,
                TerraformStage.ERROR: self._plan_error,
            },
        }

        errors = config.settings_errors()
        if errors:
            msg = f"github handler misconfigured: {'; '.join(errors)}"
            if config.required:
                raise HandlerError(msg)
            log.warn(f"{msg}; github handler disabled")
            self._ready = False
            return
        self._ready = True

    @property
    def app_state(self):
        if self._app_state is None:
            self._app_state = click.get_current_context().obj
        return self._app_state

    def _run_id(self) -> str | None:
        """The run id plans are stored under; None when the run has none."""
        try:
            return self.app_state.root_options.run_id or None
        except (AttributeError, RuntimeError):  # no click context or root options
            return None

    def is_ready(self) -> bool:
        return self._ready

    ###########################################################################
    # lifecycle
    ###########################################################################
    def setup(
        self,
        deployment: str,
        definitions: "DefinitionsCollection",
        working_dir: str,
        terraform_options,
    ) -> None:
        """Authenticate, create the check runs, and seed the status comment."""
        if not self._ready:
            return
        if not terraform_options.plan:
            log.debug("github handler: plan not requested, nothing to report")
            self._ready = False
            return
        try:
            self._connect()
            run_id = self._run_id()
            self._report = GithubStatusReport(
                deployment=deployment,
                marker=self.config.comment_marker,
                max_detail_chars=self.config.max_detail_chars,
                include_details=self.config.comment_details,
                run_id=run_id,
            )
            for defn in definitions.values():
                self._report.ensure(defn.name)
            # needs the report: the claimed marker is scoped to the deployment
            self._claim_comments()

            head_sha = self._resolve_sha()
            check_name = self.config.check_run_name or f"tfworker/{deployment}/plan"
            check_kwargs: dict[str, Any] = {"external_id": run_id} if run_id else {}
            self._check = self._repo.create_check_run(
                name=check_name,
                head_sha=head_sha,
                status="in_progress",
                output={
                    "title": "Terraform plan in progress",
                    "summary": self._report.render_check_summary(),
                },
                **check_kwargs,
            )
            self._report.check_url = self._check.html_url
            for defn in definitions.values():
                check = self._repo.create_check_run(
                    name=f"{check_name}: {defn.name}",
                    head_sha=head_sha,
                    status="queued",
                )
                self._def_checks[defn.name] = check
                self._report.set_url(defn.name, check.html_url)
            self._update_comments()
        except Exception as e:
            log.error(f"github handler setup failed: {e}")
            self._ready = False
            if self.config.required:
                raise HandlerError(f"github handler setup failed: {e}")

    def execute(
        self,
        action: "TerraformAction",
        stage: "TerraformStage",
        deployment: str,
        definition: "Definition",
        working_dir: str,
        result: "TerraformResult | None" = None,
    ) -> GithubResult | None:
        if not self._ready or self._report is None:
            return None
        function = self.execution_functions.get(action, {}).get(stage)
        if function is None or (stage != TerraformStage.PRE and result is None):
            return None
        try:
            ret = function(definition, result)
        except Exception as e:
            self._record_api_failure(f"{action} {stage} for {definition.name}", e)
            return None
        self._api_failures = 0
        return ret

    def teardown(self, deployment: str, working_dir: str) -> None:
        """Conclude the check runs and finalize the comment."""
        if self._report is None:
            return
        try:
            self._report.finalize()
            for name in self._def_checks:
                self._conclude_def_check(name, "skipped", "Plan skipped")
            conclusion = self._report.conclusion()
            title = f"Terraform plan {conclusion}: {self._report.summary_line()}"
            if self._check is not None:
                self._check.edit(
                    status="completed",
                    conclusion=conclusion,
                    output={
                        "title": title,
                        "summary": self._report.render_check_summary(),
                    },
                )
            self._update_comments()
        except Exception as e:
            log.error(f"github handler teardown failed: {e}")

    ###########################################################################
    # plan handling
    ###########################################################################
    def _pre_plan(
        self, definition: "Definition", result: "TerraformResult | None" = None
    ) -> None:
        self._report.mark(definition.name, "running")
        check = self._def_checks.get(definition.name)
        if check is not None:
            check.edit(status="in_progress")
        self._update_comments()

    def _post_plan(
        self, definition: "Definition", result: "TerraformResult"
    ) -> GithubResult:
        if result.exit_code == 0:
            self._report.mark(definition.name, "no_changes", plan_line="No changes.")
            self._conclude_def_check(definition.name, "success", "No changes.")
        elif result.has_changes():
            text = strip_ansi(result.stdout_str)
            detail = self._summary_for(definition) or self._trimmed_plan(
                text, self.config.wrap_width
            )
            plan_line = self._plan_line(text)
            self._report.mark(
                definition.name, "changes", plan_line=plan_line, detail=detail
            )
            self._conclude_def_check(
                definition.name, "success", plan_line or "Changes planned", detail
            )
        else:
            error_detail = self._error_detail(result)
            self._report.mark(definition.name, "failed", detail=error_detail)
            self._conclude_def_check(
                definition.name, "failure", "Terraform plan failed", error_detail
            )
        self._update_comments()
        self._update_check("Terraform plan in progress")
        def_check = self._def_checks.get(definition.name)
        return GithubResult(
            handler="github",
            action=TerraformAction.PLAN,
            stage=TerraformStage.POST,
            definition=definition.name,
            check_run_url=self._check.html_url if self._check else None,
            definition_check_url=def_check.html_url if def_check else None,
            comment_url=self._comments[0].html_url if self._comments else None,
        )

    def _plan_error(self, definition: "Definition", result: "TerraformResult") -> None:
        detail = self._error_detail(result)
        self._report.mark(definition.name, "failed", detail=detail)
        self._conclude_def_check(
            definition.name, "failure", "Terraform plan failed", detail
        )
        self._update_comments()
        self._update_check("Terraform plan in progress")

    def _summary_for(self, definition: "Definition") -> str | None:
        """The openai-generated summary for this definition, if any."""
        if definition.plan_file is None:
            return None
        try:
            expected = str(
                Path(definition.plan_file)
                .with_suffix(".tfplan.json")
                .with_suffix(".summary.md")
            )
            for r in self.app_state.handlers.get_results(handler_name="openai"):
                if getattr(r, "task", None) == "summary" and r.file == expected:
                    return r.content
        except Exception as e:
            log.debug(f"github handler: unable to read openai results: {e}")
        return None

    @staticmethod
    def _plan_line(text: str) -> str:
        # a plan that only changes outputs has no "Plan:" line; name that case instead
        outputs_only = False
        for line in text.splitlines():
            if line.startswith("Plan:"):
                return line.strip()
            if line.startswith(OUTPUT_CHANGES_HEADING):
                outputs_only = True
        return "Output changes only." if outputs_only else ""

    @staticmethod
    def _trimmed_plan(text: str, wrap_width: int = 0) -> str:
        """The planned changes, from the first section heading to terraform's footer, fenced for markdown."""
        capture = False
        lines = []
        for line in text.splitlines():
            if not capture and line.startswith(PLAN_SECTION_HEADINGS):
                capture = True
            elif capture and _is_plan_footer(line):
                capture = False
            if capture:
                lines.append(line)
        trimmed = "\n".join(lines).rstrip()
        if not trimmed:
            return ""
        return f"```diff\n{_hard_wrap(_diff_format(trimmed), wrap_width)}\n```"

    def _error_detail(self, result: "TerraformResult") -> str:
        text = strip_ansi(result.stderr_str or result.stdout_str)
        lines = [line for line in text.splitlines() if line.strip()]
        snippet = _hard_wrap("\n".join(lines[-20:]), self.config.wrap_width)
        return f"```\n{snippet}\n```" if snippet else ""

    ###########################################################################
    # github api plumbing
    ###########################################################################
    def _connect(self) -> None:
        from github import Auth, GithubIntegration

        pem = self.config.private_key or Path(self.config.private_key_file).read_text()
        integration = GithubIntegration(auth=Auth.AppAuth(self.config.app_id, pem))
        owner, repo_name = self.config.repository.split("/", 1)
        installation_id = (
            self.config.installation_id
            or integration.get_repo_installation(owner, repo_name).id
        )
        gh = integration.get_github_for_installation(installation_id)
        self._repo = gh.get_repo(self.config.repository)
        if self.config.pull_request:
            self._pr = self._repo.get_pull(self.config.pull_request)
            self._issue = self._repo.get_issue(self.config.pull_request)

    def _resolve_sha(self) -> str:
        """The commit to report on: commit_sha, then the PR head, then GITHUB_SHA."""
        # GITHUB_SHA is a merge commit on pull_request events; its checks do not render on the PR
        sha = (
            self.config.commit_sha
            or (self._pr.head.sha if self._pr is not None else None)
            or os.environ.get("GITHUB_SHA")
        )
        if not sha:
            raise HandlerError("no commit to report on; set commit_sha or GITHUB_SHA")
        return sha

    def _claim_comments(self) -> None:
        """Adopt this deployment's status comments from a prior run so they are edited, not duplicated."""
        if self._issue is None or self._report is None:
            return
        primary_marker = self._report.primary_marker
        part_prefix = f"{self._report.marker_prefix()} part="
        primary, parts = None, []
        for comment in self._issue.get_comments():
            # exact first-line match: one deployment must not claim another whose name it prefixes
            first_line = (comment.body or "").split("\n", 1)[0].strip()
            if first_line == primary_marker:
                primary = comment
            elif first_line.startswith(part_prefix):
                parts.append(comment)
        self._comments = ([primary] if primary else []) + parts

    def _update_comments(self) -> None:
        if self._issue is None or self._report is None:
            return
        bodies = self._report.render_comment_bodies()
        for i, body in enumerate(bodies):
            if i < len(self._comments):
                if self._comments[i].body != body:
                    self._comments[i].edit(body)
            else:
                self._comments.append(self._issue.create_comment(body))
        # remove continuation comments left over from a larger update
        while len(self._comments) > len(bodies):
            self._comments.pop().delete()

    def _update_check(self, title: str) -> None:
        if self._check is None or self._report is None:
            return
        self._check.edit(
            output={"title": title, "summary": self._report.render_check_summary()}
        )

    def _conclude_def_check(
        self, name: str, conclusion: str, title: str, summary: str = ""
    ) -> None:
        """Complete a per-definition check run; once concluded it stays as-is."""
        check = self._def_checks.get(name)
        if check is None or name in self._def_concluded:
            return
        self._def_concluded.add(name)
        check.edit(
            status="completed",
            conclusion=conclusion,
            output={
                "title": title[:255],
                "summary": _fence_safe_truncate(summary, BODY_BUDGET),
            },
        )

    def _record_api_failure(self, context: str, exc: Exception) -> None:
        self._api_failures += 1
        log.error(f"github handler {context} failed: {exc}")
        if self._api_failures >= MAX_API_FAILURES:
            log.warn(
                f"github handler disabled after {self._api_failures} consecutive API failures"
            )
            self._ready = False
