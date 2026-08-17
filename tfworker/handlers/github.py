"""GitHub handler for reporting plan status to a pull request.

Authenticates as a GitHub App and, for plan runs, maintains:

- a rollup check run on the PR head commit whose markdown output carries the
  per-definition job summary, plus one check run per definition (created
  queued, started at plan pre, and concluded from the plan result) so each
  definition reports its own status like Atlantis project checks
- a live status comment on the pull request (when one is configured),
  updated in place as each definition plans; the comment carries the
  status table, with per-definition detail included only when
  comment_details is enabled (split across additional comments when the
  body exceeds GitHub's size limit)

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
import subprocess
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

import click
from pydantic import BaseModel, Field, field_validator, model_validator

import tfworker.util.log as log
from tfworker.custom_types.terraform import TerraformAction, TerraformStage
from tfworker.exceptions import HandlerError

from ..util.system import strip_ansi
from .base import BaseHandler
from .registry import HandlerRegistry
from .results import BaseHandlerResult

if TYPE_CHECKING:  # pragma: no cover
    from tfworker.commands.terraform import TerraformResult
    from tfworker.definitions.collection import DefinitionsCollection
    from tfworker.definitions.model import Definition

# GitHub caps issue comment and check run output bodies at 65536 characters;
# leave headroom for the markers and truncation notices added around them.
GITHUB_BODY_LIMIT = 65536
BODY_BUDGET = 60000
# room left for a detail chunk after a continuation comment's marker,
# heading, and <details> wrapper
DETAIL_CHUNK_LIMIT = BODY_BUDGET - 1000

# maximum consecutive API failures before the handler disables itself
MAX_API_FAILURES = 3


def _hard_wrap(text: str, width: int) -> str:
    """Hard-wrap long lines, keeping indentation on continuation lines.

    GitHub renders fenced code blocks without soft wrapping and strips
    style attributes, so pre-wrapping is the only way to avoid horizontal
    scrolling on check run pages. Long unbreakable tokens (ARNs, URLs)
    are left intact rather than split.
    """
    if width <= 0:
        return text
    out: List[str] = []
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


# terraform plan change markers, optionally indented: +, -, ~, -/+, +/-
_PLAN_MARKER = re.compile(r"^( +)([+~-]|[+-]/[+-]) ", re.MULTILINE)


def _diff_format(text: str) -> str:
    """Hoist terraform's change markers to column 0 (update/replace markers
    become ``!``) so GitHub's ```diff fenced-block highlighting colors them."""

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


def _chunk_markdown(text: str, limit: int) -> List[str]:
    """Split markdown into chunks on line boundaries, closing and reopening
    code fences across chunk boundaries so each chunk renders standalone."""
    if len(text) <= limit:
        return [text]
    chunks: List[str] = []
    current: List[str] = []
    size = 0
    open_fence: Optional[str] = None
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
            # remember the opening fence line (with any language tag) so the
            # next chunk reopens the block identically
            open_fence = None if open_fence else line.strip()
    if current:
        chunks.append("".join(current).rstrip("\n"))
    return chunks


class GithubConfig(BaseModel):
    repository: Optional[str] = Field(
        default=None,
        description="Repository in 'owner/repo' form; falls back to GITHUB_REPOSITORY.",
    )
    pull_request: Optional[int] = Field(
        default=None,
        description="Pull request number to comment on; falls back to GITHUB_PULL_REQUEST or PULL_REQUEST. When unset, only the check run is maintained.",
    )
    commit_sha: Optional[str] = Field(
        default=None,
        description="Commit sha for the check run; defaults to the PR head sha, then GITHUB_SHA, then `git rev-parse HEAD`.",
    )
    app_id: Optional[str] = Field(
        default=None,
        description="GitHub App id; falls back to GITHUB_APP_ID.",
    )
    private_key: Optional[str] = Field(
        default=None,
        repr=False,
        description="GitHub App private key (PEM content); falls back to GITHUB_APP_PRIVATE_KEY.",
    )
    private_key_file: Optional[str] = Field(
        default=None,
        description="Path to the GitHub App private key; falls back to GITHUB_APP_PRIVATE_KEY_FILE.",
    )
    installation_id: Optional[int] = Field(
        default=None,
        description="App installation id; discovered from the repository when unset. Falls back to GITHUB_APP_INSTALLATION_ID.",
    )
    check_run_name: Optional[str] = Field(
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

    @field_validator("*", mode="before")
    @classmethod
    def _empty_to_none(cls, v):
        # tolerate empty Jinja renders like pull_request: "{{ env.PR_NUMBER }}"
        if isinstance(v, str) and v.strip().lower() in ("", "null", "none"):
            return None
        return v

    @model_validator(mode="after")
    def _env_fallbacks(self):
        if self.repository is None:
            self.repository = os.environ.get("GITHUB_REPOSITORY") or None
        if self.app_id is None:
            self.app_id = os.environ.get("GITHUB_APP_ID") or None
        if self.private_key is None:
            self.private_key = os.environ.get("GITHUB_APP_PRIVATE_KEY") or None
        if self.private_key_file is None:
            self.private_key_file = (
                os.environ.get("GITHUB_APP_PRIVATE_KEY_FILE") or None
            )
        if self.pull_request is None:
            raw = os.environ.get("GITHUB_PULL_REQUEST") or os.environ.get(
                "PULL_REQUEST"
            )
            if raw:
                try:
                    self.pull_request = int(raw)
                except ValueError:
                    log.warn(f"github handler: ignoring non-numeric PR number {raw!r}")
        if self.installation_id is None:
            raw = os.environ.get("GITHUB_APP_INSTALLATION_ID")
            if raw:
                try:
                    self.installation_id = int(raw)
                except ValueError:
                    log.warn(
                        f"github handler: ignoring non-numeric installation id {raw!r}"
                    )
        return self

    def missing_settings(self) -> List[str]:
        missing = []
        if not self.repository:
            missing.append("repository")
        if not self.app_id:
            missing.append("app_id")
        if not self.private_key and not self.private_key_file:
            missing.append("private_key or private_key_file")
        if self.private_key and self.private_key_file:
            missing.append("only one of private_key / private_key_file")
        return missing


class GithubResult(BaseHandlerResult):
    definition: str
    check_run_url: Optional[str] = None
    definition_check_url: Optional[str] = None
    comment_url: Optional[str] = None


STATUS_DISPLAY = {
    "pending": "⏳ pending",
    "running": "🔄 planning",
    "no_changes": "✅ no changes",
    "changes": "📝 changes",
    "failed": "❌ failed",
    "skipped": "⏭️ skipped",
}


class GithubStatusReport:
    """Pure state and markdown rendering for the status comment and check run.

    Holds one row per definition and renders the comment bodies (splitting
    across multiple comments when over GitHub's size limit) and the check run
    summary. Performs no API calls.
    """

    def __init__(
        self,
        deployment: str,
        marker: str,
        max_detail_chars: int,
        include_details: bool = False,
    ) -> None:
        self.deployment = deployment
        self.marker = marker
        self.max_detail_chars = max_detail_chars
        self.include_details = include_details
        self.check_url: Optional[str] = None
        self._rows: dict = {}

    @property
    def primary_marker(self) -> str:
        return f"<!-- {self.marker}: {self.deployment} -->"

    def part_marker(self, part: int) -> str:
        return f"<!-- {self.marker}: {self.deployment} part={part} -->"

    def marker_prefix(self) -> str:
        return f"<!-- {self.marker}: {self.deployment}"

    def ensure(self, name: str) -> None:
        self._rows.setdefault(
            name, {"status": "pending", "plan_line": "", "detail": "", "url": ""}
        )

    def set_url(self, name: str, url: str) -> None:
        self.ensure(name)
        self._rows[name]["url"] = url

    def mark(self, name: str, status: str, plan_line: str = "", detail: str = ""):
        self.ensure(name)
        row = self._rows[name]
        row["status"] = status
        if plan_line:
            row["plan_line"] = plan_line
        if detail:
            row["detail"] = detail

    def finalize(self) -> None:
        """Mark anything that never ran as skipped (aborted or filtered runs)."""
        for row in self._rows.values():
            if row["status"] in ("pending", "running"):
                row["status"] = "skipped"

    def conclusion(self) -> str:
        if any(r["status"] == "failed" for r in self._rows.values()):
            return "failure"
        return "success"

    def summary_line(self) -> str:
        counts: dict = {}
        for row in self._rows.values():
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        parts = [
            f"{counts[s]} {STATUS_DISPLAY[s]}" for s in STATUS_DISPLAY if s in counts
        ]
        return ", ".join(parts) if parts else "no definitions"

    def _header(self) -> str:
        lines = [f"## Terraform plan status: `{self.deployment}`", ""]
        if self.check_url:
            lines.append(f"[View check run]({self.check_url})")
            lines.append("")
        return "\n".join(lines)

    def _table(self) -> str:
        lines = ["| Definition | Status | Plan |", "| --- | --- | --- |"]
        for name, row in self._rows.items():
            display = STATUS_DISPLAY.get(row["status"], row["status"])
            label = f"[`{name}`]({row['url']})" if row["url"] else f"`{name}`"
            lines.append(f"| {label} | {display} | {row['plan_line']} |")
        return "\n".join(lines)

    def _wrap_details(self, name: str, row: dict, body: str, part: str = "") -> str:
        display = STATUS_DISPLAY.get(row["status"], row["status"])
        return (
            "<details>\n"
            f"<summary><code>{name}</code> — {display}"
            f"{' — ' + row['plan_line'] if row['plan_line'] else ''}{part}</summary>\n\n"
            f"{body}\n\n"
            "</details>"
        )

    def _detail_blocks(self) -> List[str]:
        """Detail blocks for the PR comments. Full detail is preserved; a
        detail too large for one comment is chunked (closing and reopening
        code fences) so it flows across continuation comments."""
        blocks = []
        for name, row in self._rows.items():
            if not row["detail"]:
                continue
            chunks = _chunk_markdown(row["detail"], DETAIL_CHUNK_LIMIT)
            for i, chunk in enumerate(chunks):
                part = f" (part {i + 1}/{len(chunks)})" if len(chunks) > 1 else ""
                blocks.append(self._wrap_details(name, row, chunk, part))
        return blocks

    def _check_detail_blocks(self) -> List[str]:
        """Detail blocks for the check run output, which cannot be split;
        each definition's detail is capped at max_detail_chars."""
        blocks = []
        for name, row in self._rows.items():
            if not row["detail"]:
                continue
            body = _fence_safe_truncate(row["detail"], self.max_detail_chars)
            blocks.append(self._wrap_details(name, row, body))
        return blocks

    def render_comment_bodies(self) -> List[str]:
        """Render the comment bodies: a primary comment with the summary table,
        plus continuation comments for detail blocks that do not fit."""
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
        """Render the check run output markdown (the job summary surface)."""
        body = "\n".join(
            [self._header(), self._table(), ""] + self._check_detail_blocks()
        )
        if len(body) > BODY_BUDGET:
            # keep the table; details are available in the PR comments
            body = "\n".join(
                [
                    self._header(),
                    self._table(),
                    "",
                    "_Detail sections omitted; body exceeded GitHub's size limit._",
                ]
            )
        return body


@HandlerRegistry.register("github")
class GithubHandler(BaseHandler):
    """Report plan progress and results to GitHub."""

    actions = [TerraformAction.PLAN, TerraformAction.INIT]
    config_model = GithubConfig
    _ready = False
    default_priority = {
        TerraformAction.PLAN: 90,
    }
    # soft ordering: run after openai when it is configured so its plan
    # summary is available in the shared results (absent handlers are ignored)
    dependencies = {
        TerraformAction.PLAN: {TerraformStage.POST: ["openai"]},
    }

    def __init__(self, config: GithubConfig) -> None:
        self.config = config
        self._app_state = None
        self._gh = None
        self._repo = None
        self._pr = None
        self._issue = None
        self._check = None
        self._def_checks: dict = {}
        self._def_concluded: set = set()
        self._report: Optional[GithubStatusReport] = None
        self._comments: List = []
        self._api_failures = 0

        missing = config.missing_settings()
        if missing:
            msg = f"github handler missing configuration: {'; '.join(missing)}"
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
        """Authenticate, create the check run, and seed the status comment."""
        if not self._ready:
            return
        if not getattr(terraform_options, "plan", True):
            log.debug("github handler: plan not requested, nothing to report")
            self._ready = False
            return
        try:
            self._connect()
            self._report = GithubStatusReport(
                deployment=deployment,
                marker=self.config.comment_marker,
                max_detail_chars=self.config.max_detail_chars,
                include_details=self.config.comment_details,
            )
            for defn in definitions.values():
                self._report.ensure(defn.name)
            # After the report exists: claiming comments needs the deployment to
            # scope the marker it matches on.
            self._claim_comments()

            head_sha = self._resolve_sha(working_dir)
            check_name = self.config.check_run_name or f"tfworker/{deployment}/plan"
            self._check = self._repo.create_check_run(
                name=check_name,
                head_sha=head_sha,
                status="in_progress",
                output={
                    "title": "Terraform plan in progress",
                    "summary": self._report.render_check_summary(),
                },
            )
            self._report.check_url = self._check_url(self._check)
            for defn in definitions.values():
                check = self._repo.create_check_run(
                    name=f"{check_name}: {defn.name}",
                    head_sha=head_sha,
                    status="queued",
                )
                self._def_checks[defn.name] = check
                self._report.set_url(defn.name, self._check_url(check))
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
        result: Union["TerraformResult", None] = None,
    ) -> Union[GithubResult, None]:
        if not self._ready or self._report is None:
            return None
        # INIT is only interesting when it fails: a prepare/init failure aborts
        # the run, and without a failed row the aborted run concludes "success"
        # with every definition skipped
        if action not in (TerraformAction.PLAN, TerraformAction.INIT):
            return None
        try:
            if stage == TerraformStage.PRE and action == TerraformAction.PLAN:
                self._report.mark(definition.name, "running")
                if definition.name in self._def_checks:
                    self._def_checks[definition.name].edit(status="in_progress")
                self._update_comments()
                return None
            if (
                stage == TerraformStage.POST
                and action == TerraformAction.PLAN
                and result is not None
            ):
                return self._post_plan(definition, result)
            if stage == TerraformStage.ERROR and result is not None:
                title = (
                    "Terraform init failed"
                    if action == TerraformAction.INIT
                    else "Terraform plan failed"
                )
                self._report.mark(
                    definition.name,
                    "failed",
                    detail=self._error_detail(result),
                )
                self._conclude_def_check(
                    definition.name,
                    "failure",
                    title,
                    self._error_detail(result),
                )
                self._update_comments()
                self._update_check("Terraform plan in progress")
        except Exception as e:
            self._record_api_failure(f"execute({definition.name})", e)
        return None

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
    def _post_plan(
        self, definition: "Definition", result: "TerraformResult"
    ) -> Union[GithubResult, None]:
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
                definition.name,
                "changes",
                plan_line=plan_line,
                detail=detail,
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
            check_run_url=self._check_url(self._check) if self._check else None,
            definition_check_url=self._check_url(def_check) if def_check else None,
            comment_url=self._comments[0].html_url if self._comments else None,
        )

    def _summary_for(self, definition: "Definition") -> Optional[str]:
        """Find the openai-generated summary for this definition, if any."""
        if definition.plan_file is None:
            return None
        try:
            handlers = self.app_state.handlers
            expected = str(
                Path(definition.plan_file)
                .with_suffix(".tfplan.json")
                .with_suffix(".summary.md")
            )
            for r in handlers.get_results(handler_name="openai"):
                if getattr(r, "task", None) == "summary" and r.file == expected:
                    return r.content
        except Exception as e:
            log.debug(f"github handler: unable to read openai results: {e}")
        return None

    @staticmethod
    def _plan_line(text: str) -> str:
        for line in text.splitlines():
            if line.startswith("Plan:"):
                return line.strip()
        return ""

    @staticmethod
    def _trimmed_plan(text: str, wrap_width: int = 0) -> str:
        """Trim plan output to the planned actions, fenced for markdown."""
        capture = False
        trimmed = ""
        for line in text.splitlines():
            if line.startswith("Terraform will perform the following actions:"):
                capture = True
            if line.startswith("Plan:"):
                trimmed += line + "\n"
                capture = False
            if capture:
                trimmed += line + "\n"
        if not trimmed:
            return ""
        return f"```diff\n{_hard_wrap(_diff_format(trimmed.rstrip()), wrap_width)}\n```"

    def _error_detail(self, result: "TerraformResult") -> str:
        text = strip_ansi(result.stderr_str or result.stdout_str)
        lines = [line for line in text.splitlines() if line.strip()]
        snippet = _hard_wrap("\n".join(lines[-20:]), self.config.wrap_width)
        return f"```\n{snippet}\n```" if snippet else ""

    ###########################################################################
    # github api plumbing
    ###########################################################################
    def _check_url(self, check) -> str:
        """Prefer the PR-scoped check view (.../pull/N/checks?check_run_id=X)
        over the generic runs page when a pull request is configured."""
        url = check.html_url
        if self.config.pull_request and url and "/runs/" in url:
            base = url.split("/runs/", 1)[0]
            return (
                f"{base}/pull/{self.config.pull_request}"
                f"/checks?check_run_id={check.id}"
            )
        return url

    def _connect(self) -> None:
        from github import Auth, GithubIntegration

        pem = self.config.private_key or Path(self.config.private_key_file).read_text()
        auth = Auth.AppAuth(self.config.app_id, pem)
        integration = GithubIntegration(auth=auth)
        owner, repo_name = self.config.repository.split("/", 1)
        installation_id = (
            self.config.installation_id
            or integration.get_repo_installation(owner, repo_name).id
        )
        self._gh = integration.get_github_for_installation(installation_id)
        self._repo = self._gh.get_repo(self.config.repository)
        if self.config.pull_request:
            self._pr = self._repo.get_pull(self.config.pull_request)
            self._issue = self._repo.get_issue(self.config.pull_request)

    def _resolve_sha(self, working_dir: str) -> str:
        if self.config.commit_sha:
            return self.config.commit_sha
        # prefer the PR head sha; GITHUB_SHA is a merge commit on pull_request
        # events and checks attached to it do not render on the PR
        if self._pr is not None:
            return self._pr.head.sha
        sha = os.environ.get("GITHUB_SHA")
        if sha:
            return sha
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=working_dir,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            .decode()
            .strip()
        )

    def _claim_comments(self) -> None:
        """Find this deployment's existing status comments (from a prior run on
        this PR) so they are edited in place rather than duplicated.

        Matching is scoped to the deployment. Several deployments can report to
        one PR -- a plan covering apps-ai-staging and apps-ai-prod, say -- and
        each owns its own comments; an unscoped match makes whichever runs last
        take over the first one's comments and delete its continuations.
        """
        if self._issue is None or self._report is None:
            return
        primary_marker = self._report.primary_marker
        part_prefix = f"{self._report.marker_prefix()} part="
        primary, parts = None, []
        for comment in self._issue.get_comments():
            # The marker is always the whole first line; match it exactly so one
            # deployment cannot claim another whose name it prefixes.
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
        # remove stale continuation comments from a previous, larger update
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
