"""GitHub handler for reporting plan status to a pull request.

Authenticates as a GitHub App and, for plan runs, maintains:

- a rollup check run on the PR head commit whose markdown output carries the
  per-definition job summary, plus one check run per definition (created
  queued, started at plan pre, and concluded from the plan result) so each
  definition reports its own status like Atlantis project checks
- a live status comment on the pull request (when one is configured),
  updated in place as each definition plans; overflow detail is split
  across additional comments when the body exceeds GitHub's size limit

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
import subprocess
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

# maximum consecutive API failures before the handler disables itself
MAX_API_FAILURES = 3


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
    max_detail_chars: int = Field(
        default=8000,
        description="Maximum characters of per-definition detail included in comments.",
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

    def __init__(self, deployment: str, marker: str, max_detail_chars: int) -> None:
        self.deployment = deployment
        self.marker = marker
        self.max_detail_chars = max_detail_chars
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
            row["detail"] = detail[: self.max_detail_chars] + (
                "\n\n_… truncated_" if len(detail) > self.max_detail_chars else ""
            )

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

    def _detail_blocks(self) -> List[str]:
        blocks = []
        for name, row in self._rows.items():
            if not row["detail"]:
                continue
            display = STATUS_DISPLAY.get(row["status"], row["status"])
            blocks.append(
                "<details>\n"
                f"<summary><code>{name}</code> — {display}"
                f"{' — ' + row['plan_line'] if row['plan_line'] else ''}</summary>\n\n"
                f"{row['detail']}\n\n"
                "</details>"
            )
        return blocks

    def render_comment_bodies(self) -> List[str]:
        """Render the comment bodies: a primary comment with the summary table,
        plus continuation comments for detail blocks that do not fit."""
        primary = "\n".join([self.primary_marker, self._header(), self._table(), ""])
        bodies = [primary]
        for block in self._detail_blocks():
            block = block[:BODY_BUDGET]
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
        body = "\n".join([self._header(), self._table(), ""] + self._detail_blocks())
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

    actions = [TerraformAction.PLAN]
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
            )
            for defn in definitions.values():
                self._report.ensure(defn.name)

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
        result: Union["TerraformResult", None] = None,
    ) -> Union[GithubResult, None]:
        if not self._ready or self._report is None:
            return None
        if action != TerraformAction.PLAN:
            return None
        try:
            if stage == TerraformStage.PRE:
                self._report.mark(definition.name, "running")
                if definition.name in self._def_checks:
                    self._def_checks[definition.name].edit(status="in_progress")
                self._update_comments()
                return None
            if stage == TerraformStage.POST and result is not None:
                return self._post_plan(definition, result)
            if stage == TerraformStage.ERROR and result is not None:
                self._report.mark(
                    definition.name,
                    "failed",
                    detail=self._error_detail(result),
                )
                self._conclude_def_check(
                    definition.name,
                    "failure",
                    "Terraform plan failed",
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
            detail = self._summary_for(definition) or self._trimmed_plan(text)
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
            check_run_url=self._check.html_url if self._check else None,
            definition_check_url=def_check.html_url if def_check else None,
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
    def _trimmed_plan(text: str) -> str:
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
        return f"```\n{trimmed}```"

    @staticmethod
    def _error_detail(result: "TerraformResult") -> str:
        text = strip_ansi(result.stderr_str or result.stdout_str)
        lines = [line for line in text.splitlines() if line.strip()]
        snippet = "\n".join(lines[-20:])
        return f"```\n{snippet}\n```" if snippet else ""

    ###########################################################################
    # github api plumbing
    ###########################################################################
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
            self._claim_comments()

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
        """Find existing status comments (from a prior run on this PR) so they
        are edited in place rather than duplicated."""
        report_prefix = f"<!-- {self.config.comment_marker}:"
        primary, parts = None, []
        for comment in self._issue.get_comments():
            body = comment.body or ""
            if not body.startswith(report_prefix):
                continue
            if "part=" in body.splitlines()[0]:
                parts.append(comment)
            else:
                primary = comment
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
            output={"title": title[:255], "summary": summary[:BODY_BUDGET]},
        )

    def _record_api_failure(self, context: str, exc: Exception) -> None:
        self._api_failures += 1
        log.error(f"github handler {context} failed: {exc}")
        if self._api_failures >= MAX_API_FAILURES:
            log.warn(
                f"github handler disabled after {self._api_failures} consecutive API failures"
            )
            self._ready = False
