from types import SimpleNamespace
from unittest import mock

import pytest

from tfworker.commands.terraform import TerraformResult
from tfworker.custom_types.terraform import TerraformAction, TerraformStage
from tfworker.exceptions import HandlerError
from tfworker.handlers.github import (
    GithubConfig,
    GithubHandler,
    GithubStatusReport,
    _chunk_markdown,
    _fence_safe_truncate,
)

GITHUB_ENV_VARS = [
    "GITHUB_REPOSITORY",
    "GITHUB_PULL_REQUEST",
    "PULL_REQUEST",
    "GITHUB_APP_ID",
    "GITHUB_APP_PRIVATE_KEY",
    "GITHUB_APP_PRIVATE_KEY_FILE",
    "GITHUB_APP_INSTALLATION_ID",
    "GITHUB_SHA",
]


@pytest.fixture(autouse=True)
def clean_github_env(monkeypatch):
    """Tests may run inside GitHub Actions; pin the environment."""
    for var in GITHUB_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def make_config(**overrides) -> GithubConfig:
    settings = {
        "repository": "myorg/myrepo",
        "app_id": "12345",
        "private_key": "---PEM---",
        "pull_request": 7,
    }
    settings.update(overrides)
    return GithubConfig.model_validate(settings)


def make_handler(config=None, with_pr=True) -> GithubHandler:
    handler = GithubHandler.__new__(GithubHandler)
    handler.config = config or make_config()
    handler._ready = True
    handler._app_state = mock.Mock()
    handler._app_state.handlers.get_results.return_value = []
    handler._gh = mock.Mock()
    handler._repo = mock.Mock()
    handler._pr = mock.Mock() if with_pr else None
    if with_pr:
        handler._issue = mock.Mock()
        handler._issue.create_comment.return_value.html_url = (
            "https://github.test/comment/1"
        )
    else:
        handler._issue = None
    handler._check = mock.Mock()
    handler._check.html_url = "https://github.test/check/1"
    handler._def_checks = {"mydef": mock.Mock()}
    handler._def_checks["mydef"].html_url = "https://github.test/check/mydef"
    handler._def_concluded = set()
    handler._comments = []
    handler._api_failures = 0
    handler._report = GithubStatusReport(
        deployment="dep", marker="tfworker-status", max_detail_chars=8000
    )
    handler._report.ensure("mydef")
    return handler


def make_definition(name="mydef", plan_file="/tmp/plans/mydef.tfplan"):
    definition = mock.Mock()
    definition.name = name
    definition.plan_file = plan_file
    return definition


PLAN_STDOUT = (
    b"Terraform used the selected providers\n"
    b"Terraform will perform the following actions:\n"
    b"  # null_resource.example will be created\n"
    b"Plan: 1 to add, 0 to change, 0 to destroy.\n"
)


class TestGithubConfig:
    def test_env_fallbacks(self, monkeypatch):
        monkeypatch.setenv("GITHUB_REPOSITORY", "envorg/envrepo")
        monkeypatch.setenv("GITHUB_APP_ID", "999")
        monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY", "envpem")
        monkeypatch.setenv("GITHUB_PULL_REQUEST", "42")
        monkeypatch.setenv("GITHUB_APP_INSTALLATION_ID", "777")

        config = GithubConfig()

        assert config.repository == "envorg/envrepo"
        assert config.app_id == "999"
        assert config.private_key == "envpem"
        assert config.pull_request == 42
        assert config.installation_id == 777
        assert config.missing_settings() == []

    def test_pull_request_env_fallback_alternate_name(self, monkeypatch):
        monkeypatch.setenv("PULL_REQUEST", "13")
        config = GithubConfig()
        assert config.pull_request == 13

    def test_explicit_config_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("GITHUB_REPOSITORY", "envorg/envrepo")
        config = make_config()
        assert config.repository == "myorg/myrepo"

    def test_empty_string_coerced_to_none(self):
        config = GithubConfig(pull_request="", repository="")
        assert config.pull_request is None
        assert config.repository is None

    def test_non_numeric_pull_request_env_ignored(self, monkeypatch):
        monkeypatch.setenv("GITHUB_PULL_REQUEST", "not-a-number")
        config = GithubConfig()
        assert config.pull_request is None

    def test_missing_settings(self):
        config = GithubConfig()
        missing = config.missing_settings()
        assert "repository" in missing
        assert "app_id" in missing
        assert "private_key or private_key_file" in missing

    def test_both_private_key_sources_rejected(self):
        config = make_config(private_key_file="/tmp/key.pem")
        assert config.missing_settings() == [
            "only one of private_key / private_key_file"
        ]


class TestGithubHandlerInit:
    def test_incomplete_config_not_required_disables(self):
        handler = GithubHandler(GithubConfig())
        assert handler.is_ready() is False

    def test_incomplete_config_required_raises(self):
        with pytest.raises(HandlerError):
            GithubHandler(GithubConfig(required=True))

    def test_complete_config_is_ready(self):
        handler = GithubHandler(make_config())
        assert handler.is_ready() is True


class TestGithubStatusReport:
    def make_report(self, **kwargs):
        settings = {
            "deployment": "dep",
            "marker": "tfworker-status",
            "max_detail_chars": 100,
        }
        settings.update(kwargs)
        return GithubStatusReport(**settings)

    def test_primary_body_has_marker_and_table(self):
        report = self.make_report()
        report.ensure("def1")
        report.mark("def2", "changes", plan_line="Plan: 1 to add")

        bodies = report.render_comment_bodies()

        assert len(bodies) == 1
        assert bodies[0].startswith("<!-- tfworker-status: dep -->")
        assert "| `def1` | ⏳ pending |" in bodies[0]
        assert "| `def2` | 📝 changes | Plan: 1 to add |" in bodies[0]

    def test_mark_preserves_full_detail(self):
        report = self.make_report(max_detail_chars=10)
        report.mark("def1", "changes", detail="x" * 50)
        assert report._rows["def1"]["detail"] == "x" * 50

    def test_check_summary_detail_truncated_fence_safe(self):
        report = self.make_report(max_detail_chars=30)
        report.mark(
            "def1", "changes", detail="```\nline one\nline two\nline three\n```"
        )
        summary = report.render_check_summary()
        assert "_… truncated_" in summary
        assert summary.count("```") % 2 == 0
        assert "line three" not in summary

    def test_details_excluded_from_comment_by_default(self):
        report = self.make_report()
        report.mark(
            "def1",
            "changes",
            plan_line="Plan: 1 to add",
            detail="```\nsecret plan\n```",
        )

        bodies = report.render_comment_bodies()

        assert len(bodies) == 1
        assert "secret plan" not in bodies[0]
        assert "| `def1` | 📝 changes | Plan: 1 to add |" in bodies[0]
        # the check run output still carries the detail
        assert "secret plan" in report.render_check_summary()

    def test_oversized_detail_splits_across_comments_without_loss(self):
        report = self.make_report(include_details=True)
        lines = "\n".join(f"resource line {i}" for i in range(100))
        report.mark(
            "def1",
            "changes",
            plan_line="Plan: 1 to add",
            detail=f"```hcl\n{lines}\n```",
        )

        with (
            mock.patch("tfworker.handlers.github.DETAIL_CHUNK_LIMIT", 400),
            mock.patch("tfworker.handlers.github.BODY_BUDGET", 900),
        ):
            bodies = report.render_comment_bodies()

        assert len(bodies) > 2
        joined = "".join(bodies)
        for i in range(100):
            assert f"resource line {i}" in joined
        for body in bodies:
            assert body.count("```") % 2 == 0
        assert "(part 1/" in joined

    def test_bodies_split_when_over_budget(self):
        report = self.make_report(max_detail_chars=400, include_details=True)
        for i in range(3):
            report.mark(f"def{i}", "changes", detail=f"detail-{i} " * 40)

        with mock.patch("tfworker.handlers.github.BODY_BUDGET", 600):
            bodies = report.render_comment_bodies()

        assert len(bodies) > 1
        assert bodies[0].startswith("<!-- tfworker-status: dep -->")
        for i, body in enumerate(bodies[1:], start=2):
            assert body.startswith(f"<!-- tfworker-status: dep part={i} -->")
        # the table stays in the primary comment only
        assert "| Definition | Status | Plan |" in bodies[0]
        assert all("| Definition | Status | Plan |" not in body for body in bodies[1:])

    def test_finalize_marks_unfinished_skipped(self):
        report = self.make_report()
        report.ensure("never_ran")
        report.mark("in_flight", "running")
        report.mark("done", "no_changes")

        report.finalize()

        assert report._rows["never_ran"]["status"] == "skipped"
        assert report._rows["in_flight"]["status"] == "skipped"
        assert report._rows["done"]["status"] == "no_changes"

    def test_conclusion(self):
        report = self.make_report()
        report.mark("ok", "no_changes")
        assert report.conclusion() == "success"
        report.mark("bad", "failed")
        assert report.conclusion() == "failure"

    def test_check_summary_over_budget_drops_details(self):
        report = self.make_report(max_detail_chars=500)
        report.mark("def1", "changes", detail="y" * 450)
        with mock.patch("tfworker.handlers.github.BODY_BUDGET", 300):
            summary = report.render_check_summary()
        assert "| `def1` |" in summary
        assert "y" * 100 not in summary
        assert "Detail sections omitted" in summary


class TestMarkdownHelpers:
    def test_chunk_markdown_under_limit_is_single_chunk(self):
        assert _chunk_markdown("short", 100) == ["short"]

    def test_chunk_markdown_balances_and_reopens_fences(self):
        text = "```hcl\n" + "\n".join(f"line {i}" for i in range(50)) + "\n```"
        chunks = _chunk_markdown(text, 100)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.count("```") % 2 == 0
        for chunk in chunks[1:]:
            assert chunk.startswith("```hcl")
        joined = "".join(chunks)
        for i in range(50):
            assert f"line {i}" in joined

    def test_fence_safe_truncate_noop_under_limit(self):
        assert _fence_safe_truncate("short", 100) == "short"

    def test_fence_safe_truncate_closes_open_fence(self):
        text = "```\n" + "x" * 200
        out = _fence_safe_truncate(text, 50)
        assert out.count("```") % 2 == 0
        assert "_… truncated_" in out


class TestGithubHandlerExecute:
    def test_non_plan_action_is_ignored(self):
        handler = make_handler()
        ret = handler.execute(
            action=TerraformAction.APPLY,
            stage=TerraformStage.POST,
            deployment="dep",
            definition=make_definition(),
            working_dir="/tmp",
            result=TerraformResult(0, b"", b""),
        )
        assert ret is None
        handler._issue.create_comment.assert_not_called()

    def test_pre_plan_marks_running(self):
        handler = make_handler()
        handler.execute(
            action=TerraformAction.PLAN,
            stage=TerraformStage.PRE,
            deployment="dep",
            definition=make_definition(),
            working_dir="/tmp",
        )
        assert handler._report._rows["mydef"]["status"] == "running"
        handler._issue.create_comment.assert_called_once()

    def test_post_plan_exit_code_1_marks_failed(self):
        handler = make_handler()
        handler.execute(
            action=TerraformAction.PLAN,
            stage=TerraformStage.POST,
            deployment="dep",
            definition=make_definition(),
            working_dir="/tmp",
            result=TerraformResult(1, b"", b"Error: something broke\n"),
        )
        row = handler._report._rows["mydef"]
        assert row["status"] == "failed"
        assert "something broke" in row["detail"]

    def test_post_plan_exit_code_0_marks_no_changes(self):
        handler = make_handler()
        handler.execute(
            action=TerraformAction.PLAN,
            stage=TerraformStage.POST,
            deployment="dep",
            definition=make_definition(),
            working_dir="/tmp",
            result=TerraformResult(0, b"No changes.", b""),
        )
        assert handler._report._rows["mydef"]["status"] == "no_changes"

    def test_post_plan_exit_code_2_marks_changes_and_returns_result(self):
        handler = make_handler()
        ret = handler.execute(
            action=TerraformAction.PLAN,
            stage=TerraformStage.POST,
            deployment="dep",
            definition=make_definition(),
            working_dir="/tmp",
            result=TerraformResult(2, PLAN_STDOUT, b""),
        )
        row = handler._report._rows["mydef"]
        assert row["status"] == "changes"
        assert row["plan_line"] == "Plan: 1 to add, 0 to change, 0 to destroy."
        assert "null_resource.example" in row["detail"]
        assert ret is not None
        assert ret.handler == "github"
        assert ret.definition == "mydef"
        handler._check.edit.assert_called_once()

    def test_post_plan_prefers_openai_summary(self):
        handler = make_handler()
        handler._app_state.handlers.get_results.return_value = [
            SimpleNamespace(
                task="summary",
                file="/tmp/plans/mydef.tfplan.summary.md",
                content="AI SUMMARY OF PLAN",
            ),
            SimpleNamespace(
                task="summary",
                file="/tmp/plans/otherdef.tfplan.summary.md",
                content="WRONG DEFINITION",
            ),
        ]
        handler.execute(
            action=TerraformAction.PLAN,
            stage=TerraformStage.POST,
            deployment="dep",
            definition=make_definition(),
            working_dir="/tmp",
            result=TerraformResult(2, PLAN_STDOUT, b""),
        )
        detail = handler._report._rows["mydef"]["detail"]
        assert "AI SUMMARY OF PLAN" in detail
        assert "WRONG DEFINITION" not in detail

    def test_error_stage_marks_failed(self):
        handler = make_handler()
        handler.execute(
            action=TerraformAction.PLAN,
            stage=TerraformStage.ERROR,
            deployment="dep",
            definition=make_definition(),
            working_dir="/tmp",
            result=TerraformResult(1, b"", b"Error: bad provider\n"),
        )
        row = handler._report._rows["mydef"]
        assert row["status"] == "failed"
        assert "bad provider" in row["detail"]

    def test_no_pr_skips_comments_but_updates_check(self):
        handler = make_handler(with_pr=False)
        handler.execute(
            action=TerraformAction.PLAN,
            stage=TerraformStage.POST,
            deployment="dep",
            definition=make_definition(),
            working_dir="/tmp",
            result=TerraformResult(2, PLAN_STDOUT, b""),
        )
        assert handler._comments == []
        handler._check.edit.assert_called_once()

    def test_api_errors_are_swallowed_and_disable_after_threshold(self):
        handler = make_handler()
        handler._issue.create_comment.side_effect = RuntimeError("boom")
        for _ in range(3):
            handler.execute(
                action=TerraformAction.PLAN,
                stage=TerraformStage.PRE,
                deployment="dep",
                definition=make_definition(),
                working_dir="/tmp",
            )
        assert handler.is_ready() is False

    def test_not_ready_handler_does_nothing(self):
        handler = make_handler()
        handler._ready = False
        ret = handler.execute(
            action=TerraformAction.PLAN,
            stage=TerraformStage.POST,
            deployment="dep",
            definition=make_definition(),
            working_dir="/tmp",
            result=TerraformResult(2, PLAN_STDOUT, b""),
        )
        assert ret is None
        handler._issue.create_comment.assert_not_called()


class TestPerDefinitionChecks:
    def test_pre_plan_starts_definition_check(self):
        handler = make_handler()
        handler.execute(
            action=TerraformAction.PLAN,
            stage=TerraformStage.PRE,
            deployment="dep",
            definition=make_definition(),
            working_dir="/tmp",
        )
        handler._def_checks["mydef"].edit.assert_called_once_with(status="in_progress")

    def test_post_plan_changes_concludes_success(self):
        handler = make_handler()
        ret = handler.execute(
            action=TerraformAction.PLAN,
            stage=TerraformStage.POST,
            deployment="dep",
            definition=make_definition(),
            working_dir="/tmp",
            result=TerraformResult(2, PLAN_STDOUT, b""),
        )
        kwargs = handler._def_checks["mydef"].edit.call_args.kwargs
        assert kwargs["status"] == "completed"
        assert kwargs["conclusion"] == "success"
        assert kwargs["output"]["title"] == "Plan: 1 to add, 0 to change, 0 to destroy."
        assert "null_resource.example" in kwargs["output"]["summary"]
        assert ret.definition_check_url == "https://github.test/check/mydef"

    def test_post_plan_no_changes_concludes_success(self):
        handler = make_handler()
        handler.execute(
            action=TerraformAction.PLAN,
            stage=TerraformStage.POST,
            deployment="dep",
            definition=make_definition(),
            working_dir="/tmp",
            result=TerraformResult(0, b"No changes.", b""),
        )
        kwargs = handler._def_checks["mydef"].edit.call_args.kwargs
        assert kwargs["conclusion"] == "success"
        assert kwargs["output"]["title"] == "No changes."

    def test_post_plan_failure_concludes_failure(self):
        handler = make_handler()
        handler.execute(
            action=TerraformAction.PLAN,
            stage=TerraformStage.POST,
            deployment="dep",
            definition=make_definition(),
            working_dir="/tmp",
            result=TerraformResult(1, b"", b"Error: broken\n"),
        )
        kwargs = handler._def_checks["mydef"].edit.call_args.kwargs
        assert kwargs["conclusion"] == "failure"
        assert "broken" in kwargs["output"]["summary"]

    def test_error_stage_concludes_failure(self):
        handler = make_handler()
        handler.execute(
            action=TerraformAction.PLAN,
            stage=TerraformStage.ERROR,
            deployment="dep",
            definition=make_definition(),
            working_dir="/tmp",
            result=TerraformResult(1, b"", b"Error: bad provider\n"),
        )
        kwargs = handler._def_checks["mydef"].edit.call_args.kwargs
        assert kwargs["conclusion"] == "failure"

    def test_definition_check_concluded_only_once(self):
        handler = make_handler()
        handler._conclude_def_check("mydef", "failure", "failed")
        handler._conclude_def_check("mydef", "skipped", "skipped")
        handler._def_checks["mydef"].edit.assert_called_once()

    def test_teardown_skips_unconcluded_definition_checks(self):
        handler = make_handler()
        handler._def_checks["never_ran"] = mock.Mock()
        handler._report.ensure("never_ran")
        handler._conclude_def_check("mydef", "success", "done")

        handler.teardown("dep", "/tmp")

        kwargs = handler._def_checks["never_ran"].edit.call_args.kwargs
        assert kwargs["conclusion"] == "skipped"
        # already-concluded checks are not edited again
        handler._def_checks["mydef"].edit.assert_called_once()

    def test_table_links_definition_to_its_check(self):
        report = GithubStatusReport(
            deployment="dep", marker="tfworker-status", max_detail_chars=100
        )
        report.ensure("linked")
        report.set_url("linked", "https://github.test/check/linked")
        report.ensure("unlinked")

        body = report.render_comment_bodies()[0]

        assert "| [`linked`](https://github.test/check/linked) |" in body
        assert "| `unlinked` |" in body


class TestCommentManagement:
    def test_claim_comments_orders_primary_first(self):
        handler = make_handler()
        part = mock.Mock(body="<!-- tfworker-status: dep part=2 -->\nmore")
        primary = mock.Mock(body="<!-- tfworker-status: dep -->\ntable")
        unrelated = mock.Mock(body="just a human comment")
        handler._issue.get_comments.return_value = [part, unrelated, primary]

        handler._claim_comments()

        assert handler._comments == [primary, part]

    def test_update_comments_edits_creates_and_deletes(self):
        handler = make_handler()
        existing_primary = mock.Mock(body="old primary")
        stale_part = mock.Mock(body="old part 2")
        stale_part_2 = mock.Mock(body="old part 3")
        handler._comments = [existing_primary, stale_part, stale_part_2]
        handler._report.mark("mydef", "changes", detail="small detail")

        handler._update_comments()

        existing_primary.edit.assert_called_once()
        # single body now: both stale continuation comments removed
        stale_part.delete.assert_called_once()
        stale_part_2.delete.assert_called_once()
        assert len(handler._comments) == 1

    def test_update_comments_skips_edit_when_unchanged(self):
        handler = make_handler()
        body = handler._report.render_comment_bodies()[0]
        existing = mock.Mock(body=body)
        handler._comments = [existing]

        handler._update_comments()

        existing.edit.assert_not_called()


class TestGithubHandlerTeardown:
    def test_teardown_concludes_check_run_failure(self):
        handler = make_handler()
        handler._report.mark("mydef", "failed", detail="boom")

        handler.teardown("dep", "/tmp")

        kwargs = handler._check.edit.call_args.kwargs
        assert kwargs["status"] == "completed"
        assert kwargs["conclusion"] == "failure"

    def test_teardown_concludes_success_and_skips_pending(self):
        handler = make_handler()
        handler._report.mark("mydef", "no_changes")
        handler._report.ensure("never_ran")

        handler.teardown("dep", "/tmp")

        kwargs = handler._check.edit.call_args.kwargs
        assert kwargs["conclusion"] == "success"
        assert handler._report._rows["never_ran"]["status"] == "skipped"

    def test_teardown_without_setup_is_noop(self):
        handler = make_handler()
        handler._report = None
        handler.teardown("dep", "/tmp")
        handler._check.edit.assert_not_called()
