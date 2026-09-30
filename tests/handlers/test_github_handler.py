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
    _diff_format,
    _fence_safe_truncate,
    _hard_wrap,
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
    "GITHUB_CHECK_DETAILS_URL",
    "GITHUB_CHECK_DEFINITION_DETAILS_URL",
    "GITHUB_CHECK_LOGS_URL",
    "GITHUB_CHECK_DEFINITION_LOGS_URL",
    "GITHUB_CHECK_LOGS_LABEL",
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


def make_handler(config=None, with_pr=True, clock=None) -> GithubHandler:
    """A handler as setup() leaves it, with the GitHub objects mocked."""
    kwargs = {"clock": clock} if clock else {}
    handler = GithubHandler(config or make_config(), **kwargs)
    handler._app_state = mock.Mock()
    handler._app_state.handlers.get_results.return_value = []
    handler._repo = mock.Mock()
    if with_pr:
        handler._pr = mock.Mock()
        handler._issue = mock.Mock()
        handler._issue.create_comment.return_value.html_url = (
            "https://github.test/comment/1"
        )
    handler._check = mock.Mock(html_url="https://github.test/check/1")
    handler._def_checks = {
        "mydef": mock.Mock(html_url="https://github.test/check/mydef")
    }
    handler._deployment = "dep"
    handler._head_sha = "abc123"
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
        assert config.settings_errors() == []

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

    def test_non_numeric_installation_id_env_ignored(self, monkeypatch):
        monkeypatch.setenv("GITHUB_APP_INSTALLATION_ID", "abc")
        config = GithubConfig()
        assert config.installation_id is None

    def test_empty_render_coercion_leaves_required_strings_alone(self):
        config = make_config(comment_marker="none")
        assert config.comment_marker == "none"

    def test_missing_settings(self):
        errors = GithubConfig().settings_errors()
        assert errors == [
            "missing repository",
            "missing app_id",
            "missing private_key or private_key_file",
        ]

    def test_both_private_key_sources_rejected(self):
        config = make_config(private_key_file="/tmp/key.pem")
        assert config.settings_errors() == [
            "private_key and private_key_file are mutually exclusive"
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

    def test_run_id_is_reported_when_set(self):
        """The run id keys the stored plans, so an apply can be requested for them."""
        report = self.make_report(run_id="run-1234")
        report.ensure("def1")

        bodies = report.render_comment_bodies()

        assert "Run `run-1234`" in bodies[0]
        assert "Run `run-1234`" in report.render_check_summary()

    def test_run_id_omitted_when_absent(self):
        report = self.make_report()
        report.ensure("def1")
        assert "Run `" not in report.render_comment_bodies()[0]

    def test_mark_preserves_full_detail(self):
        report = self.make_report(max_detail_chars=10)
        report.mark("def1", "changes", detail="x" * 50)
        assert report.rows["def1"].detail == "x" * 50

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

        assert report.rows["never_ran"].status == "skipped"
        assert report.rows["in_flight"].status == "skipped"
        assert report.rows["done"].status == "no_changes"

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


class TestHardWrap:
    def test_short_lines_untouched(self):
        text = 'resource "aws_s3_bucket" "b" {\n  bucket = "short"\n}'
        assert _hard_wrap(text, 120) == text

    def test_long_line_wrapped_with_indent(self):
        line = "      query = " + "sum:metric{tag} " * 20
        out = _hard_wrap(line, 80)
        lines = out.splitlines()
        assert len(lines) > 1
        assert all(len(ln) <= 80 for ln in lines)
        assert all(ln.startswith("      ") for ln in lines)
        assert lines[1].startswith("          ")

    def test_zero_width_disables_wrapping(self):
        line = "x" * 300
        assert _hard_wrap(line, 0) == line

    def test_unbreakable_token_left_intact(self):
        arn = "arn:aws:iam::123456789012:role/" + "a" * 150
        assert arn in _hard_wrap(f"role = {arn}", 80)


class TestDiffFormat:
    def test_markers_hoisted_to_column_zero(self):
        text = (
            '  + resource "a" "b" {\n'
            '  - resource "c" "d" {\n'
            '  ~ resource "e" "f" {\n'
            '-/+ resource "g" "h" {\n'
            "  # comment line\n"
            "      + attr = 1\n"
        )
        out = _diff_format(text).splitlines()
        assert out[0].startswith("+  ")
        assert out[1].startswith("-  ")
        assert out[2].startswith("!  ")
        # -/+ has no leading indent, left alone
        assert out[3].startswith("-/+")
        assert out[4] == "  # comment line"
        assert out[5].startswith("+      ")

    def test_replace_marker_with_indent_becomes_bang(self):
        assert _diff_format('  -/+ resource "g" "h" {').startswith("!  ")


class TestPlanParsing:
    def test_trimmed_plan_wraps_long_lines(self):
        text = (
            "Terraform will perform the following actions:\n"
            + "  attribute = "
            + "value " * 40
            + "\n"
            + "Plan: 1 to add, 0 to change, 0 to destroy.\n"
        )
        out = GithubHandler._trimmed_plan(text, wrap_width=60)
        assert all(
            len(line) <= 60 for line in out.splitlines() if not line.startswith("```")
        )
        assert out.count("value") == 40

    def test_output_only_plan_is_captured(self):
        """A plan that only moves outputs has neither the resource-actions
        heading nor a Plan: line, and used to render as nothing at all."""
        text = (
            "Terraform used the selected providers to generate the following\n"
            "execution plan.\n"
            "\n"
            "Changes to Outputs:\n"
            '  + oidc_provider_arns = "known after apply"\n'
            "\n"
            "You can apply this plan to save these new output values to the\n"
            "Terraform state, without changing any real infrastructure.\n"
        )
        out = GithubHandler._trimmed_plan(text)
        assert out.startswith("```diff\n")
        assert "Changes to Outputs:" in out
        assert "oidc_provider_arns" in out
        assert "without changing any real infrastructure" not in out

    def test_output_changes_after_plan_line_are_kept(self):
        """Output changes follow the Plan: line; capture used to stop there."""
        text = (
            "Terraform will perform the following actions:\n"
            '  + resource "null_resource" "example" {\n'
            "Plan: 1 to add, 0 to change, 0 to destroy.\n"
            "\n"
            "Changes to Outputs:\n"
            '  + example = "value"\n'
        )
        out = GithubHandler._trimmed_plan(text)
        assert "Plan: 1 to add" in out
        assert "Changes to Outputs:" in out
        assert "example" in out

    def test_trimmed_plan_stops_at_saved_plan_footer(self):
        ruler = "\u2500" * 20
        text = (
            "Terraform will perform the following actions:\n"
            '  + resource "null_resource" "example" {\n'
            "Plan: 1 to add, 0 to change, 0 to destroy.\n"
            "\n"
            f"{ruler}\n"
            "\n"
            "Saved the plan to: plan.tfplan\n"
            "\n"
            "To perform exactly these actions, run the following command to apply:\n"
            '    terraform apply "plan.tfplan"\n'
        )
        out = GithubHandler._trimmed_plan(text)
        assert "Plan: 1 to add" in out
        assert "Saved the plan to" not in out
        assert "terraform apply" not in out
        assert "\u2500" not in out

    def test_plan_line_falls_back_for_output_only(self):
        text = 'Changes to Outputs:\n  + example = "value"\n'
        assert GithubHandler._plan_line(text) == "Output changes only."

    def test_plan_line_prefers_the_plan_summary(self):
        text = (
            "Plan: 1 to add, 0 to change, 0 to destroy.\n" "\n" "Changes to Outputs:\n"
        )
        assert (
            GithubHandler._plan_line(text)
            == "Plan: 1 to add, 0 to change, 0 to destroy."
        )

    def test_plan_line_empty_when_no_changes(self):
        assert (
            GithubHandler._plan_line("No changes. Your infrastructure matches.\n") == ""
        )

    def test_trimmed_plan_uses_diff_fence(self):
        text = (
            "Terraform will perform the following actions:\n"
            '  + resource "null_resource" "example" {\n'
            "Plan: 1 to add, 0 to change, 0 to destroy.\n"
        )
        out = GithubHandler._trimmed_plan(text)
        assert out.startswith("```diff\n")
        assert '\n+   resource "null_resource" "example" {\n' in out


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
    def test_unhandled_action_is_ignored(self):
        handler = make_handler()
        ret = handler.execute(
            action=TerraformAction.DESTROY,
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
        assert handler._report.rows["mydef"].status == "running"
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
        row = handler._report.rows["mydef"]
        assert row.status == "failed"
        assert "something broke" in row.detail

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
        assert handler._report.rows["mydef"].status == "no_changes"

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
        row = handler._report.rows["mydef"]
        assert row.status == "changes"
        assert row.plan_line == "Plan: 1 to add, 0 to change, 0 to destroy."
        assert "null_resource.example" in row.detail
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
        detail = handler._report.rows["mydef"].detail
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
        row = handler._report.rows["mydef"]
        assert row.status == "failed"
        assert "bad provider" in row.detail

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
        handler._def_checks["mydef"].edit.side_effect = RuntimeError("boom")
        for _ in range(3):
            handler.execute(
                action=TerraformAction.PLAN,
                stage=TerraformStage.PRE,
                deployment="dep",
                definition=make_definition(),
                working_dir="/tmp",
            )
        assert handler.is_ready() is False

    def test_comment_failures_disable_only_the_comment(self):
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
        assert handler._comments_enabled is False
        assert handler.is_ready() is True
        # check runs keep reporting after the comment gives up
        handler._def_checks["mydef"].edit.reset_mock()
        handler._issue.create_comment.reset_mock()
        handler.execute(
            action=TerraformAction.PLAN,
            stage=TerraformStage.POST,
            deployment="dep",
            definition=make_definition(),
            working_dir="/tmp",
            result=TerraformResult(0, b"No changes.", b""),
        )
        handler._def_checks["mydef"].edit.assert_called_once()
        handler._issue.create_comment.assert_not_called()

    def test_comment_failure_count_resets_on_success(self):
        handler = make_handler()
        handler._issue.create_comment.side_effect = [
            RuntimeError("boom"),
            RuntimeError("boom"),
            mock.Mock(),
        ]
        for _ in range(3):
            handler._update_comments()
        assert handler._comment_failures == 0
        assert handler._comments_enabled is True

    def test_success_resets_the_failure_count(self):
        handler = make_handler()
        handler._def_checks["mydef"].edit.side_effect = [
            RuntimeError("boom"),
            RuntimeError("boom"),
            None,
            RuntimeError("boom"),
        ]
        for _ in range(4):
            handler.execute(
                action=TerraformAction.PLAN,
                stage=TerraformStage.PRE,
                deployment="dep",
                definition=make_definition(),
                working_dir="/tmp",
            )
        assert handler._api_failures == 1
        assert handler.is_ready() is True

    def test_post_without_result_is_ignored(self):
        handler = make_handler()
        ret = handler.execute(
            action=TerraformAction.PLAN,
            stage=TerraformStage.POST,
            deployment="dep",
            definition=make_definition(),
            working_dir="/tmp",
        )
        assert ret is None
        handler._def_checks["mydef"].edit.assert_not_called()

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
        assert handler._report.rows["never_ran"].status == "skipped"

    def test_teardown_without_setup_is_noop(self):
        handler = make_handler()
        handler._report = None
        handler.teardown("dep", "/tmp")
        handler._check.edit.assert_not_called()


class TestSetupRunId:
    """The rollup check run carries the run id, which is how a tool can ask for
    an apply of the plans this run stored."""

    def _setup_handler(self, run_id):
        handler = make_handler()
        handler._app_state.root_options.run_id = run_id
        handler._connect = mock.Mock()
        handler._claim_comments = mock.Mock()
        handler._resolve_sha = mock.Mock(return_value="abc123")
        handler._update_comments = mock.Mock()
        definitions = {"mydef": make_definition()}
        handler.setup(
            "dep",
            mock.Mock(values=lambda: definitions.values()),
            "/tmp",
            mock.Mock(plan=True),
        )
        return handler

    def test_external_id_is_the_run_id(self):
        handler = self._setup_handler("run-1234")
        rollup = handler._repo.create_check_run.call_args_list[0].kwargs
        assert rollup["external_id"] == "run-1234"
        assert rollup["name"] == "tfworker/dep/plan"
        assert "Run `run-1234`" in rollup["output"]["summary"]

    def test_external_id_omitted_without_a_run_id(self):
        handler = self._setup_handler(None)
        rollup = handler._repo.create_check_run.call_args_list[0].kwargs
        assert "external_id" not in rollup

    def test_definition_checks_carry_the_run_id(self):
        """So an apply can match them to its run when rebuilding the comment."""
        handler = self._setup_handler("run-1234")
        per_def = handler._repo.create_check_run.call_args_list[1].kwargs
        assert per_def["name"] == "tfworker/dep/plan: mydef"
        assert per_def["external_id"] == "run-1234"

    def test_links_use_the_check_run_html_url(self):
        handler = make_handler()
        handler._app_state.root_options.run_id = None
        handler._connect = mock.Mock()
        handler._claim_comments = mock.Mock()
        handler._resolve_sha = mock.Mock(return_value="abc123")
        handler._update_comments = mock.Mock()
        handler._repo.create_check_run.side_effect = [
            mock.Mock(html_url="https://github.com/o/r/runs/1"),
            mock.Mock(html_url="https://github.com/o/r/runs/2"),
        ]
        definitions = {"mydef": make_definition()}

        handler.setup(
            "dep",
            mock.Mock(values=lambda: definitions.values()),
            "/tmp",
            mock.Mock(plan=True),
        )

        assert handler._report.check_url == "https://github.com/o/r/runs/1"
        assert handler._report.rows["mydef"].url == "https://github.com/o/r/runs/2"


class TestClaimComments:
    """Claiming is scoped to the deployment: several deployments report to one
    PR and each owns its own comments."""

    @staticmethod
    def _comment(body):
        comment = mock.Mock()
        comment.body = body
        return comment

    def _issue_with(self, *bodies):
        issue = mock.Mock()
        issue.get_comments.return_value = [self._comment(b) for b in bodies]
        return issue

    def test_claims_only_this_deployments_comments(self):
        handler = make_handler()
        handler._report = GithubStatusReport(
            deployment="apps-ai-prod", marker="tfworker-status", max_detail_chars=8000
        )
        mine = "<!-- tfworker-status: apps-ai-prod -->\n## status"
        mine_part = "<!-- tfworker-status: apps-ai-prod part=2 -->\ndetail"
        theirs = "<!-- tfworker-status: apps-ai-staging -->\n## status"
        theirs_part = "<!-- tfworker-status: apps-ai-staging part=2 -->\ndetail"
        handler._issue = self._issue_with(theirs, mine, theirs_part, mine_part)

        handler._claim_comments()

        assert [c.body for c in handler._comments] == [mine, mine_part]

    def test_does_not_claim_a_deployment_it_merely_prefixes(self):
        handler = make_handler()
        handler._report = GithubStatusReport(
            deployment="apps-ai", marker="tfworker-status", max_detail_chars=8000
        )
        handler._issue = self._issue_with(
            "<!-- tfworker-status: apps-ai-staging -->\n## status"
        )

        handler._claim_comments()

        assert handler._comments == []

    def test_ignores_unrelated_comments(self):
        handler = make_handler()
        handler._issue = self._issue_with("just a human comment", "")

        handler._claim_comments()

        assert handler._comments == []

    def test_noop_without_report_or_issue(self):
        handler = make_handler()
        handler._report = None
        handler._claim_comments()
        assert handler._comments == []


class TestResolveSha:
    def test_commit_sha_config_wins(self, monkeypatch):
        handler = make_handler(config=make_config(commit_sha="configsha"))
        monkeypatch.setenv("GITHUB_SHA", "envsha")
        assert handler._resolve_sha() == "configsha"

    def test_pr_head_preferred_over_github_sha(self, monkeypatch):
        handler = make_handler()
        handler._pr.head.sha = "headsha"
        monkeypatch.setenv("GITHUB_SHA", "envsha")
        assert handler._resolve_sha() == "headsha"

    def test_github_sha_without_a_pr(self, monkeypatch):
        handler = make_handler(with_pr=False)
        monkeypatch.setenv("GITHUB_SHA", "envsha")
        assert handler._resolve_sha() == "envsha"

    def test_no_commit_raises(self):
        handler = make_handler(with_pr=False)
        with pytest.raises(HandlerError):
            handler._resolve_sha()


class TestConnect:
    @pytest.fixture
    def integration(self):
        with (
            mock.patch("github.GithubIntegration") as integration_cls,
            mock.patch("github.Auth.AppAuth") as app_auth,
        ):
            integration = integration_cls.return_value
            integration.get_repo_installation.return_value.id = 55
            yield SimpleNamespace(integration=integration, app_auth=app_auth)

    def test_discovers_installation_and_loads_pull_request(self, integration):
        handler = GithubHandler(make_config())

        handler._connect()

        integration.app_auth.assert_called_once_with("12345", "---PEM---")
        integration.integration.get_repo_installation.assert_called_once_with(
            "myorg", "myrepo"
        )
        gh = integration.integration.get_github_for_installation
        gh.assert_called_once_with(55)
        repo = gh.return_value.get_repo.return_value
        assert handler._repo is repo
        repo.get_pull.assert_called_once_with(7)
        repo.get_issue.assert_called_once_with(7)

    def test_configured_installation_and_key_file(self, integration, tmp_path):
        key = tmp_path / "app.pem"
        key.write_text("FILE PEM")
        handler = GithubHandler(
            make_config(
                private_key=None,
                private_key_file=str(key),
                installation_id=99,
                pull_request=None,
            )
        )

        handler._connect()

        integration.app_auth.assert_called_once_with("12345", "FILE PEM")
        integration.integration.get_repo_installation.assert_not_called()
        integration.integration.get_github_for_installation.assert_called_once_with(99)
        assert handler._pr is None
        assert handler._issue is None


class TestSetupFailure:
    def _setup(self, handler):
        handler._connect = mock.Mock(side_effect=RuntimeError("bad credentials"))
        handler.setup("dep", mock.Mock(values=lambda: []), "/tmp", mock.Mock(plan=True))

    def test_failure_disables_the_handler(self):
        handler = make_handler()
        self._setup(handler)
        assert handler.is_ready() is False

    def test_failure_raises_when_required(self):
        handler = make_handler(config=make_config(required=True))
        with pytest.raises(HandlerError, match="bad credentials"):
            self._setup(handler)

    def test_neither_plan_nor_apply_disables(self):
        handler = make_handler()
        handler._connect = mock.Mock()
        handler.setup(
            "dep", mock.Mock(), "/tmp", SimpleNamespace(plan=False, apply=False)
        )
        assert handler.is_ready() is False
        handler._connect.assert_not_called()


class TestHandlerHelpers:
    def test_run_id_without_a_click_context(self):
        handler = GithubHandler(make_config())
        assert handler._run_id() is None

    def test_summary_lookup_errors_fall_back_to_the_plan(self):
        handler = make_handler()
        handler._app_state.handlers.get_results.side_effect = RuntimeError("boom")
        assert handler._summary_for(make_definition()) is None

    def test_summary_lookup_without_a_plan_file(self):
        handler = make_handler()
        assert handler._summary_for(make_definition(plan_file=None)) is None
        handler._app_state.handlers.get_results.assert_not_called()


class TestSetupModes:
    def _setup(self, handler, **options):
        handler._app_state.root_options.run_id = "run-1234"
        handler._connect = mock.Mock()
        handler._resolve_sha = mock.Mock(return_value="abc123")
        handler._load_run_state = mock.Mock()
        definitions = {"mydef": make_definition()}
        handler.setup(
            "dep",
            mock.Mock(values=lambda: definitions.values()),
            "/tmp",
            SimpleNamespace(**options),
        )

    def test_apply_only_rebuilds_state_and_creates_no_plan_checks(self):
        handler = make_handler()
        self._setup(handler, plan=False, apply=True)
        assert handler.is_ready() is True
        assert handler._planning is False
        handler._resolve_sha.assert_called_once_with(for_apply=True)
        handler._load_run_state.assert_called_once_with("run-1234")
        handler._repo.create_check_run.assert_not_called()

    def test_claim_failure_disables_comments_not_the_handler(self):
        handler = make_handler()
        handler._issue.get_comments.side_effect = RuntimeError("boom")
        self._setup(handler, plan=True, apply=False)
        assert handler.is_ready() is True
        assert handler._comments_enabled is False
        handler._issue.create_comment.assert_not_called()
        handler._repo.create_check_run.assert_called()


class TestResolveShaForApply:
    def test_apply_prefers_the_requested_commit_over_pr_head(self, monkeypatch):
        handler = make_handler()
        handler._pr.head.sha = "headsha"
        monkeypatch.setenv("GITHUB_SHA", "plannedsha")
        assert handler._resolve_sha(for_apply=True) == "plannedsha"

    def test_apply_falls_back_to_pr_head(self):
        handler = make_handler()
        handler._pr.head.sha = "headsha"
        assert handler._resolve_sha(for_apply=True) == "headsha"

    def test_apply_without_a_commit_raises(self):
        handler = make_handler(with_pr=False)
        with pytest.raises(HandlerError):
            handler._resolve_sha(for_apply=True)

    def test_commit_sha_config_wins(self, monkeypatch):
        handler = make_handler(config=make_config(commit_sha="configsha"))
        monkeypatch.setenv("GITHUB_SHA", "plannedsha")
        assert handler._resolve_sha(for_apply=True) == "configsha"


APPLY_STDOUT = (
    b"null_resource.example: Creating...\n"
    b"null_resource.example: Creation complete after 0s [id=123]\n"
    b"\n"
    b"\n"
    b"Apply complete! Resources: 1 added, 0 changed, 0 destroyed.\n"
)


class TestApply:
    def apply_handler(self, planning=True):
        handler = make_handler()
        handler._planning = planning
        handler._app_state.root_options.run_id = "run-1234"
        rollup = mock.Mock(html_url="https://github.test/apply")
        per_def = mock.Mock(html_url="https://github.test/apply/mydef")
        handler._repo.create_check_run.side_effect = [rollup, per_def]
        return handler, rollup, per_def

    def run(self, handler, stage, result=None):
        return handler.execute(
            action=TerraformAction.APPLY,
            stage=stage,
            deployment="dep",
            definition=make_definition(),
            working_dir="/tmp",
            result=result,
        )

    def test_pre_apply_creates_rollup_and_definition_checks(self):
        handler, rollup, per_def = self.apply_handler()
        handler._report.mark("mydef", "changes", plan_line="Plan: 1 to add")

        self.run(handler, TerraformStage.PRE)

        calls = handler._repo.create_check_run.call_args_list
        assert calls[0].kwargs["name"] == "tfworker/dep/apply"
        assert calls[0].kwargs["external_id"] == "run-1234"
        assert calls[1].kwargs["name"] == "tfworker/dep/apply: mydef"
        assert calls[1].kwargs["head_sha"] == "abc123"
        assert calls[1].kwargs["status"] == "in_progress"
        row = handler._report.rows["mydef"]
        assert row.apply_status == "running"
        assert row.apply_url == "https://github.test/apply/mydef"
        body = handler._issue.create_comment.call_args.args[0]
        assert "| Apply |" in body
        assert "[🚀 applying](https://github.test/apply/mydef)" in body
        assert "[View apply check run](https://github.test/apply)" in body

    def test_rollup_created_once(self):
        handler, rollup, per_def = self.apply_handler()
        other = mock.Mock(html_url="https://github.test/apply/other")
        handler._repo.create_check_run.side_effect = [rollup, per_def, other]
        self.run(handler, TerraformStage.PRE)
        handler.execute(
            action=TerraformAction.APPLY,
            stage=TerraformStage.PRE,
            deployment="dep",
            definition=make_definition(name="other"),
            working_dir="/tmp",
        )
        names = [
            c.kwargs["name"] for c in handler._repo.create_check_run.call_args_list
        ]
        assert names.count("tfworker/dep/apply") == 1

    def test_post_apply_concludes_success_and_keeps_applied_plan(self):
        handler, rollup, per_def = self.apply_handler()
        handler._report.mark("mydef", "changes", detail="```diff\n+ planned\n```")
        self.run(handler, TerraformStage.PRE)

        ret = self.run(
            handler, TerraformStage.POST, TerraformResult(0, APPLY_STDOUT, b"")
        )

        kwargs = per_def.edit.call_args.kwargs
        assert kwargs["conclusion"] == "success"
        assert kwargs["output"]["title"] == (
            "Apply complete! Resources: 1 added, 0 changed, 0 destroyed."
        )
        assert "Creation complete" in kwargs["output"]["summary"]
        assert "### Applied plan" in kwargs["output"]["text"]
        assert "+ planned" in kwargs["output"]["text"]
        assert handler._report.rows["mydef"].apply_status == "applied"
        assert ret.action == TerraformAction.APPLY
        assert ret.check_run_url == "https://github.test/apply"
        assert ret.definition_check_url == "https://github.test/apply/mydef"

    def test_error_stage_concludes_failure(self):
        handler, rollup, per_def = self.apply_handler()
        self.run(handler, TerraformStage.PRE)

        self.run(
            handler,
            TerraformStage.ERROR,
            TerraformResult(1, b"", b"Error: apply broke\n"),
        )

        kwargs = per_def.edit.call_args.kwargs
        assert kwargs["conclusion"] == "failure"
        assert "apply broke" in kwargs["output"]["summary"]
        assert "text" not in kwargs["output"]
        assert handler._report.rows["mydef"].apply_status == "failed"

    def test_pre_apply_without_plan_state_reports_changes(self):
        """An apply-only run that could not read the plan back still knows a
        definition with a stored plan planned changes."""
        handler, rollup, per_def = self.apply_handler(planning=False)
        self.run(handler, TerraformStage.PRE)
        assert handler._report.rows["mydef"].status == "changes"

    def test_api_failure_is_recorded(self):
        handler, rollup, per_def = self.apply_handler()
        handler._repo.create_check_run.side_effect = RuntimeError("boom")
        assert self.run(handler, TerraformStage.PRE) is None
        assert handler._api_failures == 1

    def test_post_without_result_is_ignored(self):
        handler, rollup, per_def = self.apply_handler()
        assert self.run(handler, TerraformStage.POST) is None

    def test_teardown_concludes_apply_rollup(self):
        handler, rollup, per_def = self.apply_handler()
        handler._report.mark("mydef", "changes")
        handler._report.mark("never_applied", "changes")
        handler._report.mark("unchanged", "no_changes")
        self.run(handler, TerraformStage.PRE)
        self.run(handler, TerraformStage.ERROR, TerraformResult(1, b"", b"Error: x\n"))

        handler.teardown("dep", "/tmp")

        kwargs = rollup.edit.call_args.kwargs
        assert kwargs["status"] == "completed"
        assert kwargs["conclusion"] == "failure"
        rows = handler._report.rows
        assert rows["never_applied"].apply_status == "not_applied"
        assert rows["unchanged"].apply_status is None

    def test_teardown_skips_unconcluded_apply_checks(self):
        handler, rollup, per_def = self.apply_handler()
        self.run(handler, TerraformStage.PRE)

        handler.teardown("dep", "/tmp")

        kwargs = per_def.edit.call_args.kwargs
        assert kwargs["conclusion"] == "skipped"
        assert handler._report.rows["mydef"].apply_status == "not_applied"
        assert rollup.edit.call_args.kwargs["conclusion"] == "success"

    def test_apply_only_teardown_without_applies_leaves_comment(self):
        handler = make_handler()
        handler._planning = False
        handler._comments = [mock.Mock(body="old")]

        handler.teardown("dep", "/tmp")

        handler._comments[0].edit.assert_not_called()
        handler._check.edit.assert_not_called()

    def test_apply_teardown_errors_are_logged(self):
        handler, rollup, per_def = self.apply_handler()
        self.run(handler, TerraformStage.PRE)
        rollup.edit.side_effect = RuntimeError("boom")
        handler.teardown("dep", "/tmp")  # does not raise

    def test_plan_teardown_errors_do_not_block_apply_teardown(self):
        handler, rollup, per_def = self.apply_handler()
        self.run(handler, TerraformStage.PRE)
        handler._check.edit.side_effect = RuntimeError("boom")
        handler.teardown("dep", "/tmp")
        assert rollup.edit.call_args.kwargs["status"] == "completed"

    def test_apply_output_collapses_blank_runs(self):
        out = GithubHandler._apply_output(APPLY_STDOUT.decode())
        assert out.startswith("```\n")
        assert "\n\n\n" not in out
        assert GithubHandler._apply_output("\n\n") == ""

    def test_apply_line_absent(self):
        assert GithubHandler._apply_line("nothing here") == ""


class TestApplyReport:
    def make_report(self):
        report = GithubStatusReport(
            deployment="dep", marker="tfworker-status", max_detail_chars=100
        )
        report.mark("def1", "changes", plan_line="Plan: 1 to add")
        return report

    def test_apply_column_absent_until_an_apply(self):
        report = self.make_report()
        assert "| Apply |" not in report.render_comment_bodies()[0]
        report.mark_apply("def1", "applied")
        assert "| Apply |" in report.render_comment_bodies()[0]
        assert "| ✅ applied |" in report.render_comment_bodies()[0]

    def test_plan_check_summary_never_shows_apply(self):
        report = self.make_report()
        report.mark_apply("def1", "applied")
        assert "| Apply |" not in report.render_check_summary()

    def test_header_links_both_check_runs_in_the_comment_only(self):
        report = self.make_report()
        report.check_url = "https://github.test/plan"
        report.apply_check_url = "https://github.test/apply"
        links = (
            "[View plan check run](https://github.test/plan) · "
            "[View apply check run](https://github.test/apply)"
        )
        assert links in report.render_comment_bodies()[0]
        assert "View apply check run" not in report.render_check_summary()

    def test_rollup_summaries_do_not_link_to_themselves(self):
        report = self.make_report()
        report.check_url = "https://github.test/plan"
        report.apply_check_url = "https://github.test/apply"
        report.mark_apply("def1", "applied")
        assert "https://github.test/plan" not in report.render_check_summary()
        apply_summary = report.render_apply_summary()
        assert "[View plan check run](https://github.test/plan)" in apply_summary
        assert "https://github.test/apply" not in apply_summary

    def test_apply_summary_lists_applied_definitions(self):
        report = self.make_report()
        report.ensure("untouched")
        report.mark_apply("def1", "applied", apply_line="Apply complete!", detail="LOG")
        report.set_apply_url("def1", "https://github.test/a")
        summary = report.render_apply_summary()
        assert summary.startswith("## Terraform apply status: `dep`")
        assert "| [`def1`](https://github.test/a) | ✅ applied | Apply complete! |" in (
            summary
        )
        assert "untouched" not in summary
        assert "LOG" in summary

    def test_apply_summary_over_budget_drops_details(self):
        report = self.make_report()
        report.max_detail_chars = 500
        report.mark_apply("def1", "applied", detail="y" * 450)
        with mock.patch("tfworker.handlers.github.BODY_BUDGET", 300):
            summary = report.render_apply_summary()
        assert "y" * 100 not in summary
        assert "Detail sections omitted" in summary

    def test_apply_conclusion_scoped_to_named_definitions(self):
        report = self.make_report()
        report.mark_apply("old", "failed")
        report.mark_apply("def1", "applied")
        assert report.apply_conclusion(["def1"]) == "success"
        assert report.apply_conclusion(["def1", "old"]) == "failure"

    def test_apply_summary_line(self):
        report = self.make_report()
        assert report.apply_summary_line() == "no definitions"
        report.mark_apply("def1", "applied")
        assert report.apply_summary_line() == "1 ✅ applied"


def check_run(id, name, conclusion="success", title="", summary="", external_id=None):
    run = mock.Mock()
    run.id = id
    run.name = name
    run.conclusion = conclusion
    run.external_id = external_id
    run.html_url = f"https://github.test/runs/{id}"
    run.output = SimpleNamespace(title=title, summary=summary)
    return run


class TestLoadRunState:
    def handler_with_runs(self, *runs):
        handler = make_handler()
        handler._planning = False
        handler._report = GithubStatusReport(
            deployment="dep",
            marker="tfworker-status",
            max_detail_chars=8000,
            run_id="run-1",
        )
        handler._repo.get_commit.return_value.get_check_runs.return_value = list(runs)
        handler._issue.get_comments.return_value = []
        return handler

    def test_rebuilds_rows_from_this_runs_checks(self):
        handler = self.handler_with_runs(
            check_run(10, "tfworker/dep/plan", external_id="run-1"),
            check_run(
                11, "tfworker/dep/plan: a", title="No changes.", external_id="run-1"
            ),
            check_run(
                12,
                "tfworker/dep/plan: b",
                title="Plan: 1 to add",
                summary="DIFF",
                external_id="run-1",
            ),
            check_run(
                13, "tfworker/dep/plan: c", conclusion="failure", external_id="run-1"
            ),
            check_run(
                14, "tfworker/dep/plan: d", conclusion="skipped", external_id="run-1"
            ),
            check_run(
                15, "tfworker/dep/plan: e", title="Changes planned", external_id="run-1"
            ),
            # another run's checks on the same commit are ignored
            check_run(20, "tfworker/dep/plan", external_id="run-2"),
            check_run(21, "tfworker/dep/plan: z", external_id="run-2"),
        )

        handler._load_run_state("run-1")

        rows = handler._report.rows
        assert list(rows) == ["a", "b", "c", "d", "e"]
        assert rows["a"].status == "no_changes"
        assert rows["b"].status == "changes"
        assert rows["b"].plan_line == "Plan: 1 to add"
        assert rows["b"].detail == "DIFF"
        assert rows["b"].url == "https://github.test/runs/12"
        assert rows["c"].status == "failed"
        assert rows["d"].status == "skipped"
        assert rows["e"].plan_line == ""
        assert handler._report.check_url == "https://github.test/runs/10"
        get_runs = handler._repo.get_commit.return_value.get_check_runs
        get_runs.assert_called_once_with(filter="all")
        assert handler._comments_enabled is True

    def test_legacy_definition_checks_matched_by_creation_order(self):
        handler = self.handler_with_runs(
            check_run(5, "tfworker/dep/plan: stale", title="Plan: old"),
            check_run(10, "tfworker/dep/plan", external_id="run-1"),
            check_run(11, "tfworker/dep/plan: a", title="Plan: 1 to add"),
        )
        handler._load_run_state("run-1")
        assert list(handler._report.rows) == ["a"]

    def test_newest_check_wins_keeping_position(self):
        handler = self.handler_with_runs(
            check_run(10, "tfworker/dep/plan", external_id="run-1"),
            check_run(
                11, "tfworker/dep/plan: a", conclusion="failure", external_id="run-1"
            ),
            check_run(12, "tfworker/dep/plan: b", external_id="run-1"),
            check_run(
                13, "tfworker/dep/plan: a", title="Plan: 2 to add", external_id="run-1"
            ),
        )
        handler._load_run_state("run-1")
        rows = handler._report.rows
        assert list(rows) == ["a", "b"]
        assert rows["a"].status == "changes"

    def test_prior_applies_of_the_run_are_loaded(self):
        handler = self.handler_with_runs(
            check_run(10, "tfworker/dep/plan", external_id="run-1"),
            check_run(11, "tfworker/dep/plan: a", title="Plan: 1", external_id="run-1"),
            check_run(12, "tfworker/dep/plan: b", title="Plan: 1", external_id="run-1"),
            check_run(13, "tfworker/dep/plan: c", title="Plan: 1", external_id="run-1"),
            check_run(14, "tfworker/dep/plan: d", title="Plan: 1", external_id="run-1"),
            check_run(
                20,
                "tfworker/dep/apply: a",
                title="Apply complete!",
                external_id="run-1",
            ),
            check_run(
                21,
                "tfworker/dep/apply: b",
                conclusion="failure",
                title="Terraform apply failed",
                external_id="run-1",
            ),
            check_run(
                22, "tfworker/dep/apply: c", conclusion="skipped", external_id="run-1"
            ),
            check_run(
                23, "tfworker/dep/apply: d", conclusion=None, external_id="run-1"
            ),
            check_run(
                24, "tfworker/dep/apply: a", title="ignored", external_id="run-2"
            ),
        )
        handler._load_run_state("run-1")
        rows = handler._report.rows
        assert rows["a"].apply_status == "applied"
        assert rows["a"].apply_line == "Apply complete!"
        assert rows["a"].apply_url == "https://github.test/runs/20"
        assert rows["b"].apply_status == "failed"
        assert rows["b"].apply_line == ""
        assert rows["c"].apply_status == "not_applied"
        assert rows["d"].apply_status is None

    def test_claims_comment_of_this_run(self):
        handler = self.handler_with_runs(
            check_run(10, "tfworker/dep/plan", external_id="run-1"),
        )
        mine = mock.Mock(body="<!-- tfworker-status: dep -->\nRun `run-1`\n")
        handler._issue.get_comments.return_value = [mine]
        handler._load_run_state("run-1")
        assert handler._comments == [mine]
        assert handler._comments_enabled is True

    def test_comment_of_another_run_is_left_alone(self):
        handler = self.handler_with_runs(
            check_run(10, "tfworker/dep/plan", external_id="run-1"),
        )
        newer = mock.Mock(body="<!-- tfworker-status: dep -->\nRun `run-2`\n")
        handler._issue.get_comments.return_value = [newer]

        handler._load_run_state("run-1")
        handler._update_comments()

        assert handler._comments_enabled is False
        newer.edit.assert_not_called()

    def test_no_plan_rollup_disables_comments(self):
        handler = self.handler_with_runs(
            check_run(10, "tfworker/dep/plan", external_id="run-2"),
        )
        handler._load_run_state("run-1")
        assert handler._comments_enabled is False
        assert handler._report.rows == {}

    def test_no_run_id_disables_comments(self):
        handler = self.handler_with_runs()
        handler._load_run_state(None)
        assert handler._comments_enabled is False
        handler._repo.get_commit.assert_not_called()

    def test_read_failure_disables_comments_not_the_handler(self):
        handler = self.handler_with_runs()
        handler._repo.get_commit.side_effect = RuntimeError("boom")
        handler._load_run_state("run-1")
        assert handler._comments_enabled is False
        assert handler.is_ready() is True

    def test_disable_without_a_pr_does_not_warn(self):
        handler = make_handler(with_pr=False)
        with mock.patch("tfworker.handlers.github.log.warn") as warn:
            handler._disable_comments("reason")
        warn.assert_not_called()
        assert handler._comments_enabled is False

    def test_plan_status_of_an_unfinished_check(self):
        run = check_run(1, "x", conclusion=None)
        assert GithubHandler._plan_status(run) == ("pending", "")


class TestAccessors:
    def test_report_and_repo_raise_before_setup(self):
        handler = GithubHandler(make_config())
        with pytest.raises(HandlerError):
            handler.report
        with pytest.raises(HandlerError):
            handler.repo


class FakeClock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


DETAILS_TEMPLATE = (
    "https://logs.example.com/search?q=run%3A{run_id}%20dep%3A{deployment}"
    "%20def%3A{definition}&from={from_ts}&to={to_ts}"
)


def details(definition="", from_ts=700000, to_ts=22600000, run_id="run-1234"):
    """The expected details_url; defaults are a clock started and read at 1000s."""
    return (
        f"https://logs.example.com/search?q=run%3A{run_id}%20dep%3Adep"
        f"%20def%3A{definition}&from={from_ts}&to={to_ts}"
    )


class TestDetailsUrl:
    """Check runs link to the run's logs: while running, a window open well past
    creation; once concluded, exactly the run."""

    def handler(self, clock=None, **config):
        config.setdefault("details_url", DETAILS_TEMPLATE)
        handler = make_handler(config=make_config(**config), clock=clock or FakeClock())
        handler._app_state.root_options.run_id = "run-1234"
        return handler

    def setup_plan(self, handler):
        handler._connect = mock.Mock()
        handler._claim_comments = mock.Mock()
        handler._resolve_sha = mock.Mock(return_value="abc123")
        handler._update_comments = mock.Mock()
        definitions = {"mydef": make_definition()}
        handler.setup(
            "dep",
            mock.Mock(values=lambda: definitions.values()),
            "/tmp",
            mock.Mock(plan=True),
        )

    def run(self, handler, action, stage, result=None):
        return handler.execute(
            action=action,
            stage=stage,
            deployment="dep",
            definition=make_definition(),
            working_dir="/tmp",
            result=result,
        )

    def test_config_env_fallbacks(self, monkeypatch):
        monkeypatch.setenv("GITHUB_CHECK_DETAILS_URL", "https://logs.example.com/a")
        monkeypatch.setenv(
            "GITHUB_CHECK_DEFINITION_DETAILS_URL", "https://logs.example.com/b"
        )
        config = GithubConfig()
        assert config.details_url == "https://logs.example.com/a"
        assert config.definition_details_url == "https://logs.example.com/b"

    def test_config_wins_over_env_and_empty_renders_fall_back(self, monkeypatch):
        monkeypatch.setenv("GITHUB_CHECK_DETAILS_URL", "https://logs.example.com/a")
        config = GithubConfig(
            details_url="https://logs.example.com/c", definition_details_url=""
        )
        assert config.details_url == "https://logs.example.com/c"
        assert config.definition_details_url is None

    def test_plan_checks_created_with_a_running_window(self):
        clock = FakeClock(500.0)
        handler = self.handler(clock=clock)
        clock.now = 1000.0  # setup time, not construction time, starts the window
        self.setup_plan(handler)

        rollup, per_def = handler._repo.create_check_run.call_args_list
        assert rollup.kwargs["details_url"] == details()
        assert per_def.kwargs["details_url"] == details(definition="mydef")

    def test_plan_checks_concluded_with_the_run_window(self):
        clock = FakeClock()
        handler = self.handler(clock=clock)
        clock.now = 1060.0

        self.run(
            handler,
            TerraformAction.PLAN,
            TerraformStage.POST,
            TerraformResult(0, b"No changes.", b""),
        )
        def_kwargs = handler._def_checks["mydef"].edit.call_args.kwargs
        assert def_kwargs["details_url"] == details("mydef", to_ts=1360000)

        clock.now = 1120.0
        handler.teardown("dep", "/tmp")
        rollup_kwargs = handler._check.edit.call_args.kwargs
        assert rollup_kwargs["status"] == "completed"
        assert rollup_kwargs["details_url"] == details(to_ts=1420000)

    def test_skipped_definition_checks_concluded_at_teardown(self):
        clock = FakeClock()
        handler = self.handler(clock=clock)
        clock.now = 1060.0
        handler.teardown("dep", "/tmp")
        kwargs = handler._def_checks["mydef"].edit.call_args.kwargs
        assert kwargs["conclusion"] == "skipped"
        assert kwargs["details_url"] == details("mydef", to_ts=1360000)

    def test_apply_checks_created_and_concluded(self):
        clock = FakeClock()
        handler = self.handler(clock=clock)
        rollup = mock.Mock(html_url="https://github.test/apply")
        per_def = mock.Mock(html_url="https://github.test/apply/mydef")
        handler._repo.create_check_run.side_effect = [rollup, per_def]

        self.run(handler, TerraformAction.APPLY, TerraformStage.PRE)
        created = handler._repo.create_check_run.call_args_list
        assert created[0].kwargs["details_url"] == details()
        assert created[1].kwargs["details_url"] == details(definition="mydef")

        clock.now = 1060.0
        self.run(
            handler,
            TerraformAction.APPLY,
            TerraformStage.POST,
            TerraformResult(0, APPLY_STDOUT, b""),
        )
        concluded = per_def.edit.call_args.kwargs
        assert concluded["details_url"] == details("mydef", to_ts=1360000)

        clock.now = 1120.0
        handler.teardown("dep", "/tmp")
        assert rollup.edit.call_args.kwargs["details_url"] == details(to_ts=1420000)

    def test_definition_template_preferred_for_definition_checks(self):
        handler = self.handler(
            definition_details_url="https://logs.example.com/def/{definition}"
        )
        self.setup_plan(handler)
        rollup, per_def = handler._repo.create_check_run.call_args_list
        assert rollup.kwargs["details_url"] == details()
        assert per_def.kwargs["details_url"] == "https://logs.example.com/def/mydef"

    def test_definition_template_alone_leaves_rollups_unlinked(self):
        handler = self.handler(
            details_url=None,
            definition_details_url="https://logs.example.com/def/{definition}",
        )
        self.setup_plan(handler)
        rollup, per_def = handler._repo.create_check_run.call_args_list
        assert "details_url" not in rollup.kwargs
        assert per_def.kwargs["details_url"] == "https://logs.example.com/def/mydef"

    def test_env_templates_are_applied(self, monkeypatch):
        monkeypatch.setenv("GITHUB_CHECK_DETAILS_URL", "https://logs.example.com/r")
        monkeypatch.setenv(
            "GITHUB_CHECK_DEFINITION_DETAILS_URL",
            "https://logs.example.com/d/{definition}",
        )
        handler = make_handler(clock=FakeClock())
        handler._app_state.root_options.run_id = None
        self.setup_plan(handler)
        rollup, per_def = handler._repo.create_check_run.call_args_list
        assert rollup.kwargs["details_url"] == "https://logs.example.com/r"
        assert per_def.kwargs["details_url"] == "https://logs.example.com/d/mydef"

    def test_values_are_url_quoted(self):
        handler = self.handler(
            details_url="https://logs.example.com/?r={run_id}&d={deployment}&n={definition}"
        )
        handler._app_state.root_options.run_id = "run 1/2"
        handler._deployment = "a&b"
        assert handler._details_url("x y#z") == {
            "details_url": "https://logs.example.com/?r=run%201%2F2&d=a%26b&n=x%20y%23z"
        }

    def test_no_templates_send_no_details_url(self):
        handler = make_handler(clock=FakeClock())
        handler._app_state.root_options.run_id = "run-1234"
        self.setup_plan(handler)
        self.run(
            handler,
            TerraformAction.PLAN,
            TerraformStage.POST,
            TerraformResult(0, b"No changes.", b""),
        )
        handler.teardown("dep", "/tmp")
        calls = handler._repo.create_check_run.call_args_list + [
            *handler._repo.create_check_run.return_value.edit.call_args_list
        ]
        assert calls
        assert all("details_url" not in c.kwargs for c in calls)

    @pytest.mark.parametrize(
        "template",
        [
            "https://logs.example.com/{unknown}",
            "https://logs.example.com/{0}",
            "https://logs.example.com/{from_ts:s}",
            "https://logs.example.com/{run_id.x}",
            "https://logs.example.com/{",
        ],
    )
    def test_bad_template_warns_once_and_is_omitted(self, template):
        handler = self.handler(details_url=template)
        with mock.patch("tfworker.handlers.github.log.warn") as warn:
            self.setup_plan(handler)
            handler.teardown("dep", "/tmp")
        assert handler.is_ready() is True
        created = handler._repo.create_check_run.call_args_list
        assert len(created) == 2
        assert all("details_url" not in c.kwargs for c in created)
        assert "details_url" not in handler._check.edit.call_args.kwargs
        assert warn.call_count == 1
        assert "details url template" in warn.call_args.args[0]


def logs(definition="", from_ts=700000, to_ts=22600000, label="View logs"):
    """The expected logs link line; defaults are a clock started and read at 1000s."""
    return f"[{label}]({details(definition, from_ts, to_ts)})"


class TestLogsLink:
    """Check run output carries a labelled logs link: rollups in their header,
    per-definition checks at the top of their concluded summary."""

    setup_plan = TestDetailsUrl.setup_plan
    run = TestDetailsUrl.run

    def handler(self, clock=None, **config):
        config.setdefault("logs_url", DETAILS_TEMPLATE)
        handler = make_handler(config=make_config(**config), clock=clock or FakeClock())
        handler._app_state.root_options.run_id = "run-1234"
        return handler

    def test_config_env_fallbacks(self, monkeypatch):
        monkeypatch.setenv("GITHUB_CHECK_LOGS_URL", "https://logs.example.com/a")
        monkeypatch.setenv(
            "GITHUB_CHECK_DEFINITION_LOGS_URL", "https://logs.example.com/b"
        )
        monkeypatch.setenv("GITHUB_CHECK_LOGS_LABEL", "Logs")
        config = GithubConfig()
        assert config.logs_url == "https://logs.example.com/a"
        assert config.definition_logs_url == "https://logs.example.com/b"
        assert config.logs_label == "Logs"

    def test_config_wins_over_env_and_empty_renders_fall_back(self, monkeypatch):
        monkeypatch.setenv("GITHUB_CHECK_LOGS_URL", "https://logs.example.com/a")
        monkeypatch.setenv("GITHUB_CHECK_LOGS_LABEL", "Logs")
        config = GithubConfig(
            logs_url="https://logs.example.com/c",
            definition_logs_url="",
            logs_label="Run logs",
        )
        assert config.logs_url == "https://logs.example.com/c"
        assert config.definition_logs_url is None
        assert config.logs_label == "Run logs"
        assert GithubConfig(logs_label="").logs_label == "Logs"

    def test_default_label(self):
        config = GithubConfig()
        assert config.logs_url is None
        assert config.logs_label == "View logs"

    def test_plan_rollup_links_logs_in_its_header(self):
        handler = self.handler()
        self.setup_plan(handler)
        rollup, per_def = handler._repo.create_check_run.call_args_list
        summary = rollup.kwargs["output"]["summary"]
        assert f"Run `run-1234`\n\n{logs()}\n" in summary
        # a check created without output gets none just for the link
        assert "output" not in per_def.kwargs

    def test_plan_checks_concluded_with_the_run_window(self):
        clock = FakeClock()
        handler = self.handler(clock=clock)
        clock.now = 1060.0
        self.run(
            handler,
            TerraformAction.PLAN,
            TerraformStage.POST,
            TerraformResult(2, PLAN_STDOUT, b""),
        )
        summary = handler._def_checks["mydef"].edit.call_args.kwargs["output"][
            "summary"
        ]
        assert summary.startswith(f"{logs('mydef', to_ts=1360000)}\n\n```diff\n")
        # the rollup re-render while running keeps the open window
        running = handler._check.edit.call_args.kwargs["output"]["summary"]
        assert logs(to_ts=22660000) in running

        clock.now = 1120.0
        handler.teardown("dep", "/tmp")
        kwargs = handler._check.edit.call_args.kwargs
        assert kwargs["status"] == "completed"
        assert logs(to_ts=1420000) in kwargs["output"]["summary"]

    def test_skipped_definition_check_is_just_the_link(self):
        handler = self.handler()
        handler.teardown("dep", "/tmp")
        output = handler._def_checks["mydef"].edit.call_args.kwargs["output"]
        assert output["summary"] == logs("mydef", to_ts=1300000)

    def test_apply_checks(self):
        clock = FakeClock()
        handler = self.handler(clock=clock)
        rollup = mock.Mock(html_url="https://github.test/apply")
        per_def = mock.Mock(html_url="https://github.test/apply/mydef")
        handler._repo.create_check_run.side_effect = [rollup, per_def]

        self.run(handler, TerraformAction.APPLY, TerraformStage.PRE)
        created = handler._repo.create_check_run.call_args_list
        assert logs() in created[0].kwargs["output"]["summary"]
        assert "output" not in created[1].kwargs
        assert logs() in rollup.edit.call_args.kwargs["output"]["summary"]

        clock.now = 1060.0
        self.run(
            handler,
            TerraformAction.APPLY,
            TerraformStage.POST,
            TerraformResult(0, APPLY_STDOUT, b""),
        )
        summary = per_def.edit.call_args.kwargs["output"]["summary"]
        assert summary.startswith(
            f"{logs('mydef', to_ts=1360000)} · "
            "[View dep apply check run](https://github.test/apply)\n\n```\n"
        )

        clock.now = 1120.0
        handler.teardown("dep", "/tmp")
        kwargs = rollup.edit.call_args.kwargs
        assert kwargs["status"] == "completed"
        assert logs(to_ts=1420000) in kwargs["output"]["summary"]

    def test_definition_template_preferred_for_definition_checks(self):
        handler = self.handler(
            definition_logs_url="https://logs.example.com/def/{definition}"
        )
        assert handler._logs_link() == logs(to_ts=22600000)
        assert handler._logs_link("mydef") == (
            "[View logs](https://logs.example.com/def/mydef)"
        )

    def test_definition_checks_fall_back_to_the_rollup_template(self):
        handler = self.handler()
        assert handler._logs_link("mydef") == logs("mydef")

    def test_definition_template_alone_leaves_rollups_unlinked(self):
        handler = self.handler(
            logs_url=None,
            definition_logs_url="https://logs.example.com/def/{definition}",
        )
        assert handler._logs_link() == ""
        assert handler._logs_link("mydef") == (
            "[View logs](https://logs.example.com/def/mydef)"
        )

    def test_env_templates_and_label_are_applied(self, monkeypatch):
        monkeypatch.setenv("GITHUB_CHECK_LOGS_URL", "https://logs.example.com/r")
        monkeypatch.setenv(
            "GITHUB_CHECK_DEFINITION_LOGS_URL",
            "https://logs.example.com/d/{definition}",
        )
        monkeypatch.setenv("GITHUB_CHECK_LOGS_LABEL", "Logs")
        handler = make_handler(clock=FakeClock())
        handler._app_state.root_options.run_id = None
        assert handler._logs_link() == "[Logs](https://logs.example.com/r)"
        assert handler._logs_link("mydef") == "[Logs](https://logs.example.com/d/mydef)"

    def test_custom_label(self):
        handler = self.handler(logs_label="Search logs")
        assert handler._logs_link() == logs(label="Search logs")

    def test_values_are_url_quoted(self):
        handler = self.handler(
            logs_url="https://logs.example.com/?r={run_id}&d={deployment}&n={definition}"
        )
        handler._app_state.root_options.run_id = "run 1/2"
        handler._deployment = "a&b"
        assert handler._logs_link("x y#z") == (
            "[View logs](https://logs.example.com/?r=run%201%2F2&d=a%26b&n=x%20y%23z)"
        )

    def test_no_templates_leave_output_unchanged(self):
        handler = make_handler(clock=FakeClock())
        handler._app_state.root_options.run_id = "run-1234"
        self.setup_plan(handler)
        self.run(
            handler,
            TerraformAction.PLAN,
            TerraformStage.POST,
            TerraformResult(0, b"No changes.", b""),
        )
        handler.teardown("dep", "/tmp")
        calls = handler._repo.create_check_run.call_args_list + [
            *handler._repo.create_check_run.return_value.edit.call_args_list
        ]
        outputs = [str(c.kwargs.get("output", "")) for c in calls]
        assert outputs
        assert not any("View logs" in o for o in outputs)

    def test_status_comment_has_no_logs_link(self):
        handler = self.handler()
        handler._update_comments()
        assert "View logs" not in handler._issue.create_comment.call_args.args[0]

    def test_bad_template_warns_once_and_is_omitted(self):
        handler = self.handler(logs_url="https://logs.example.com/{unknown}")
        with mock.patch("tfworker.handlers.github.log.warn") as warn:
            self.setup_plan(handler)
            self.run(
                handler,
                TerraformAction.PLAN,
                TerraformStage.POST,
                TerraformResult(0, b"No changes.", b""),
            )
            handler.teardown("dep", "/tmp")
        assert handler.is_ready() is True
        edits = handler._repo.create_check_run.return_value.edit.call_args_list
        assert edits
        assert not any("View logs" in str(c.kwargs.get("output")) for c in edits)
        assert warn.call_count == 1
        assert "logs url template" in warn.call_args.args[0]

    def test_details_url_unaffected(self):
        handler = self.handler(details_url="https://docs.example.com/{deployment}")
        self.setup_plan(handler)
        rollup, per_def = handler._repo.create_check_run.call_args_list
        assert rollup.kwargs["details_url"] == "https://docs.example.com/dep"
        assert per_def.kwargs["details_url"] == "https://docs.example.com/dep"
        assert logs() in rollup.kwargs["output"]["summary"]

    def test_logs_templates_alone_set_no_details_url(self):
        handler = self.handler()
        self.setup_plan(handler)
        created = handler._repo.create_check_run.call_args_list
        assert all("details_url" not in c.kwargs for c in created)


class TestRollupBackLinks:
    """Per-definition check runs link back to their rollup check run."""

    def test_definition_plan_check_links_to_the_plan_rollup(self):
        handler = make_handler()
        handler._report.check_url = "https://github.test/check/1"
        handler._conclude_def_check("mydef", "success", "No changes.")
        summary = handler._def_checks["mydef"].edit.call_args.kwargs["output"][
            "summary"
        ]
        assert summary == "[View dep plan check run](https://github.test/check/1)"

    def test_definition_apply_check_links_to_the_apply_rollup(self):
        handler = make_handler()
        handler._report.apply_check_url = "https://github.test/apply"
        per_def = mock.Mock()
        handler._apply_checks = {"mydef": per_def}
        handler._conclude_apply_check("mydef", "success", "done", "LOG")
        summary = per_def.edit.call_args.kwargs["output"]["summary"]
        assert summary == "[View dep apply check run](https://github.test/apply)\n\nLOG"

    def test_omitted_when_the_rollup_is_unknown(self):
        handler = make_handler()
        handler._conclude_def_check("mydef", "failure", "failed", "ERR")
        output = handler._def_checks["mydef"].edit.call_args.kwargs["output"]
        assert output["summary"] == "ERR"
        per_def = mock.Mock()
        handler._apply_checks = {"mydef": per_def}
        handler._report = None
        handler._conclude_apply_check("mydef", "success", "done", "LOG")
        assert per_def.edit.call_args.kwargs["output"]["summary"] == "LOG"

    def test_setup_links_definition_checks_to_the_created_rollup(self):
        handler = make_handler()
        handler._app_state.root_options.run_id = None
        rollup = mock.Mock(html_url="https://github.test/plan")
        per_def = mock.Mock(html_url="https://github.test/plan/mydef")
        handler._repo.create_check_run.side_effect = [rollup, per_def]
        TestDetailsUrl.setup_plan(None, handler)
        handler.teardown("dep", "/tmp")
        output = per_def.edit.call_args.kwargs["output"]
        assert (
            output["summary"] == "[View dep plan check run](https://github.test/plan)"
        )
        assert (
            "https://github.test/plan)"
            not in rollup.edit.call_args.kwargs["output"]["summary"]
        )

    def test_links_are_stripped_when_read_back(self):
        handler = TestLoadRunState().handler_with_runs(
            check_run(10, "tfworker/dep/plan", external_id="run-1"),
            check_run(
                11,
                "tfworker/dep/plan: a",
                title="Plan: 1 to add",
                summary="[View logs](https://l/a) · [View dep plan check run](https://p)"
                "\n\nDIFF",
                external_id="run-1",
            ),
            check_run(
                12,
                "tfworker/dep/plan: b",
                title="No changes.",
                summary="[View logs](https://l/b)",
                external_id="run-1",
            ),
            check_run(
                13,
                "tfworker/dep/plan: c",
                title="Plan: 1 to add",
                summary="[Other](https://x)\n\nKEEP",
                external_id="run-1",
            ),
            check_run(
                20,
                "tfworker/dep/apply: a",
                title="Apply complete!",
                summary="[View dep apply check run](https://a)\n\nLOG",
                external_id="run-1",
            ),
        )
        handler._load_run_state("run-1")
        rows = handler._report.rows
        assert rows["a"].detail == "DIFF"
        assert rows["b"].detail == ""
        assert rows["c"].detail == "[Other](https://x)\n\nKEEP"
        assert rows["a"].apply_detail == "LOG"
