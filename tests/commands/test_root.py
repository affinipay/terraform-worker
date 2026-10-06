from tfworker.cli_options import CLIOptionsRoot
from tfworker.commands.root import RootCommand


class TestRootHelpers:
    def test_resolve_working_dir_tmp(self, tmp_path, mocker):
        tmp = mocker.Mock()
        tmp.name = str(tmp_path)
        mocker.patch("tempfile.TemporaryDirectory", return_value=tmp)
        assert RootCommand._resolve_working_dir(None) == tmp_path

    def test_prepare_template_vars(self):
        opts = CLIOptionsRoot(
            config_file=[], aws_region="us-east-1", config_var=["foo=bar"]
        )
        res = RootCommand._prepare_template_vars(opts, "dep")
        assert res["aws_region"] == "us-east-1"
        assert res["foo"] == "bar"

    def test_prepare_template_vars_includes_deployment(self):
        opts = CLIOptionsRoot(config_file=[], aws_region="us-east-1")
        res = RootCommand._prepare_template_vars(opts, "my-deployment")
        assert res["deployment"] == "my-deployment"

    def test_prepare_template_vars_deployment_beats_config_var(self):
        opts = CLIOptionsRoot(config_file=[], config_var=["deployment=other"])
        res = RootCommand._prepare_template_vars(opts, "my-deployment")
        assert res["deployment"] == "my-deployment"


class TestRootCommandInit:
    def test_load_config_receives_deployment(self, mocker, tmp_path):
        app_state = mocker.MagicMock()
        app_state.root_options = CLIOptionsRoot(
            config_file=[], working_dir=str(tmp_path)
        )
        mocker.patch(
            "click.get_current_context", return_value=mocker.Mock(obj=app_state)
        )
        load = mocker.patch("tfworker.commands.root.load_config")
        mocker.patch("tfworker.commands.root.resolve_model_with_cli_options")

        RootCommand(deployment="dep")

        assert load.call_args.args[1]["deployment"] == "dep"
        assert app_state.loaded_config == load.return_value
