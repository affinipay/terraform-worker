import json
import os
import stat

from tfworker.definitions import sources as s

SCRIPT = """#!/bin/sh
cat <<EOF
generated:
  template_vars:
    cwd: $(pwd)
    deployment: $WORKER_DEPLOYMENT
    inherited: $SOURCE_TEST_VAR
EOF
"""


def _make_source(tmp_path, script=SCRIPT, name="bin/list-definitions"):
    """A local source dir holding an executable script."""
    src = tmp_path / "catalog"
    script_path = src / name
    script_path.parent.mkdir(parents=True)
    script_path.write_text(script)
    script_path.chmod(script_path.stat().st_mode | stat.S_IXUSR)
    return src


def _stdout(mocker, *outputs):
    """Patch pipe_exec to return each output in turn."""
    return mocker.patch(
        "tfworker.definitions.sources.pipe_exec",
        side_effect=[(0, o.encode(), b"") for o in outputs],
    )


def _copier(mocker):
    """Patch the copier factory; returns the mocked create."""
    return mocker.patch("tfworker.definitions.sources.CopyFactory.create")


class TestApplyDefinitionsSourcesLocal:
    def test_runs_command_in_fetched_copy(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SOURCE_TEST_VAR", "from-env")
        src = _make_source(tmp_path)
        work = tmp_path / "work"
        work.mkdir()
        config = {
            "definitions": {"base": {"path": "./base"}},
            "definitions_sources": {
                "local": {"path": str(src), "command": "bin/list-definitions"}
            },
        }

        s.apply_definitions_sources(config, "my-deployment", str(tmp_path), work)

        copy_dir = work / "definitions_sources" / "local"
        assert (copy_dir / "bin" / "list-definitions").exists()
        generated = config["definitions"]["generated"]
        assert generated["template_vars"] == {
            "cwd": str(copy_dir.resolve()),
            "deployment": "my-deployment",
            "inherited": "from-env",
        }
        assert generated["path"] == str(src)
        assert list(config["definitions"]) == ["base", "generated"]
        assert os.environ.get("WORKER_DEPLOYMENT") != "my-deployment"

    def test_relative_path_resolves_against_repository_path(self, tmp_path):
        _make_source(tmp_path)
        work = tmp_path / "work"
        work.mkdir()
        config = {
            "definitions": {},
            "definitions_sources": {
                "local": {"path": "catalog", "command": "bin/list-definitions"}
            },
        }

        s.apply_definitions_sources(config, "d", str(tmp_path), work)

        assert config["definitions"]["generated"]["path"] == "catalog"


class TestApplyDefinitionsSourcesMocked:
    def test_yaml_output(self, mocker, tmp_path):
        _copier(mocker)
        _stdout(mocker, "one:\n  path: ./one\ntwo:\n  path: ./two\n")
        config = {
            "definitions": {"base": {"path": "./base"}},
            "definitions_sources": {"src": {"path": "./x", "command": "run"}},
        }
        s.apply_definitions_sources(config, "d", ".", tmp_path)
        assert list(config["definitions"]) == ["base", "one", "two"]

    def test_json_output(self, mocker, tmp_path):
        _copier(mocker)
        _stdout(mocker, json.dumps({"one": {"path": "./one"}, "two": {}}))
        config = {
            "definitions": {},
            "definitions_sources": {"src": {"path": "./x", "command": "run"}},
        }
        s.apply_definitions_sources(config, "d", ".", tmp_path)
        assert config["definitions"] == {
            "one": {"path": "./one"},
            "two": {"path": "./x"},
        }

    def test_inserted_after_anchor_in_printed_order(self, mocker, tmp_path):
        _copier(mocker)
        _stdout(mocker, "zeta: {}\nalpha: {}\n")
        config = {
            "definitions": {
                "network": {"path": "./n"},
                "app": {"path": "./a"},
                "last": {"path": "./l"},
            },
            "definitions_sources": {
                "src": {"path": "./x", "command": "run", "after": "network"}
            },
        }
        s.apply_definitions_sources(config, "d", ".", tmp_path)
        assert list(config["definitions"]) == [
            "network",
            "zeta",
            "alpha",
            "app",
            "last",
        ]

    def test_appended_without_anchor(self, mocker, tmp_path):
        _copier(mocker)
        _stdout(mocker, "zeta: {}\nalpha: {}\n")
        config = {
            "definitions": {"network": {"path": "./n"}, "app": {"path": "./a"}},
            "definitions_sources": {"src": {"path": "./x", "command": "run"}},
        }
        s.apply_definitions_sources(config, "d", ".", tmp_path)
        assert list(config["definitions"]) == ["network", "app", "zeta", "alpha"]

    def test_missing_definitions_key(self, mocker, tmp_path):
        _copier(mocker)
        _stdout(mocker, "one: {}\n")
        config = {"definitions_sources": {"src": {"path": "./x", "command": "run"}}}
        s.apply_definitions_sources(config, "d", ".", tmp_path)
        assert config["definitions"] == {"one": {"path": "./x"}}

    def test_location_inheritance(self, mocker, tmp_path):
        _copier(mocker)
        _stdout(
            mocker,
            "inherits:\n  always_apply: true\n"
            "own:\n  path: git@github.com:example/other.git\n",
        )
        config = {
            "definitions": {},
            "definitions_sources": {
                "src": {
                    "path": "git@github.com:example/service-catalog.git",
                    "remote_path_options": {"branch": "main"},
                    "command": "run",
                }
            },
        }
        s.apply_definitions_sources(config, "d", ".", tmp_path)
        assert config["definitions"]["inherits"] == {
            "always_apply": True,
            "path": "git@github.com:example/service-catalog.git",
            "remote_path_options": {"branch": "main"},
        }
        assert config["definitions"]["own"] == {
            "path": "git@github.com:example/other.git"
        }

    def test_own_remote_path_options_kept(self, mocker, tmp_path):
        _copier(mocker)
        _stdout(mocker, "one:\n  remote_path_options:\n    sub_path: mod\n")
        config = {
            "definitions": {},
            "definitions_sources": {
                "src": {
                    "path": "./x",
                    "remote_path_options": {"branch": "main"},
                    "command": "run",
                }
            },
        }
        s.apply_definitions_sources(config, "d", ".", tmp_path)
        assert config["definitions"]["one"] == {
            "path": "./x",
            "remote_path_options": {"sub_path": "mod"},
        }

    def test_two_sources_in_config_order(self, mocker, tmp_path):
        _copier(mocker)
        pipe = _stdout(mocker, "first_gen: {}\n", "second_gen: {}\n")
        config = {
            "definitions": {"network": {"path": "./n"}, "app": {"path": "./a"}},
            "definitions_sources": {
                "first": {"path": "./one", "command": "run-one", "after": "network"},
                "second": {
                    "path": "./two",
                    "command": "run-two",
                    "after": "first_gen",
                },
            },
        }
        s.apply_definitions_sources(config, "d", ".", tmp_path)
        assert [c.args[0] for c in pipe.call_args_list] == ["run-one", "run-two"]
        assert list(config["definitions"]) == [
            "network",
            "first_gen",
            "second_gen",
            "app",
        ]
        assert config["definitions"]["second_gen"]["path"] == "./two"

    def test_command_environment(self, mocker, tmp_path, monkeypatch):
        monkeypatch.setenv("SOURCE_TEST_VAR", "inherited")
        _copier(mocker)
        pipe = _stdout(mocker, "one: {}\n")
        config = {
            "definitions": {},
            "definitions_sources": {"src": {"path": "./x", "command": "run --flag"}},
        }
        s.apply_definitions_sources(config, "the-deployment", ".", tmp_path)
        kwargs = pipe.call_args.kwargs
        assert kwargs["cwd"] == str(tmp_path / "definitions_sources" / "src")
        assert kwargs["env"]["WORKER_DEPLOYMENT"] == "the-deployment"
        assert kwargs["env"]["SOURCE_TEST_VAR"] == "inherited"
        assert pipe.call_args.args[0] == "run --flag"

    def test_branch_passed_to_copier(self, mocker, tmp_path):
        create = _copier(mocker)
        _stdout(mocker, "one: {}\n")
        config = {
            "definitions": {},
            "definitions_sources": {
                "src": {
                    "path": "git@github.com:example/service-catalog.git",
                    "remote_path_options": {"branch": "feature"},
                    "command": "run",
                }
            },
        }
        s.apply_definitions_sources(config, "d", "/repo", tmp_path)
        create.assert_called_once_with(
            "git@github.com:example/service-catalog.git",
            root_path="/repo",
            conflicts=[],
        )
        create.return_value.copy.assert_called_once_with(
            destination=str(tmp_path / "definitions_sources" / "src"),
            branch="feature",
            sub_path=None,
        )

    def test_no_remote_options_copies_without_options(self, mocker, tmp_path):
        create = _copier(mocker)
        _stdout(mocker, "one: {}\n")
        config = {
            "definitions": {},
            "definitions_sources": {"src": {"path": "./x", "command": "run"}},
        }
        s.apply_definitions_sources(config, "d", ".", tmp_path)
        create.return_value.copy.assert_called_once_with(
            destination=str(tmp_path / "definitions_sources" / "src")
        )
