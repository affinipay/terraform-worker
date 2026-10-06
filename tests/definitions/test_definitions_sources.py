import json
import os
import stat

import pytest
from pydantic import ValidationError

from tfworker.copier import CopyFactory
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


def _config(**source):
    """A config with one configured definition and one source named src."""
    return {
        "definitions": {"network": {"path": "./n"}},
        "definitions_sources": {"src": {"path": "./x", "command": "run", **source}},
    }


class TestApplyDefinitionsSourcesErrors:
    def test_non_zero_exit(self, mocker, tmp_path):
        _copier(mocker)
        mocker.patch(
            "tfworker.definitions.sources.pipe_exec",
            return_value=(3, b"partial", b"something broke"),
        )
        with pytest.raises(s.DefinitionsSourceError) as e:
            s.apply_definitions_sources(_config(), "d", ".", tmp_path)
        assert "src" in str(e.value)
        assert "exited 3" in str(e.value)
        assert "something broke" in str(e.value)

    def test_stderr_logged_at_debug_on_success(self, mocker, tmp_path):
        _copier(mocker)
        mocker.patch(
            "tfworker.definitions.sources.pipe_exec",
            return_value=(0, b"one: {}\n", b"a warning"),
        )
        debug = mocker.patch("tfworker.util.log.debug")
        s.apply_definitions_sources(_config(), "d", ".", tmp_path)
        assert any("a warning" in str(c.args[0]) for c in debug.call_args_list)

    @pytest.mark.parametrize("error", [FileNotFoundError, PermissionError])
    def test_command_not_runnable(self, mocker, tmp_path, error):
        _copier(mocker)
        mocker.patch(
            "tfworker.definitions.sources.pipe_exec", side_effect=error("nope")
        )
        with pytest.raises(s.DefinitionsSourceError, match="src"):
            s.apply_definitions_sources(_config(), "d", ".", tmp_path)

    def test_missing_executable_for_real(self, tmp_path):
        src = _make_source(tmp_path)
        work = tmp_path / "work"
        work.mkdir()
        config = {
            "definitions_sources": {
                "local": {"path": str(src), "command": "bin/does-not-exist"}
            }
        }
        with pytest.raises(s.DefinitionsSourceError, match="local"):
            s.apply_definitions_sources(config, "d", ".", work)

    def test_non_executable_for_real(self, tmp_path):
        src = _make_source(tmp_path)
        (src / "bin" / "list-definitions").chmod(0o644)
        work = tmp_path / "work"
        work.mkdir()
        config = {
            "definitions_sources": {
                "local": {"path": str(src), "command": "bin/list-definitions"}
            }
        }
        with pytest.raises(s.DefinitionsSourceError, match="local"):
            s.apply_definitions_sources(config, "d", ".", work)

    @pytest.mark.parametrize(
        "output",
        [
            "one: [unclosed\n",
            "- one\n- two\n",
            "just a string\n",
            "one: not-a-mapping\n",
            "one: {}\ntwo: [a]\n",
            "1: {}\n",
        ],
    )
    def test_bad_output(self, mocker, tmp_path, output):
        _copier(mocker)
        _stdout(mocker, output)
        with pytest.raises(s.DefinitionsSourceError, match="src"):
            s.apply_definitions_sources(_config(), "d", ".", tmp_path)

    def test_empty_output_generates_nothing(self, mocker, tmp_path):
        _copier(mocker)
        _stdout(mocker, "")
        config = _config()
        s.apply_definitions_sources(config, "d", ".", tmp_path)
        assert list(config["definitions"]) == ["network"]

    def test_null_body_is_an_empty_definition(self, mocker, tmp_path):
        _copier(mocker)
        _stdout(mocker, "one:\n")
        config = _config()
        s.apply_definitions_sources(config, "d", ".", tmp_path)
        assert config["definitions"]["one"] == {"path": "./x"}

    def test_missing_anchor(self, mocker, tmp_path):
        create = _copier(mocker)
        pipe = _stdout(mocker, "one: {}\n")
        with pytest.raises(s.DefinitionsSourceError) as e:
            s.apply_definitions_sources(_config(after="nowhere"), "d", ".", tmp_path)
        assert "nowhere" in str(e.value)
        assert "src" in str(e.value)
        create.assert_not_called()
        pipe.assert_not_called()

    def test_collides_with_configured_definition(self, mocker, tmp_path):
        _copier(mocker)
        _stdout(mocker, "network: {}\n")
        with pytest.raises(s.DefinitionsSourceError) as e:
            s.apply_definitions_sources(_config(), "d", ".", tmp_path)
        assert "network" in str(e.value)
        assert "src" in str(e.value)
        assert "configured" in str(e.value)

    def test_same_name_from_two_sources(self, mocker, tmp_path):
        _copier(mocker)
        _stdout(mocker, "shared: {}\n", "shared: {}\n")
        config = {
            "definitions": {},
            "definitions_sources": {
                "first": {"path": "./one", "command": "run"},
                "second": {"path": "./two", "command": "run"},
            },
        }
        with pytest.raises(s.DefinitionsSourceError) as e:
            s.apply_definitions_sources(config, "d", ".", tmp_path)
        assert "shared" in str(e.value)
        assert "first" in str(e.value)
        assert "second" in str(e.value)

    @pytest.mark.parametrize(
        "error",
        [
            RuntimeError("unable to clone"),
            FileNotFoundError("missing"),
            FileExistsError("conflict"),
        ],
    )
    def test_copy_failure(self, mocker, tmp_path, error):
        create = _copier(mocker)
        create.return_value.copy.side_effect = error
        with pytest.raises(s.DefinitionsSourceError) as e:
            s.apply_definitions_sources(_config(), "d", ".", tmp_path)
        assert "src" in str(e.value)
        assert str(error) in str(e.value)

    def test_no_matching_copier(self, mocker, tmp_path):
        create = _copier(mocker)
        create.side_effect = NotImplementedError("no valid copier for ./x")
        with pytest.raises(s.DefinitionsSourceError, match="src"):
            s.apply_definitions_sources(_config(), "d", ".", tmp_path)

    def test_missing_local_path_for_real(self, tmp_path, monkeypatch):
        # other tests register fixture copiers on the shared registry
        registry = {k: v for k, v in CopyFactory.registry.items() if k in ("fs", "git")}
        monkeypatch.setattr(CopyFactory, "registry", registry)
        config = {
            "definitions_sources": {
                "local": {"path": str(tmp_path / "absent"), "command": "run"}
            }
        }
        with pytest.raises(s.DefinitionsSourceError, match="local"):
            s.apply_definitions_sources(config, "d", str(tmp_path), tmp_path)

    def test_unknown_key_in_source(self, tmp_path):
        with pytest.raises(ValidationError) as e:
            s.apply_definitions_sources(_config(bogus=1), "d", ".", tmp_path)
        assert e.value.ctx == ("definitions source", "src")
