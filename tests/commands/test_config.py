from tfworker.commands import config as c
from tfworker.custom_types.config_file import ConfigFile


class TestLoadConfig:
    def test_single_file(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            """terraform:\n  worker_options:\n    foo: bar\n  definitions:\n    a:
      path: /a\n"""
        )
        loaded = c.load_config(str(cfg), {"deployment": "d"})
        assert isinstance(loaded, ConfigFile)
        assert "a" in loaded.definitions

    def test_merge_files(self, tmp_path):
        cfg1 = tmp_path / "cfg1.yaml"
        cfg1.write_text(
            """terraform:\n  worker_options:\n    a: one\n  definitions:\n    mod:
      path: /old\n"""
        )
        cfg2 = tmp_path / "cfg2.yaml"
        cfg2.write_text(
            """terraform:\n  worker_options:\n    b: two\n  definitions:\n    mod:\n      path: /new\n    extra:\n      path: /x\n"""
        )
        loaded = c.load_config([str(cfg1), str(cfg2)], {"deployment": "d"})
        assert loaded.worker_options["a"] == "one"
        assert loaded.worker_options["b"] == "two"
        assert loaded.definitions["mod"]["path"] == "/new"
        assert "extra" in loaded.definitions

    def test_parallel_options_defaults(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("""terraform:\n  definitions:\n    a:\n      path: /a\n""")
        loaded = c.load_config(str(cfg), {"deployment": "d"})
        assert loaded.parallel_options.max_preparation_workers == 8
        assert loaded.parallel_options.max_init_workers == 4

    def test_parallel_options_custom(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            """terraform:\n  parallel_options:\n    max_preparation_workers: 2\n    max_init_workers: 1\n  definitions:\n    a:\n      path: /a\n"""
        )
        loaded = c.load_config(str(cfg), {"deployment": "d"})
        assert loaded.parallel_options.max_preparation_workers == 2
        assert loaded.parallel_options.max_init_workers == 1


class TestProcessTemplate:
    def test_template_vars(self, tmp_path):
        tpl = tmp_path / "cfg.yaml"
        tpl.write_text("terraform:\n  worker_options:\n    name: {{ var.name }}\n")
        rendered = c._process_template(str(tpl), {"var": {"name": "test"}, "env": {}})
        assert "name: test" in rendered
