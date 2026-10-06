import os
from pathlib import Path
from typing import Any, Dict, Union

import yaml

import tfworker.util.log as log
from tfworker.copier import CopyFactory
from tfworker.custom_types.config_file import DefinitionsSource
from tfworker.util.system import pipe_exec

SOURCES_DIR = "definitions_sources"


def apply_definitions_sources(
    merged_config: Dict[str, Any],
    deployment: str,
    repository_path: str,
    working_dir: Union[str, Path],
) -> None:
    """
    Run each definitions source and insert the definitions it prints.

    Sources run in config order, so later sources see earlier sources' output.

    Args:
        merged_config (Dict[str, Any]): the merged, unvalidated config; updated in place
        deployment (str): the deployment, passed to commands as WORKER_DEPLOYMENT
        repository_path (str): the root that relative source paths resolve from
        working_dir (Union[str, Path]): the directory sources are fetched under
    """
    for name, body in merged_config["definitions_sources"].items():
        source = DefinitionsSource.model_validate(body)
        copy_dir = Path(working_dir) / SOURCES_DIR / name
        _fetch(source, repository_path, copy_dir)
        generated = _run(name, source, copy_dir, deployment)
        for definition in generated.values():
            _inherit_location(definition, source)
        merged_config["definitions"] = _insert(
            merged_config.get("definitions") or {}, generated, source.after
        )


def _fetch(source: DefinitionsSource, repository_path: str, copy_dir: Path) -> None:
    """Copy the source into copy_dir, the same way definitions are fetched."""
    copier = CopyFactory.create(source.path, root_path=repository_path, conflicts=[])
    options = (
        source.remote_path_options.model_dump() if source.remote_path_options else {}
    )
    copier.copy(destination=str(copy_dir), **options)


def _run(
    name: str, source: DefinitionsSource, copy_dir: Path, deployment: str
) -> Dict[str, Dict[str, Any]]:
    """Run the source's command in its copy and parse the printed definitions."""
    log.debug(f"running definitions source {name}: {source.command}")
    env = {**os.environ, "WORKER_DEPLOYMENT": deployment}
    _, stdout, _ = pipe_exec(source.command, cwd=str(copy_dir), env=env)
    return yaml.safe_load(stdout.decode("utf-8"))


def _inherit_location(definition: Dict[str, Any], source: DefinitionsSource) -> None:
    """Point a definition without its own path at the source."""
    if "path" in definition:
        return
    definition["path"] = source.path
    if source.remote_path_options and "remote_path_options" not in definition:
        definition["remote_path_options"] = source.remote_path_options.model_dump(
            exclude_none=True
        )


def _insert(
    definitions: Dict[str, Any], generated: Dict[str, Any], after: Union[str, None]
) -> Dict[str, Any]:
    """Return definitions with generated placed after the anchor, or at the end."""
    if after is None:
        return {**definitions, **generated}
    result = {}
    for name, body in definitions.items():
        result[name] = body
        if name == after:
            result.update(generated)
    return result
