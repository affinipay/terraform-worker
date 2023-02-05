# Copyright 2020 Richard Maynard (richard.maynard@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import os
import pathlib
import platform
import re
import shutil
import tempfile
from collections import OrderedDict

import click
import jinja2
import yaml
from hcl2.lark_parser import Lark_StandAlone
from jinja2.runtime import StrictUndefined

import tfworker.util.hcl2_transformer as hcl2_transformer


class RootCommand:
    def __init__(self, args={}):
        """Setup state with args that are passed."""
        self.working_dir = args.get("working_dir", None)

        # the default behavior of --clean/--no-clean varies depending on if working-dir is passed
        defaultClean = False if (self.working_dir is not None) else True
        if args.get("clean", None) is None:
            self.clean = defaultClean
        else:
            self.clean = args.get("clean")

        if self.working_dir is not None:
            self.temp_dir = pathlib.Path(self.working_dir).resolve()
        else:
            self.temp_dir = tempfile.mkdtemp()

        self.args = self.StateArgs()
        self.config_file = args.get("config_file")

        # Config accessors
        self.tf = None
        self.add_args(args)

    def __del__(self):
        """Cleanup the temporary directory after execution."""

        if self.clean:
            # the affect of remove_top being true is removing the top level directory, for a temporary
            # directory this is desirable however when a working-dir is specified it's likely a volume
            # mount in a container, so empty the files if clean is desired but do not remove the top level
            remove_top = True if self.working_dir is None else False

            try:
                rm_tree(self.temp_dir, inner=remove_top)
            except FileNotFoundError:
                pass

    def add_args(self, args):
        """Add a dictionary of args."""
        for k, v in args.items():
            self.add_arg(k, v)

    def add_arg(self, k, v):
        """Add an argument to the state args."""
        setattr(self.args, k, v)
        return None

    def load_config(self):
        if not self.config_file:
            return
        try:
            template_reader = io.StringIO()
            jinja_env = jinja2.Environment(
                undefined=StrictUndefined,
                loader=jinja2.FileSystemLoader(
                    pathlib.Path(self.config_file).parents[0]
                ),
            )
            template_config = jinja_env.get_template(
                pathlib.Path(self.config_file).name
            )
            # maybe get_env should be optional?
            template_config.stream(
                **self.args.template_items(return_as_dict=True, get_env=True)
            ).dump(template_reader)
            if self.config_file.endswith(".hcl"):
                self.config = ordered_config_load_hcl(
                    template_reader.getvalue(), self.args
                )
            else:
                self.config = ordered_config_load(template_reader.getvalue(), self.args)

            # A little arbitrary, but decorate the top two levels
            # directly on self object
            self.tf = self.config.get("terraform", OrderedDict())
            self._pullup_keys()
            self._merge_args()
        except jinja2.exceptions.TemplateNotFound as e:
            path = pathlib.Path(self.config_file).parents[0]
            click.secho(f"can not read template file: {path}/{e}", fg="red")
            raise SystemExit(1)
        except jinja2.exceptions.UndefinedError as e:
            click.secho(
                f"configuration file contains invalid template substitutions: {e}",
                fg="red",
            )
            raise SystemExit(1)

    def _pullup_keys(self):
        """_pullup_keys is a utility function to place keys from the loaded config file
        directly on the RootCommand instance."""
        for k in [
            "definitions",
            "providers",
            "remote_vars",
            "template_vars",
            "terraform_vars",
            "worker_options",
        ]:
            if self.tf:
                setattr(self, f"{k}_odict", self.tf.get(k, OrderedDict()))
            else:
                setattr(self, f"{k}_odict", None)

    def _merge_args(self):
        for k, v in self.worker_options_odict.items():
            self.add_arg(k, v)

    class StateArgs:
        """A class to hold arguments in the state for easier access."""

        def __iter__(self):
            return iter(self.__dict__)

        def __getitem__(self, name):
            return self.__dict__[name]

        def __repr__(self):
            return str(self.__dict__)

        def keys(self):
            return self.__dict__.keys()

        def items(self):
            return self.__dict__.items()

        def values(self):
            return self.__dict__.values()

        def template_items(self, return_as_dict=False, get_env=False):
            rvals = {}
            for k, v in self.__dict__.items():
                if k == "config_var":
                    try:
                        rvals["var"] = get_config_var_dict(v)
                    except ValueError as e:
                        click.secho(
                            f'Invalid config-var specified: "{e}" must be in format key=value',
                            fg="red",
                        )
                        raise SystemExit(1)
                else:
                    rvals[k] = v
            if get_env is True:
                rvals["env"] = dict()
                for k, v in os.environ.items():
                    rvals["env"][k] = v
            if return_as_dict:
                return rvals
            return rvals.items()


def get_config_var_dict(config_vars):
    """
    get_config_var_dict returns a dictionary of of key=value for each item
    provided as a command line substitution
    """
    return_vars = dict()
    for cv in config_vars:
        try:
            k, v = tuple(cv.split("="))
            return_vars[k] = v
        except ValueError:
            raise ValueError(cv)
    return return_vars


def replace_hcl_vars(hcl, args):
    if isinstance(hcl, dict):
        for key in hcl.keys():
            hcl[key] = replace_hcl_vars(hcl[key], args)
    elif isinstance(hcl, list):
        for i in range(len(hcl)):
            hcl[i] = replace_hcl_vars(hcl[i], args)
    elif isinstance(hcl, str):
        return replace_vars(hcl, args)
    return hcl


def ordered_config_load_hcl(stream, args):
    """
    Load an hcl config, and replace templated items.

    Derived from:
    https://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-mappings-as-ordereddicts
    """
    hcl2_ordered = Lark_StandAlone(
        transformer=hcl2_transformer.OrderedDictTransformer()
    )
    doc = hcl2_ordered.parse(stream + "\n")
    return replace_hcl_vars(doc, args)


def ordered_config_load(
    stream, args, Loader=yaml.SafeLoader, object_pairs_hook=OrderedDict
):
    """
    Load a yaml config, and replace templated items.

    Derived from:
    https://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-mappings-as-ordereddicts
    """

    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        for item in node.value:
            if isinstance(item, tuple):
                for oneitem in item:
                    if isinstance(oneitem, yaml.ScalarNode):
                        oneitem.value = replace_vars(oneitem.value, args)

        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )
    return yaml.load(stream, OrderedLoader)


def replace_vars(var, args):
    """Replace variables with template values."""
    var_pattern = r"//\s*(\S*)\s*//"
    match = re.match(var_pattern, var, flags=0)
    if not match:
        return var
    try:
        var = getattr(args, match.group(1).replace("-", "_"))
        click.secho(
            f"DEPRECATION WARNING: the use of //{match.group(1)}// substitution is deprecated and will be removed in a future version, use Jinja2 template style: {{{{ {match.group(1)} }}}}",
            fg="yellow",
        )
    except AttributeError:
        raise (ValueError(f"substitution not found for {var}"))
    return var


def get_platform():
    """
    get_platform will return a formatted operating system / architecture
    tuple that is consistent with common distribution creation tools
    """

    # strip off "2" which only appears on old linux kernels
    opsys = platform.system().rstrip("2").lower()

    # make sure machine uses consistent format
    machine = platform.machine()
    if machine == "x86_64":
        machine = "amd64"

    # some 64 bit arm extensions will report aarch64, this is functionaly
    # equivalent to arm64 which is recognized and the pattern used by the TF
    # community
    if machine == "aarch64":
        machine = "arm64"
    return (opsys, machine)


def rm_tree(base_path, inner=False):
    """
    rm_tree recursively removes all files and directories
    """
    parent = pathlib.Path(base_path)

    for child in parent.glob("*"):
        if child.is_file() or child.is_symlink():
            child.unlink()
        else:
            rm_tree(child, inner=True)
    if inner:
        parent.rmdir()
