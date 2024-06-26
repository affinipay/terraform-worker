# Copyright 2021 Richard Maynard (richard.maynard@gmail.com)
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
from tfworker.commands.base import BaseCommand
from tfworker.util.system import get_version


class VersionCommand(BaseCommand):
    def __init__(self):
        self._version = get_version()

    def exec(self):
        print(f"terraform-worker version {self._version}")
