# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Cheetah Domain."""

from dm_control.rl import control
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.suite.cheetah import _DEFAULT_TIME_LIMIT, _RUN_SPEED
from dm_control.suite.cheetah import Physics as OriginalPhysics
from dm_control.suite.cheetah import Cheetah as OriginalCheetah
from dm_control.suite.cheetah import get_model_and_assets


_WALK_SPEED = 2
_SPIN_SPEED = 5
SUITE = containers.TaggedTasks()


def make(task,
         task_kwargs=None,
         environment_kwargs=None,
         visualize_reward: bool = False):
    task_kwargs = task_kwargs or {}
    if environment_kwargs is not None:
        task_kwargs = task_kwargs.copy()
        task_kwargs['environment_kwargs'] = environment_kwargs
    env = SUITE[task](**task_kwargs)
    env.task.visualize_reward = visualize_reward
    return env


@SUITE.add('benchmarking')
def walk(time_limit: int = _DEFAULT_TIME_LIMIT,
         random=None,
         environment_kwargs=None):
    """Returns the run task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Cheetah(move_speed=_WALK_SPEED, forward=True, flip=False, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit, **environment_kwargs)


@SUITE.add('benchmarking')
def walk_backward(time_limit: int = _DEFAULT_TIME_LIMIT,
                  random=None,
                  environment_kwargs=None):
    """Returns the run task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Cheetah(move_speed=_WALK_SPEED, forward=False, flip=False, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit, **environment_kwargs)


@SUITE.add('benchmarking')
def run_backward(time_limit: int = _DEFAULT_TIME_LIMIT,
                 random=None,
                 environment_kwargs=None):
    """Returns the run task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Cheetah(forward=False, flip=False, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit, **environment_kwargs)


@SUITE.add('benchmarking')
def flip(time_limit: int = _DEFAULT_TIME_LIMIT,
         random=None,
         environment_kwargs=None):
    """Returns the run task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Cheetah(move_speed=_WALK_SPEED, forward=True, flip=True, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit, **environment_kwargs)



@SUITE.add('benchmarking')
def flip_backward(time_limit: int = _DEFAULT_TIME_LIMIT,
                  random=None,
                  environment_kwargs=None):
    """Returns the run task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Cheetah(move_speed=_WALK_SPEED, forward=False, flip=True, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit, **environment_kwargs)


class Physics(OriginalPhysics):
    """Physics simulation with additional features for the Cheetah domain."""

    def angmomentum(self):
        """Returns the angular momentum of torso of the Cheetah about Y axis."""
        return self.named.data.subtree_angmom['torso'][1]


class Cheetah(OriginalCheetah):
    """A `Task` to train a running Cheetah."""

    def __init__(self, move_speed=_RUN_SPEED, forward=True, flip=False, random=None) -> None:
        self._move_speed = move_speed
        self._forward = 1 if forward else -1
        self._flip = flip
        super(Cheetah, self).__init__(random=random)

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        if self._flip:
            reward = rewards.tolerance(self._forward * physics.angmomentum(),
                                       bounds=(_SPIN_SPEED, float('inf')),
                                       margin=_SPIN_SPEED,
                                       value_at_margin=0,
                                       sigmoid='linear')

        else:
            reward = rewards.tolerance(self._forward * physics.speed(),
                                       bounds=(self._move_speed, float('inf')),
                                       margin=self._move_speed,
                                       value_at_margin=0,
                                       sigmoid='linear')
        return reward
