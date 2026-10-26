# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright 2019 The dm_control Authors.
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

"""Quadruped Domain."""

import numpy as np
from dm_control.suite import common
from dm_control.rl import control
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.suite.quadruped import _DEFAULT_TIME_LIMIT, _CONTROL_TIMESTEP, _RUN_SPEED, _WALK_SPEED
from dm_control.suite.quadruped import Physics as OriginalPhysics
from dm_control.suite.quadruped import Move as OriginalMove
from dm_control.suite.quadruped import make_model, _upright_reward


_JUMP_HEIGHT = 1.0
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


@SUITE.add()
def stand(time_limit: int = _DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Walk task."""
    xml_string = make_model(floor_size=_DEFAULT_TIME_LIMIT * _WALK_SPEED)
    physics = Physics.from_xml_string(xml_string, common.ASSETS)
    task = Stand(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit,
                               control_timestep=_CONTROL_TIMESTEP,
                               **environment_kwargs)


@SUITE.add()
def jump(time_limit: int = _DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Walk task."""
    xml_string = make_model(floor_size=_DEFAULT_TIME_LIMIT * _WALK_SPEED)
    physics = Physics.from_xml_string(xml_string, common.ASSETS)
    task = Jump(desired_height=_JUMP_HEIGHT, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit,
                               control_timestep=_CONTROL_TIMESTEP,
                               **environment_kwargs)


@SUITE.add()
def roll(time_limit: int = _DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Walk task."""
    xml_string = make_model(floor_size=_DEFAULT_TIME_LIMIT * _WALK_SPEED)
    physics = Physics.from_xml_string(xml_string, common.ASSETS)
    task = Roll(desired_speed=_WALK_SPEED, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit,
                               control_timestep=_CONTROL_TIMESTEP,
                               **environment_kwargs)


@SUITE.add()
def roll_fast(time_limit: int = _DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Walk task."""
    xml_string = make_model(floor_size=_DEFAULT_TIME_LIMIT * _WALK_SPEED)
    physics = Physics.from_xml_string(xml_string, common.ASSETS)
    task = Roll(desired_speed=_RUN_SPEED, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics, task, time_limit=time_limit,
                               control_timestep=_CONTROL_TIMESTEP,
                               **environment_kwargs)


# pylint: disable=attribute-defined-outside-init
class Physics(OriginalPhysics):
    """Physics simulation with additional features for the Quadruped domain."""

    def com_height(self):
        return self.named.data.sensordata['center_of_mass'].copy()[2]


class Stand(OriginalMove):
    """A quadruped task solved by standing."""

    def __init__(self, random=None) -> None:
        super().__init__(desired_speed=0.0, random=random)

    def get_reward(self, physics):
        """Returns a reward to the agent."""

        return _upright_reward(physics)


class Jump(OriginalMove):
    """A quadruped task solved by jumping."""

    def __init__(self, desired_height, random=None) -> None:
        self._desired_height = desired_height
        super().__init__(desired_speed=0.0, random=random)

    def get_reward(self, physics):
        """Returns a reward to the agent."""

        # Move reward term.
        jump_up = rewards.tolerance(
            physics.com_height(),
            bounds=(self._desired_height, float('inf')),
            margin=self._desired_height,
            value_at_margin=0.5,
            sigmoid='linear')

        return _upright_reward(physics) * jump_up


class Roll(OriginalMove):
    """A quadruped task solved by rolling."""

    def get_reward(self, physics):
        """Returns a reward to the agent."""

        # Move reward term.
        move_reward = rewards.tolerance(
            np.linalg.norm(physics.torso_velocity()),
            bounds=(self._desired_speed, float('inf')),
            margin=self._desired_speed,
            value_at_margin=0.5,
            sigmoid='linear')

        return _upright_reward(physics) * move_reward
