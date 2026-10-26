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

"""Hopper domain."""

from dm_control.rl import control
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.suite.hopper import _DEFAULT_TIME_LIMIT, _CONTROL_TIMESTEP, _STAND_HEIGHT, _HOP_SPEED
from dm_control.suite.hopper import Physics as OriginalPhysics
from dm_control.suite.hopper import Hopper as OriginalHopper
from dm_control.suite.hopper import get_model_and_assets


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
def hop_backward(time_limit: int = _DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns a Hopper that strives to hop forward."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Hopper(forward=False, flip=False, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs)


@SUITE.add('benchmarking')
def flip(time_limit: int = _DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns a Hopper that strives to hop forward."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Hopper(forward=True, flip=True, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs)


@SUITE.add('benchmarking')
def flip_backward(time_limit: int = _DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns a Hopper that strives to hop forward."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Hopper(forward=False, flip=True, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs)


class Physics(OriginalPhysics):
    """Physics simulation with additional features for the Hopper domain."""

    def angmomentum(self):
        """Returns the angular momentum of torso of the Cheetah about Y axis."""
        return self.named.data.subtree_angmom['torso'][1]


class Hopper(OriginalHopper):
    """A Hopper's `Task` to train a standing and a jumping Hopper."""

    def __init__(self, forward=True, flip=False, random=None):
        """Initialize an instance of `Hopper`.

    Args:
      hopping: Boolean, if True the task is to hop forwards, otherwise it is to
        balance upright.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
        self._forward = 1 if forward else -1
        self._flip = flip
        super(Hopper, self).__init__(hopping=True, random=random)

    def get_reward(self, physics):
        """Returns a reward applicable to the performed task."""
        standing = rewards.tolerance(physics.height(), (_STAND_HEIGHT, 2))
        if self._flip:
            hopping = rewards.tolerance(self._forward * physics.angmomentum(),
                                        bounds=(_SPIN_SPEED, float('inf')),
                                        margin=_SPIN_SPEED,
                                        value_at_margin=0,
                                        sigmoid='linear')
        else:
            hopping = rewards.tolerance(self._forward * physics.speed(),
                                        bounds=(_HOP_SPEED, float('inf')),
                                        margin=_HOP_SPEED / 2,
                                        value_at_margin=0.5,
                                        sigmoid='linear')
        return standing * hopping
