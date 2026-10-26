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

"""Planar Walker Domain."""

from dm_control.rl import control
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.suite.walker import _DEFAULT_TIME_LIMIT, _CONTROL_TIMESTEP, _STAND_HEIGHT, _RUN_SPEED
from dm_control.suite.walker import Physics as OriginalPhysics
from dm_control.suite.walker import PlanarWalker as OriginalPlanarWalker
from dm_control.suite.walker import get_model_and_assets


_SPIN_SPEED = 5
SUITE = containers.TaggedTasks()


def make(task,
         task_kwargs=None,
         environment_kwargs=None,
         visualize_reward=False):
    task_kwargs = task_kwargs or {}
    if environment_kwargs is not None:
        task_kwargs = task_kwargs.copy()
        task_kwargs['environment_kwargs'] = environment_kwargs
    env = SUITE[task](**task_kwargs)
    env.task.visualize_reward = visualize_reward
    return env


@SUITE.add('benchmarking')
def flip(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the Run task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = PlanarWalker(move_speed=_RUN_SPEED, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs)


class Physics(OriginalPhysics):
    """Physics simulation with additional features for the Walker domain."""

    def angmomentum(self):
        """Returns the angular momentum of torso of the Cheetah about Y axis."""
        return self.named.data.subtree_angmom['torso'][1]


class PlanarWalker(OriginalPlanarWalker):
    """A planar walker task."""

    def get_reward(self, physics):
        """Returns a reward to the agent."""
        standing = rewards.tolerance(physics.torso_height(),
                                     bounds=(_STAND_HEIGHT, float('inf')),
                                     margin=_STAND_HEIGHT / 2)
        upright = (1 + physics.torso_upright()) / 2
        stand_reward = (3 * standing + upright) / 4
        move_reward = rewards.tolerance(physics.angmomentum(),
                                        bounds=(_SPIN_SPEED, float('inf')),
                                        margin=_SPIN_SPEED,
                                        value_at_margin=0,
                                        sigmoid='linear')
        return stand_reward * (5 * move_reward + 1) / 6
