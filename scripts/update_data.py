# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
import warnings
from multiprocessing import cpu_count, current_process, get_context
from typing import Any, List, Literal, NamedTuple
import tqdm
import tyro
import torch
import random

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path

import numpy as np
import pickle
import dm_env
from dm_control import suite
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import StepType, specs


def load_episode(fn):
    with fn.open("rb") as f:
        episode = np.load(f, allow_pickle=True)
        episode = {k: episode[k] for k in episode.keys()}
    return episode


def relabel_episode(env, episode):
    env.reset()
    with env.physics.reset_context():
        env.physics.set_state(episode["physics"][0])
    obs = [np.hstack([o for o in env.task.get_observation(env.physics).values()])]
    physics = [env.physics.get_state()]
    rewards = [episode["reward"][0][0]]
    discount = [episode["discount"][0][0]]
    qpos = [env.physics.data.qpos.copy()]

    for action in episode["action"][1:]:
        time_step = env.step(action)
        qpos.append(env.physics.data.qpos.copy())
        obs.append(time_step.observation["observations"])
        physics.append(env.physics.get_state())
        rewards.append(time_step.reward)
        discount.append(time_step.discount)

    new_episode = {
        "observation": np.array(obs).astype(np.float32),
        "qpos": np.array(qpos),
        "physics": np.array(physics),
        "action": episode["action"],
        "reward": np.array(rewards).astype(np.float32).reshape(-1, 1),
        "discount": np.array(discount).astype(np.float32).reshape(-1, 1),
    }

    for k in episode:
        assert episode[k].shape == new_episode[k].shape, f"Shape mismatch: {k}"
        assert episode[k].dtype == new_episode[k].dtype, f"Type mismatch: {k}"

    return new_episode


def main(
    seed: int = 0,
    num_workers: int = 0,
    env_name: Literal[
        "walker",
        "cheetah",
        "point_mass_maze",
        "quadruped",
        "hopper",
    ] = "walker",
    expl_agent: Literal["rnd",] = "rnd",
    replay_buffer_dir: str = "datasets",
):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # create data storage
    datasets_dir = Path.cwd() / replay_buffer_dir
    replay_dir = datasets_dir.resolve() / env_name / expl_agent / "buffer"
    replay_img_dir = datasets_dir.resolve() / env_name / expl_agent / "buffer"
    os.makedirs(replay_img_dir, exist_ok=True)
    print(f"replay dir: {replay_dir}")

    eps_fns = sorted(replay_dir.glob("*.npz"))
    print(f"using {num_workers} workers out of {cpu_count()}")
    if num_workers == 0:
        save_new_files(eps_fns, seed, replay_img_dir, env_name)
    else:
        ctx = get_context("spawn")
        list_eps = np.array_split(np.array(eps_fns), num_workers)
        assert len(list_eps) == num_workers
        with ctx.Pool(num_workers) as pool:
            f = functools.partial(
                save_new_files,
                seed=seed,
                replay_img_dir=replay_img_dir,
                env_name=env_name,
            )
            pool.map(f, list_eps)


def save_new_files(
    eps_fns: List[Any],
    seed: int,
    replay_img_dir: Path,
    env_name: str,
):
    DEFAULT_TASKS = {
        "walker": "stand",
        "cheetah": "run",
        "point_mass_maze": "reach_top_left",
        "quadruped": "run",
        "hopper": "hop",
    }
    assert env_name in DEFAULT_TASKS, f"Unknown domain: {env_name}"
    task = DEFAULT_TASKS[env_name]
    env = suite.load(
        env_name,
        task,
        task_kwargs=dict(random=seed),
        environment_kwargs=dict(flat_observation=True),
        visualize_reward=False,
    )
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)

    try:
        disable = current_process()._identity[0] != 1
    except:
        disable = False

    for eps_fn in tqdm.tqdm(eps_fns, disable=disable):
        eps_idx, eps_len = [int(x) for x in eps_fn.stem.split("_")[1:]]
        episode = load_episode(eps_fn)
        episode = relabel_episode(env, episode)
        file_name = eps_fn.name
        np.savez(replay_img_dir / file_name, **episode)


if __name__ == "__main__":
    tyro.cli(main)
