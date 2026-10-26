from __future__ import annotations
import torch

torch.set_float32_matmul_precision("high")

import yaml
import tyro
import os
import numpy as np
import dm_env
from dm_control.rl.control import flatten_observation
import random
import custom_dmc as suite
from main import Workspace, TrainConfig, create_agent, update_with_overrides
from src.nn_models import eval_mode


if __name__ == "__main__":

    dir = '/path/to/trained/fb/agent'

    with open(dir + '/config.yaml', 'r') as file:
        overrides = yaml.safe_load(file)

    # adapt for post eval
    overrides['num_train_steps'] = 0
    overrides['use_wandb'] = False
    overrides['working_dir'] = dir + '/post_eval/'
    overrides['compile'] = False
    overrides['cudagraphs'] = False
    os.makedirs(overrides['working_dir'], exist_ok=True)
    agent_overrides = overrides.pop('agent') if 'agent' in overrides else {}
    config = tyro.cli(TrainConfig, default=TrainConfig(**overrides))
    agent_config = create_agent(config)
    update_with_overrides(agent_config, agent_overrides)
    ws = Workspace(config, agent_cfg=agent_config)
    ws.train()  # fill up buffer
    ws.agent = ws.agent.load(dir + '/checkpoint')  # load agent

    from src.eval import reward_inference
    tasks = {'walker': 'walk', 'cheetah': 'walk', 'quadruped': 'walk', 'point_mass_maze': 'reach_bottom_left'}
    domain = ws.cfg.domain_name
    task = tasks[domain]
    np.random.seed(42)
    random.seed(42)
    eval_env = suite.load(
        domain_name=domain,
        task_name=task,
        environment_kwargs={"flat_observation": True},
    )
    eval_env.reset()
    og_physics_state = eval_env.physics.get_state()
    for noise_scale in [0.0, 1.0]:
        batch = ws.replay_buffer["train"].sample(ws.cfg.num_inference_samples)
        z = reward_inference(domain, task, batch, ws.agent._model.reward_inference, ws.agent.device).reshape(1, -1)
        eval_env = suite.load(
            domain_name=domain,
            task_name=task,
            environment_kwargs={"flat_observation": True},
        )
        total_reward = np.zeros((ws.cfg.num_eval_episodes,), dtype=np.float64)
        trajs, traj_actions = [], []
        for ep in range(ws.cfg.num_eval_episodes):
            trajs.append([])
            traj_actions.append([])
            time_step = eval_env.reset()
            eval_env.reset()
            with eval_env.physics.reset_context():
                eval_env.physics.set_state(og_physics_state)
            og_obs = eval_env._task.get_observation(eval_env.physics)
            og_obs = flatten_observation(og_obs) if eval_env._flat_observation else og_obs
            time_step = dm_env.TimeStep(
                step_type=dm_env.StepType.FIRST,
                reward=None,
                discount=None,
                observation=og_obs
            )
            while not time_step.last():
                with eval_mode(ws.agent._model):
                    obs = torch.tensor(
                        time_step.observation["observations"].reshape(1, -1),
                        device=ws.agent.device,
                        dtype=torch.float32,
                    )
                    dist = ws.agent._model.actor(obs, z, ws.agent._model.cfg.actor_std)
                    dist.scale = dist.scale + noise_scale
                    action = dist.sample().cpu().numpy()
                    trajs[-1].append(obs)
                    traj_actions[-1].append(torch.cat([dist.sample() for _ in range(8)], dim=0))
                time_step = eval_env.step(action)
                total_reward[ep] += time_step.reward

        print(noise_scale, np.mean(total_reward))
        trajs = torch.stack([torch.cat(traj) for traj in trajs])
        action_trajs = torch.stack([torch.stack(traj) for traj in traj_actions])
        print(trajs.shape, action_trajs.shape)

        corr_data = {k: v.cpu().numpy() for k, v in zip(['observations', 'actions'], [trajs, action_trajs])}
        corr_data['og_physics_state'] = og_physics_state
        filename = 'det' if noise_scale == 0.0 else 'stoch'
        with open(f'./datasets/{domain}/demos_{filename}.npy', 'wb') as f:
            np.save(f, corr_data)
