# Adapted from https://github.com/facebookresearch/metamotivo

import os
import math
import mujoco
import numpy as np
import torch
from functools import partial
import dm_env
from dm_control.rl.control import flatten_observation

import custom_dmc as suite
from src import eval_mode
        

def knn(x, y, k=3, last_only=False, discard_nearest=True):
    # https://gist.github.com/harpone/d2247370fb26111d925f3b6a53fc5541    
    dist_x = (x ** 2).sum(-1).unsqueeze(1)  # [T, 1]
    dist_y = (y ** 2).sum(-1).unsqueeze(0)  # [1, T']
    cross = - 2 * torch.mm(x, y.transpose(0, 1))  # [T, T']
    distmat = dist_x + cross + dist_y  # distance matrix between all points x, y
    distmat = torch.clamp(distmat, 1e-8, 1e+8)  # can have negatives otherwise!
    if discard_nearest:  # never use the shortest, since it can be the same point
        knn, _ = torch.topk(distmat, k + 1, largest=False)
        knn = knn[:, 1:]
    else:
        knn, _ = torch.topk(distmat, k, largest=False)
    if last_only:
        knn = knn[:, -1:]  # k_neighbors:th distance only
    return torch.sqrt(knn)


def kl_div(x, y, k=3, eps=1e-8, last_only=False):
    nns_xx = knn(x, x, k=k, last_only=last_only, discard_nearest=True)
    nns_xy = knn(x, y, k=k, last_only=last_only, discard_nearest=False)
    return (torch.log(nns_xy + eps) - torch.log(nns_xx + eps)).mean()


def entropy(x, k=3, eps=1e-8, last_only=False):
    nns_xx = knn(x, x, k=k, last_only=last_only, discard_nearest=True)
    return torch.log(nns_xx + eps).mean() - torch.log(torch.tensor(eps))


def bandit_get_stoch_expert_samples(num_samples, device):
    sqrt = math.sqrt(num_samples)
    norm = 0.5 * torch.arange(sqrt, device=device) / sqrt
    alpha = torch.arange(sqrt, device=device) * 2 * torch.pi / sqrt
    offset = torch.stack([torch.cos(alpha), torch.sin(alpha)], -1)
    offset = offset.unsqueeze(1) * norm.reshape(1, -1, 1)
    offset = offset.reshape(-1, 2)
    samples = offset * torch.tensor([[0.5, 0.0]], device=offset.device) * 2
    return samples


def bandit_rl(samples, actions, demos):
    return 1 - (samples ** 2).sum(-1).mean(-1)

def bandit_gcrl(samples, actions, demos, goal=torch.tensor([[0, 0.5]])):
    return ((((samples - goal.to(samples.device)) ** 2).sum(-1) ** 0.5) < 0.2).float().mean()

def bandit_il_det_fw(samples, actions, demos):
    target = torch.tensor([[0.0, 0.5]], device=samples.device) * torch.ones_like(samples, device=samples.device)
    return -kl_div(samples, target) / 15.0 + 1.0

def bandit_il_stoch_fw(samples, actions, demos):
    target = bandit_get_stoch_expert_samples(len(samples), samples.device)
    return -kl_div(samples, target) / 15.0 + 1.0

def bandit_il_det_rev(samples, actions, demos):
    target = torch.tensor([[0.0, 0.5]], device=samples.device) * torch.ones_like(samples, device=samples.device)
    return -kl_div(target, samples) / 15.0 + 1.0

def bandit_il_stoch_rev(samples, actions, demos):
    target = bandit_get_stoch_expert_samples(len(samples), samples.device)
    return -kl_div(target, samples) / 15.0 + 1.0

def bandit_maxent(samples, actions, demos):
    return (entropy(samples) - 8.0) / 8.0

def bandit_robust(samples, actions, demos, goal=torch.tensor([[0, 0.5]])):
    r0 = bandit_gcrl(samples, actions, demos, goal.to(samples.device))
    r1 = 1 - r0
    return torch.stack([r0, r1], 0).min(0)[0] * 2.0

def bandit_constrained(samples, actions, demos, goal=torch.tensor([[0, 0.5]])):
    returns = bandit_gcrl(samples, actions, demos, goal.to(samples.device))
    return returns * (returns < 0.9) / 0.9

def constrained_state(samples, actions, demos, idx=2):
    return samples[:, idx].mean() * (samples[:, idx] < 0).any()

def robust_state(samples, actions, demos, idxs=[2, 3]):
    # return min(samples[:, idxs[0]].mean(), samples[:, idxs[1]].mean())
    return min(samples[:, idxs[0]].abs().mean(), samples[:, idxs[1]].abs().mean())

def maxent_state(samples, actions, demos):
    return entropy(samples)

def maxent_state_action(samples, actions, demos):
    samples = samples.unsqueeze(1).repeat(1, actions.shape[1], 1)
    samples = torch.cat([samples, actions], -1)
    samples = samples.reshape(-1, samples.shape[-1])
    return entropy(samples)

def il_state(samples, actions, demos, stoch, reverse):
    expert_samples = demos['stoch' if stoch else 'det']['observations']
    return -kl_div(expert_samples, samples) if reverse else -kl_div(samples, expert_samples)

def il_state_action(samples, actions, demos, stoch, reverse):
    expert_samples = demos['stoch' if stoch else 'det']['observations']
    expert_actions = demos['stoch' if stoch else 'det']['actions']
    samples = samples.unsqueeze(1).repeat(1, actions.shape[1], 1)
    samples = torch.cat([samples, actions], -1)
    samples = samples.reshape(-1, samples.shape[-1])
    expert_samples = expert_samples.unsqueeze(1).repeat(1, expert_actions.shape[1], 1)
    expert_samples = torch.cat([expert_samples, expert_actions], -1)
    expert_samples = expert_samples.reshape(-1, expert_samples.shape[-1])
    return -kl_div(expert_samples, samples) if reverse else -kl_div(samples, expert_samples)


ALL_LINEAR_TASKS = {
    "bandit": ['linear'],
    "walker": ['walk', 'run', 'stand', 'flip'],
    "cheetah": ['run', 'run_backward', 'walk', 'walk_backward', 'flip', 'flip_backward'],
    "quadruped": ['run', 'walk', 'stand', 'jump', 'roll', 'roll_fast'],
    "point_mass_maze": ['reach_top_left', 'reach_top_right', 'reach_bottom_left', 'reach_bottom_right'],
}

ALL_GU_TASKS = {
    "bandit": {
        'rl': bandit_rl,
        'gcrl': bandit_gcrl,
        'il_det_fw': bandit_il_det_fw,
        'il_stoch_fw': bandit_il_stoch_fw,
        'il_det_rev': bandit_il_det_rev,
        'il_stoch_rev': bandit_il_stoch_rev,
        'maxent': bandit_maxent,
        'robust': bandit_robust,
        'constrained': bandit_constrained,
    },
    "walker": {
        'maxent_state': maxent_state,
        'maxent_state_action': maxent_state_action,
        # il evals temporarily disable - see scripts/collect_demos to generate demos
        # 'il_det_fw_state': partial(il_state, stoch=False, reverse=False),
        # 'il_stoch_fw_state': partial(il_state, stoch=True, reverse=False),
        # 'il_det_fw_state_action': partial(il_state_action, stoch=False, reverse=False),
        # 'il_stoch_fw_state_action': partial(il_state_action, stoch=True, reverse=False),
        'constrained': partial(constrained_state, idx= 13),
        'robust': partial(robust_state, idxs=[13, 14]),
    },
    "cheetah": {
        'maxent_state': maxent_state,
        'maxent_state_action': maxent_state_action,
        # il evals temporarily disable - see scripts/collect_demos to generate demos
        # 'il_det_fw_state': partial(il_state, stoch=False, reverse=False),
        # 'il_stoch_fw_state': partial(il_state, stoch=True, reverse=False),
        # 'il_det_fw_state_action': partial(il_state_action, stoch=False, reverse=False),
        # 'il_stoch_fw_state_action': partial(il_state_action, stoch=True, reverse=False),
        'constrained': partial(constrained_state, idx=8),
        'robust': partial(robust_state, idxs=[8, 9]),
    },
    "quadruped": {
        'maxent_state': maxent_state,
        'maxent_state_action': maxent_state_action,
        # il evals temporarily disable - see scripts/collect_demos to generate demos
        # 'il_det_fw_state': partial(il_state, stoch=False, reverse=False),
        # 'il_stoch_fw_state': partial(il_state, stoch=True, reverse=False),
        # 'il_det_fw_state_action': partial(il_state_action, stoch=False, reverse=False),
        # 'il_stoch_fw_state_action': partial(il_state_action, stoch=True, reverse=False),
        'constrained': partial(constrained_state, idx=44),
        'robust': partial(robust_state, idxs=[44, 45]),
    },
    "point_mass_maze": {
        'maxent_state': maxent_state,
        'maxent_state_action': maxent_state_action,
        # il evals temporarily disable - see scripts/collect_demos to generate demos
        # 'il_det_fw_state': partial(il_state, stoch=False, reverse=False),
        # 'il_stoch_fw_state': partial(il_state, stoch=True, reverse=False),
        # 'il_det_fw_state_action': partial(il_state_action, stoch=False, reverse=False),
        # 'il_stoch_fw_state_action': partial(il_state_action, stoch=True, reverse=False),
        'constrained': partial(constrained_state, idx=2),
        'robust': partial(robust_state, idxs=[2, 3]),  # 13, 14 for walker, 8, 9 for cheetah, 44, 45 for quadruped
    },
}


def get_tasks_by_domain(domain, gu=False):
    tasks = []
    if domain in ALL_LINEAR_TASKS:
        tasks += ALL_LINEAR_TASKS[domain]
    if gu and domain in ALL_GU_TASKS:
        tasks += [t for t in ALL_GU_TASKS[domain]]
    return tasks


def eval(domain, agent, replay_buffer, cfg, step, alpha=1.0, skip_linear=False, skip_gu=False, random=False):
    metrics = {}
    if not skip_linear:
        metrics.update(reward_eval(domain, agent, replay_buffer, cfg, step, alpha))
    if not skip_gu:
        metrics.update(gu_eval(domain, agent, replay_buffer, cfg, step, random=random))
    return metrics
    

def reward_inference(domain, task, batch, reward_inference_fn, device) -> torch.Tensor:
    env = suite.load(
        domain_name=domain,
        task_name=task,
        environment_kwargs={"flat_observation": True},
    )
    rewards = []
    for i in range(len(batch['action'])):
        with env._physics.reset_context():
            env._physics.set_state(batch["next"]["physics"][i].cpu().numpy())
            env._physics.set_control(batch["action"][i].cpu().detach().numpy())
        mujoco.mj_forward(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
        mujoco.mj_fwdPosition(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
        mujoco.mj_sensorVel(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
        mujoco.mj_subtreeVel(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
        rewards.append(env._task.get_reward(env._physics))
    rewards = np.array(rewards).reshape(-1, 1)
    z = reward_inference_fn(
        next_obs=batch["next"]["observation"],
        reward=torch.tensor(rewards, dtype=torch.float32, device=device),
    )
    return z


@torch.no_grad()
def reward_eval(domain, agent, replay_buffer, cfg, step, alpha=1.0):
    fn = reward_eval_bandit if domain == 'bandit' else reward_eval_mujoco
    return fn(domain, agent, replay_buffer, cfg, step, alpha)


def reward_eval_bandit(domain, agent, replay_buffer, cfg, step, alpha=1.0):
    metrics = {}
    obs = torch.zeros(1024, 2, device=agent.device)
    goal = torch.tensor([[0.0, 0.5]], device=agent.device)
    z = agent._model.goal_inference(goal) + torch.zeros((1024, 1), device=agent.device)
    actions = agent._model.actor(obs, z, agent.cfg.model.actor_std).sample()
    metrics['eval/linear/rl_reward'] = (torch.linalg.norm(actions - goal, dim=-1) < 0.1).float().mean().item()

    if agent._model.cfg.train_vector_field:
        raise NotImplementedError("Flow-based GHM not implemented.")
    
    print(metrics)

    return metrics


def reward_eval_mujoco(domain, agent, replay_buffer, cfg, step, alpha=1.0):
    metrics = {}
    batch = replay_buffer["train"].sample(cfg.num_inference_samples)

    for task in ALL_LINEAR_TASKS[domain]:
        z = reward_inference(domain, task, batch, agent._model.reward_inference, agent.device).reshape(1, -1)
        z = z * alpha
        eval_env = suite.load(
            domain_name=domain,
            task_name=task,
            environment_kwargs={"flat_observation": True},
        )
        total_reward = np.zeros((cfg.num_eval_episodes,), dtype=np.float64)
        total_entropy = np.zeros((cfg.num_eval_episodes,), dtype=np.float64)
        real_samples, actions, initial_obs = [], [], []
        for ep in range(cfg.num_eval_episodes):
            time_step = eval_env.reset()
            real_samples.append([])
            while not time_step.last():
                with eval_mode(agent._model):
                    obs = torch.tensor(
                        time_step.observation["observations"].reshape(1, -1),
                        device=agent.device,
                        dtype=torch.float32,
                    )
                    dist = agent._model.actor(obs, z, agent._model.cfg.actor_std)
                    action = dist.sample().cpu().numpy()
                    entropy = dist.entropy().sum().cpu().numpy().item()
                    if len(actions) == ep:
                        actions.append(dist.mean)
                    if len(initial_obs) == ep:
                        initial_obs.append(obs)
                    real_samples[-1].append(obs)
                time_step = eval_env.step(action)
                total_reward[ep] += time_step.reward
                total_entropy[ep] += entropy

        metrics.update({
            f"eval/linear/{task}_reward": np.mean(total_reward).item(),
            f"eval/linear/{task}_entropy": np.mean(total_entropy).item(),
        })

        if agent._model.cfg.train_vector_field:
            raise NotImplementedError("Flow-based GHM not implemented.")

        print({k: v for k,v in metrics.items() if task in k})

    return metrics


@torch.no_grad()
def gu_eval(domain, agent, replay_buffer, cfg, step, random=False):
    fn = gu_eval_bandit if domain == 'bandit' else gu_eval_mujoco
    return fn(domain, agent, replay_buffer, cfg, step, random=random)


def gu_eval_bandit(domain, agent, replay_buffer, cfg, step, random=False):

    metrics = {}
    NUM_CANDIDATES = 1024
    NUM_SAMPLES = 1024
    CORRECTION = 2.0

    # sample zs
    z = agent._model.sample_z(NUM_CANDIDATES, device=agent.device)
    scale = torch.rand((NUM_CANDIDATES, 1), device=agent.device)
    scaled_z = z * scale
    candidate_zs = scaled_z if agent.cfg.train.soft else z

    # predict occupancy
    if agent._model.cfg.train_vector_field:
        raise NotImplementedError("Flow-based GHM not implemented.")
    else:
        obs = torch.zeros((NUM_CANDIDATES, 2), device=agent.device)
        next_obs = replay_buffer["train"].sample(NUM_SAMPLES)["next"]["observation"].to(agent.device)
        with eval_mode(agent._model):
            dist = agent._model.actor(obs, candidate_zs, agent.cfg.model.actor_std)
            Fs = agent._model._forward_map(obs, candidate_zs, dist.sample())  # num_parallel x batch x z_dim
            B = agent._model._backward_map(next_obs)  # batch x z_dim
        M = torch.matmul(Fs, B.T).mean(0)  # batch x batch
        M = M.clip(1.e-6)
        M = M ** CORRECTION
        M = M / M.sum(-1, keepdims=True)
        idxs = torch.vmap(partial(torch.multinomial, num_samples=NUM_SAMPLES, replacement=True), randomness='different')(M)
        pred_samples = torch.vmap(lambda x: next_obs[x])(idxs)

    for task in ALL_GU_TASKS[domain]:

        # get obj_fn
        obj_fn = ALL_GU_TASKS[domain][task]
        pred_scores = torch.vmap(obj_fn)(pred_samples, torch.zeros_like(pred_samples), torch.zeros_like(pred_samples))

        # evaluate occupancy
        _obs = obs.repeat(NUM_SAMPLES, 1)
        _z = candidate_zs.repeat_interleave(NUM_SAMPLES, 0)
        with eval_mode(agent._model):
            real_samples = agent._model.actor(_obs, _z, agent.cfg.model.actor_std).sample().reshape(len(candidate_zs), NUM_SAMPLES, -1)
        real_scores = torch.vmap(obj_fn)(real_samples, torch.zeros_like(real_samples), torch.zeros_like(real_samples))
        metrics[f'eval/gu/{task}_reward'] = real_scores[pred_scores.argmax().item()].item()
        metrics[f'eval/gu/{task}_avg'] = real_scores.mean().item()
        metrics[f'eval/gu/{task}_corr'] = np.corrcoef(pred_scores.cpu().numpy(), real_scores.cpu().numpy())[0, 1].item()

        # log
        corr_data = {k: v.cpu().numpy() for k, v in zip(['z', 'pred_scores', 'real_scores'], [candidate_zs, pred_scores, real_scores])}
        if cfg.working_dir is not None:

            with open(cfg.working_dir+f'/corr_{task}_{step}.npy', 'wb') as f:
                np.save(f, corr_data)

            if task == 'rl':
                # draw action spread
                with torch.no_grad():
                    z = agent._model._backward_map(torch.tensor([[0.0, 0.5]], device=agent.device)) + torch.zeros((1024, 1), device=agent.device)
                    data = {}
                    arange = torch.arange(21, device=agent.device) / 10 - 1
                    arange = arange.repeat(21).reshape(21, 21)
                    next_obs = torch.stack([arange.reshape(-1), arange.T.reshape(-1)], -1)
                    for i, coeff in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
                        with eval_mode(agent._model):
                            samples = []
                            for _ in range(10):
                                samples.append(agent._model.actor(torch.zeros(1024, 2, device=agent.device), z*coeff, agent.cfg.model.actor_std).sample())
                            samples = torch.cat(samples, 0)
                            Fs = agent._model._forward_map(obs[:len(next_obs)], z[:len(next_obs)]*coeff, next_obs).mean(0)  # num_parallel x batch x z_dim
                        Q = (Fs * z[:len(next_obs)] * coeff).sum(-1)
                        data[i] ={'actions': next_obs.cpu().numpy(), 'values': Q.cpu().numpy(), 'samples': samples.cpu().numpy()}
                    with open(cfg.working_dir+f'/samples_{step}.npy', 'wb') as f:
                        np.save(f, data)

        print({k: v for k,v in metrics.items() if task in k})

    return metrics

def gu_eval_mujoco(domain, agent, replay_buffer, cfg, step, random=False):

    NUM_CANDIDATES = 1024
    NUM_SAMPLES = 256
    NUM_ACTIONS = 8
    CORRECTION = 2.0
    metrics = {}
    eval_env = suite.load(
        domain_name=domain,
        task_name=ALL_LINEAR_TASKS[domain][0],
        environment_kwargs={"flat_observation": True},
    )

    candidate_zs = agent._model.sample_z(NUM_CANDIDATES, device=agent.device)
    # assume that solution is not deterministic
    if agent.cfg.train.soft:
        scale = torch.rand((NUM_CANDIDATES, 1), device=agent.device)
        candidate_zs = candidate_zs * scale

    # initialize environment
    time_step = eval_env.reset()
    obs = torch.tensor(
        time_step.observation["observations"].reshape(1, -1),
        device=agent.device,
        dtype=torch.float32,
    )

    # estimate occupancy
    if agent._model.cfg.train_vector_field:
        raise NotImplementedError("Flow-based GHM not implemented.")
    else:
        obs = obs.repeat(NUM_CANDIDATES, 1)
        next_obs = replay_buffer["train"].sample(NUM_SAMPLES)["next"]["observation"].to(agent.device)
        with eval_mode(agent._model):
            dist = agent._model.actor(obs, candidate_zs, agent.cfg.model.actor_std)
            Fs = agent._model._forward_map(obs, candidate_zs, dist.sample())  # num_parallel x batch x z_dim
            B = agent._model._backward_map(next_obs)  # batch x z_dim
        M = torch.matmul(Fs, B.T).mean(0)  # batch x batch
        M = M.clip(1.e-6)
        M = M ** CORRECTION
        M = M / M.sum(-1, keepdims=True)
        idxs = torch.vmap(partial(torch.multinomial, num_samples=NUM_SAMPLES, replacement=True), randomness='different')(M)
        pred_samples = torch.vmap(lambda x: next_obs[x])(idxs)

    # estimate action occupancy
    pred_actions = []
    for _z, _samples in zip(candidate_zs, pred_samples):
        _z = _z[None].repeat(_samples.shape[0], 1)
        with eval_mode(agent._model):
            dist = agent._model.actor(_samples, _z, agent.cfg.model.actor_std)
        pred_actions.append(torch.stack([dist.sample() for _ in range(NUM_ACTIONS)], 1))
    pred_actions = torch.stack(pred_actions, 0)

    # normalize state samples
    dummy_obs = replay_buffer["train"].sample(1024)["observation"].to(agent.device)
    obs_min = dummy_obs.min(0, keepdim=True)[0]
    obs_max = dummy_obs.max(0, keepdim=True)[0]
    norm_pred_samples = (pred_samples - obs_min) / (obs_max - obs_min) * 2 - 1

    # load and normalize demos
    demos = {}
    random_reset = False
    try:
        for type in ['stoch', 'det']:
            demo_path = os.path.join(cfg.dataset_root, domain)
            demos[type] = np.load(os.path.join(demo_path, f'demos_{type}.npy'), allow_pickle=True).tolist()
            idxs = torch.distributions.Geometric(probs=torch.ones(NUM_SAMPLES) * (1-agent.cfg.train.discount)).sample().int()
            idxs = idxs.clip(0, demos[type]['observations'].shape[1]-1)
            demos[type]['observations'] = demos[type]['observations'][:, idxs].reshape(-1, demos[type]['observations'].shape[-1])
            demos[type]['actions'] = demos[type]['actions'][:, idxs].reshape(-1, *demos[type]['actions'].shape[-2:])
            demos[type]['observations'] = torch.tensor(demos[type]['observations'], device=agent.device, dtype=torch.float32)
            demos[type]['actions'] = torch.tensor(demos[type]['actions'], device=agent.device, dtype=torch.float32)
            demos[type]['observations'] = (demos[type]['observations'] - obs_min) / (obs_max - obs_min) * 2 - 1
    except Exception as e:
        print(f"Error loading demos for GU evaluation: {e}. GU evaluation will start from random states.")
        random_reset = True

    for task in ALL_GU_TASKS[domain]:

        # get obj_fn
        obj_fn = ALL_GU_TASKS[domain][task]
        pred_scores = torch.stack([obj_fn(_nps, _pa, demos) for _nps, _pa in zip(norm_pred_samples, pred_actions)])
        if random:
            best_z = candidate_zs[[0]]
        else:
            best_z = candidate_zs[[pred_scores.argmax().item()]]

        real_samples = []
        for _ in range(cfg.num_eval_episodes):
            eval_env.reset()
            if not random_reset:
                with eval_env.physics.reset_context():
                    eval_env.physics.set_state(demos['det']['og_physics_state'])
            og_obs = eval_env._task.get_observation(eval_env.physics)
            og_obs = flatten_observation(og_obs) if eval_env._flat_observation else og_obs
            time_step = dm_env.TimeStep(
                step_type=dm_env.StepType.FIRST,
                reward=None,
                discount=None,
                observation=og_obs
            )
            real_samples.append([])
            while not time_step.last():
                with torch.no_grad(), eval_mode(agent._model):
                    obs = torch.tensor(
                        time_step.observation["observations"].reshape(1, -1),
                        device=agent.device,
                        dtype=torch.float32,
                    )
                    action = agent.act(obs=obs, z=best_z, mean=False).cpu().numpy()

                time_step = eval_env.step(action)
                real_samples[-1].append(obs)

        real_samples = torch.stack([torch.cat(rs, 0) for rs in real_samples], 0)
        idxs = torch.distributions.Geometric(probs=torch.ones(NUM_SAMPLES) * (1-agent.cfg.train.discount)).sample().int()
        idxs = idxs.clip(0, real_samples.shape[1]-1)
        real_samples = real_samples[:, idxs].reshape(-1, real_samples.shape[-1])
        real_actions = agent._model.actor(real_samples, best_z.repeat(real_samples.shape[0], 1), agent.cfg.model.actor_std)
        real_actions = torch.stack([real_actions.sample() for _ in range(NUM_ACTIONS)], 1)
        norm_real_samples = (real_samples - obs_min) / (obs_max - obs_min) * 2 - 1
        real_score = obj_fn(norm_real_samples, real_actions, demos)
        metrics[f"eval/gu/{task}_reward"] = real_score.item()

        print({k: v for k,v in metrics.items() if task in k})

    return metrics
