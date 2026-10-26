# Adapted from https://github.com/facebookresearch/metamotivo

from __future__ import annotations
import torch

torch.set_float32_matmul_precision("high")

import numpy as np
import dataclasses
from src import FBAgent, FBAgentConfig, DictBuffer, eval_mode
from src.eval import eval, get_tasks_by_domain
from src.utils import update_with_overrides, dump
from tqdm import tqdm
import time
import custom_dmc as suite
import random
from pathlib import Path
import wandb
import sys
import yaml
from typing import List
import tyro


def create_agent(config) -> FBAgent:
    if config.domain_name not in ["walker", "cheetah", "quadruped", "point_mass_maze", "hopper", "bandit"]:
        raise RuntimeError('FB configuration defined only for "walker", "cheetah", "quadruped", "point_mass_maze", "hopper", "bandit')
    env = suite.load(domain_name=config.domain_name, task_name=config.task_name, environment_kwargs={"flat_observation": True})
    agent_config = FBAgentConfig()
    agent_config.model.obs_dim = env.observation_spec()["observations"].shape[0]
    agent_config.model.action_dim = env.action_spec().shape[0]
    agent_config.model.device = config.device
    if config.domain_name in ["cheetah", "quadruped", "bandit"]:
        agent_config.model.archi.z_dim = 50
    if config.domain_name == "point_mass_maze":
        agent_config.train.lr_b = 1e-6
        agent_config.train.lr_actor = 1e-6
        agent_config.train.discount = 0.99
    if config.domain_name == 'bandit':
        agent_config.train.discount = 0.5
    agent_config.compile = config.compile
    agent_config.cudagraphs = config.cudagraphs
    return agent_config


def load_data(dataset_path, expl_agent, domain_name, num_episodes=1):
    path = Path(dataset_path) / f"{domain_name}/{expl_agent}/buffer"
    print(f"Data path: {path}")
    storage = {
        "observation": [],
        "action": [],
        "physics": [],
        "next": {"observation": [], "terminated": [], "physics": []},
    }
    files = list(path.glob("*.npz"))
    num_episodes = min(num_episodes, len(files))
    for i in tqdm(range(num_episodes)):
        data = np.load(str(files[i]))
        storage["observation"].append(data["observation"][:-1].astype(np.float32))
        storage["action"].append(data["action"][1:].astype(np.float32))
        storage["next"]["observation"].append(data["observation"][1:].astype(np.float32))
        storage["next"]["terminated"].append(np.array(1 - data["discount"][1:], dtype=np.bool))
        storage["physics"].append(data["physics"][:-1])
        storage["next"]["physics"].append(data["physics"][1:])

    for k in storage:
        if k == "next":
            for k1 in storage[k]:
                storage[k][k1] = np.concat(storage[k][k1])
        else:
            storage[k] = np.concat(storage[k])
    return storage


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@dataclasses.dataclass
class TrainConfig:
    dataset_root: str = Path.cwd() / "datasets"
    seed: int = 0
    domain_name: str = "quadruped"
    task_name: str | None = None
    dataset_expl_agent: str = "rnd"
    num_train_steps: int = 3_000_000
    load_n_episodes: int = 5_000
    log_every_updates: int = 10_000
    working_dir: str | None = None
    checkpoint_every_steps: int = 1_000_000

    # eval
    num_eval_episodes: int = 10
    num_inference_samples: int = 50_000
    eval_every_steps: int = 100_000
    gu_eval: bool = False

    # misc
    compile: bool = False
    cudagraphs: bool = False
    device: str = "cpu"

    # WANDB
    use_wandb: bool = False
    wandb_ename: str | None = None
    wandb_gname: str | None = None
    wandb_pname: str | None = "fb_train_dmc"
    wandb_name_prefix: str | None = None


class Workspace:
    def __init__(self, cfg: TrainConfig, agent_cfg: FBAgentConfig) -> None:
        self.cfg = cfg
        self.agent_cfg = agent_cfg
        if self.cfg.working_dir is None:
            import string
            tmp_name = "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
            self.working_dir = Path.cwd() / "results" / tmp_name
            self.cfg.working_dir = str(self.working_dir)
            self.working_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.working_dir = Path(self.cfg.working_dir)
        print(f"working dir: {self.working_dir}")

        set_seed_everywhere(self.cfg.seed)
        self.agent = FBAgent(**dataclasses.asdict(self.agent_cfg))

        wandb_config = {**dataclasses.asdict(self.cfg), 'agent': dataclasses.asdict(self.agent_cfg)}
        if self.cfg.use_wandb:
            exp_name = "fb"
            wandb_name = exp_name
            if self.cfg.wandb_name_prefix:
                wandb_name = f"{self.cfg.wandb_name_prefix}_{exp_name}"
            # fmt: off
            wandb.init(entity=self.cfg.wandb_ename, project=self.cfg.wandb_pname,
                group=self.cfg.wandb_gname, name=wandb_name,  # mode="disabled",
                config=wandb_config)  # type: ignore
            # fmt: on

        with (self.working_dir / "wandb_config.yaml").open("w") as f:
            yaml.dump(wandb_config, f)

    def train(self):
        self.start_time = time.time()
        self.train_offline()

    def train_offline(self) -> None:
        self.replay_buffer = {}
        # LOAD DATA FROM EXORL
        data = load_data(
            self.cfg.dataset_root,
            self.cfg.dataset_expl_agent,
            self.cfg.domain_name,
            self.cfg.load_n_episodes,
        )
        self.replay_buffer = {"train": DictBuffer(capacity=data["observation"].shape[0], device=self.agent.device)}
        self.replay_buffer["train"].extend(data)
        print(self.replay_buffer["train"])
        del data

        total_metrics = None
        fps_start_time = time.time()
        for t in tqdm(range(0, int(self.cfg.num_train_steps))):

            if t % self.cfg.eval_every_steps == 0:
                self.eval(t)

            metrics = self.agent.update(self.replay_buffer, t)

            # we need to copy tensors returned by a cudagraph module
            if total_metrics is None:
                total_metrics = {k: metrics[k].clone() for k in metrics.keys()}
            else:
                total_metrics = {k: total_metrics[k] + metrics[k] for k in metrics.keys()}

            if t % self.cfg.log_every_updates == 0:
                m_dict = {}
                for k in sorted(list(total_metrics.keys())):
                    tmp = total_metrics[k] / (1 if t == 0 else self.cfg.log_every_updates)
                    m_dict[k] = np.round(tmp.mean().item(), 6)
                m_dict["duration"] = time.time() - self.start_time
                m_dict["FPS"] = (1 if t == 0 else self.cfg.log_every_updates) / (time.time() - fps_start_time)
                if self.cfg.use_wandb:
                    wandb.log(
                        {f"train/{k}": v for k, v in m_dict.items()},
                        step=t,
                    )
                print(m_dict)
                total_metrics = None
                fps_start_time = time.time()
            if t % self.cfg.checkpoint_every_steps == 0:
                self.agent.save(str(self.working_dir / "checkpoint"))

        self.agent.save(str(self.working_dir / "checkpoint"))
        return

    def eval(self, t):
        m_dict = eval(self.cfg.domain_name, self.agent, self.replay_buffer, self.cfg, t)
        m_dict[f"eval/overall_reward"] = np.mean([v for k, v in m_dict.items() if k.endswith("reward")]).item()
        m_dict[f"eval/overall_corr"] = np.mean([v for k, v in m_dict.items() if k.startswith("corr")]).item()
        m_dict[f"eval/overall_ot_distance"] = np.mean([v for k, v in m_dict.items() if k.startswith("ot_distance")]).item()
        if self.cfg.use_wandb:
            wandb.log({k: v for k, v in m_dict.items()}, step=t)
        print(m_dict)
        dump(m_dict, t, (self.working_dir / "metrics.yaml"))


if __name__ == "__main__":

    overrides = {}
    if sys.argv[-1].endswith('.yaml'):
        with open(sys.argv.pop(-1), 'r') as file:
            overrides = yaml.safe_load(file)
    agent_overrides = overrides.pop('agent') if 'agent' in overrides else {}
    config = tyro.cli(TrainConfig, default=TrainConfig(**overrides))
    agent_config = create_agent(config)
    update_with_overrides(agent_config, agent_overrides)

    ws = Workspace(config, agent_cfg=agent_config)
    ws.train()
