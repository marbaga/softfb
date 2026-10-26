# Adapted from https://github.com/facebookresearch/metamotivo

import math
import dataclasses
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import copy
from pathlib import Path
from safetensors.torch import save_model as safetensors_save_model
import json

from .nn_models import build_backward, build_forward, build_actor, build_critic, eval_mode
from .utils import config_from_dict, load_model


@dataclasses.dataclass
class ActorArchiConfig:
    hidden_dim: int = 1024
    model: str = "simple"  # {'simple', 'residual'}
    hidden_layers: int = 1
    embedding_layers: int = 2
    learn_std: bool = False
    per_state_std: bool = True
    squash: bool = True


@dataclasses.dataclass
class ForwardArchiConfig:
    hidden_dim: int = 1024
    model: str = "simple"  # {'simple', 'residual'}
    hidden_layers: int = 1
    embedding_layers: int = 2
    num_parallel: int = 2
    ensemble_mode: str = "batch"  # {'batch', 'seq', 'vmap'}


@dataclasses.dataclass
class BackwardArchiConfig:
    hidden_dim: int = 256
    hidden_layers: int = 2
    norm: bool = True


@dataclasses.dataclass
class CriticArchiConfig:
    hidden_dim: int = 1024
    hidden_layers: int = 1
    embedding_layers: int = 2
    num_parallel: int = 2
    ensemble_mode: str = "batch"  # {'batch', 'seq', 'vmap'}


@dataclasses.dataclass
class ArchiConfig:
    z_dim: int = 100
    norm_z: bool = True
    f: ForwardArchiConfig = dataclasses.field(default_factory=ForwardArchiConfig)
    b: BackwardArchiConfig = dataclasses.field(default_factory=BackwardArchiConfig)
    actor: ActorArchiConfig = dataclasses.field(default_factory=ActorArchiConfig)
    critic: CriticArchiConfig = dataclasses.field(default_factory=CriticArchiConfig)


@dataclasses.dataclass
class Config:
    obs_dim: int = -1
    action_dim: int = -1
    device: str = "cpu"
    archi: ArchiConfig = dataclasses.field(default_factory=ArchiConfig)
    inference_batch_size: int = 500_000
    seq_length: int = 1
    actor_std: float = 0.2
    norm_obs: bool = False
    train_vector_field: bool = False

class FBModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.cfg = config_from_dict(kwargs, Config)
        obs_dim, action_dim = self.cfg.obs_dim, self.cfg.action_dim
        arch = self.cfg.archi

        # create networks
        self._backward_map = build_backward(obs_dim, arch.z_dim, arch.b)
        self._forward_map = build_forward(obs_dim, arch.z_dim, action_dim, arch.f)
        self._actor = build_actor(obs_dim, arch.z_dim, action_dim, arch.actor)
        self._critic = build_critic(obs_dim, arch.z_dim, action_dim, arch.critic)
        if self.cfg.train_vector_field:
            raise NotImplementedError("Flow-based GHM not implemented.")
        self._obs_normalizer = nn.BatchNorm1d(obs_dim, affine=False, momentum=0.01) if self.cfg.norm_obs else nn.Identity()
        self.register_buffer('_obs_min', 1e6 * torch.ones(obs_dim, device=self.cfg.device).unsqueeze(0))
        self.register_buffer('_obs_max', -1e6 * torch.ones(obs_dim, device=self.cfg.device).unsqueeze(0))

        # make sure the model is in eval mode and never computes gradients
        self.train(False)
        self.requires_grad_(False)
        self.to(self.cfg.device)

    def _prepare_for_train(self) -> None:
        # create TARGET networks
        self._target_backward_map = copy.deepcopy(self._backward_map)
        self._target_forward_map = copy.deepcopy(self._forward_map)
        self._target_critic = copy.deepcopy(self._critic)

    def to(self, *args, **kwargs):
        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.cfg.device = device.type  # type: ignore
        return super().to(*args, **kwargs)

    @classmethod
    def load(cls, path: str, device: str | None = None):
        return load_model(path, device, cls=cls)

    def save(self, output_folder: str) -> None:
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True)
        safetensors_save_model(self, output_folder / "model.safetensors")
        with (output_folder / "config.json").open("w+") as f:
            json.dump(dataclasses.asdict(self.cfg), f, indent=4)

    def _normalize(self, obs: torch.Tensor):
        with torch.no_grad(), eval_mode(self._obs_normalizer):
            return self._obs_normalizer(obs)

    @torch.no_grad()
    def backward_map(self, obs: torch.Tensor):
        return self._backward_map(self._normalize(obs))

    @torch.no_grad()
    def forward_map(self, obs: torch.Tensor, z: torch.Tensor, action: torch.Tensor):
        return self._forward_map(self._normalize(obs), z, action)

    @torch.no_grad()
    def actor(self, obs: torch.Tensor, z: torch.Tensor, std: float):
        return self._actor(self._normalize(obs), z, std)

    @torch.no_grad()
    def critic(self, obs: torch.Tensor, z: torch.Tensor, action: torch.Tensor):
        return self._critic(self._normalize(obs), z, action)

    def sample_z(self, size: int, device: str = "cpu") -> torch.Tensor:
        z = torch.randn((size, self.cfg.archi.z_dim), dtype=torch.float32, device=device)
        return self.project_z(z)

    def project_z(self, z):
        if self.cfg.archi.norm_z:
            z = math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)
        return z

    def act(self, obs: torch.Tensor, z: torch.Tensor, mean: bool = True) -> torch.Tensor:
        dist = self.actor(obs, z, self.cfg.actor_std)
        if mean:
            return dist.mean
        return dist.sample()

    @torch.no_grad()
    def generate(self, obs: torch.Tensor, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if not self.cfg.train_vector_field:
            raise NotImplementedError("Flow-based generation not implemented when vector field training is disabled.")

        obs = 2 * (obs - self._obs_min) / (self._obs_max - self._obs_min) - 1
        current_noises = torch.randn(obs.shape, device=obs.device, dtype=obs.dtype)
        samples = self._target_vector_field.generate(
                current_noises, obs, action, z,
                observation_min=-1, observation_max=1)
        
        samples = ((samples + 1) / 2) * (self._obs_max - self._obs_min) + self._obs_min
        return samples

    def reward_inference(self, next_obs: torch.Tensor, reward: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
        num_batches = int(np.ceil(next_obs.shape[0] / self.cfg.inference_batch_size))
        z = 0
        wr = reward if weight is None else reward * weight
        for i in range(num_batches):
            start_idx, end_idx = i * self.cfg.inference_batch_size, (i + 1) * self.cfg.inference_batch_size
            B = self.backward_map(next_obs[start_idx:end_idx].to(self.cfg.device))
            z += torch.matmul(wr[start_idx:end_idx].to(self.cfg.device).T, B)
        return self.project_z(z)

    def reward_wr_inference(self, next_obs: torch.Tensor, reward: torch.Tensor) -> torch.Tensor:
        return self.reward_inference(next_obs, reward, F.softmax(10 * reward, dim=0))

    def goal_inference(self, next_obs: torch.Tensor) -> torch.Tensor:
        z = self.backward_map(next_obs)
        return self.project_z(z)

    def tracking_inference(self, next_obs: torch.Tensor) -> torch.Tensor:
        z = self.backward_map(next_obs)
        for step in range(z.shape[0]):
            end_idx = min(step + self.cfg.seq_length, z.shape[0])
            z[step] = z[step:end_idx].mean(dim=0)
        return self.project_z(z)
