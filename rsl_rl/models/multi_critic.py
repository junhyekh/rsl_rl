from __future__ import annotations

import math

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import EmpiricalNormalization, HiddenState, MLP
from rsl_rl.utils import resolve_nn_activation, unpad_trajectories


class _BatchedMLPHeads(nn.Module):
    """Apply the same MLP head structure to K critics in parallel."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        num_heads: int,
        activation: str,
    ) -> None:
        super().__init__()
        self.activation = resolve_nn_activation(activation)
        layer_dims = (input_dim, *hidden_dims, 1)
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:], strict=False):
            weight = nn.Parameter(torch.empty(num_heads, out_dim, in_dim))
            bias = nn.Parameter(torch.empty(num_heads, out_dim))
            self.weights.append(weight)
            self.biases.append(bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight, bias in zip(self.weights, self.biases, strict=False):
            for head_idx in range(weight.shape[0]):
                nn.init.kaiming_uniform_(weight[head_idx], a=math.sqrt(5))
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight[head_idx])
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(bias[head_idx], -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1).expand(-1, self.weights[0].shape[0], -1)
        for layer_idx, (weight, bias) in enumerate(zip(self.weights, self.biases, strict=False)):
            x = torch.einsum("bki,koi->bko", x, weight) + bias.unsqueeze(0)
            if layer_idx < len(self.weights) - 1:
                x = self.activation(x)
        return x.squeeze(-1)


class MultiCriticModel(nn.Module):
    """Shared-trunk critic with multiple scalar value heads."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        num_critics: int,
        trunk_hidden_dims: tuple[int, ...] | list[int] = (256, 256),
        head_hidden_dims: tuple[int, ...] | list[int] = (128, 64),
        activation: str = "elu",
        obs_normalization: bool = False,
    ) -> None:
        super().__init__()

        if num_critics < 2:
            raise ValueError(f"MultiCriticModel requires num_critics >= 2, got {num_critics}.")
        if len(trunk_hidden_dims) < 1:
            raise ValueError("trunk_hidden_dims must contain at least one hidden dimension.")
        if len(head_hidden_dims) < 1:
            raise ValueError("head_hidden_dims must contain at least one hidden dimension.")

        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        self.obs_normalization = obs_normalization
        self.num_critics = num_critics

        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
        else:
            self.obs_normalizer = nn.Identity()

        trunk_hidden_dims = tuple(trunk_hidden_dims)
        head_hidden_dims = tuple(head_hidden_dims)
        trunk_out_dim = trunk_hidden_dims[-1]
        if len(trunk_hidden_dims) == 1:
            self.trunk = nn.Sequential(
                nn.Linear(self.obs_dim, trunk_out_dim),
                resolve_nn_activation(activation),
            )
        else:
            self.trunk = MLP(
                self.obs_dim,
                trunk_out_dim,
                trunk_hidden_dims[:-1],
                activation,
                last_activation=activation,
            )
        self.batched_heads = _BatchedMLPHeads(trunk_out_dim, head_hidden_dims, num_critics, activation)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        del hidden_state, stochastic_output
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs)
        trunk_output = self.trunk(latent)
        return self.batched_heads(trunk_output)

    def get_latent(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups]
        latent = torch.cat(obs_list, dim=-1)
        return self.obs_normalizer(latent)

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        del dones, hidden_state

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        del dones

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            obs_list = [obs[obs_group] for obs_group in self.obs_groups]
            critic_obs = torch.cat(obs_list, dim=-1)
            self.obs_normalizer.update(critic_obs)  # type: ignore[arg-type]

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        active_obs_groups = obs_groups[obs_set]
        obs_dim = 0
        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    f"The multi-critic model only supports 1D observations, got shape {obs[obs_group].shape} "
                    f"for '{obs_group}'."
                )
            obs_dim += obs[obs_group].shape[-1]
        return active_obs_groups, obs_dim
