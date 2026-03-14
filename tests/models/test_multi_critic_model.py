from __future__ import annotations

import torch
import torch.nn.functional as F

from rsl_rl.models import MultiCriticModel
from tests.conftest import make_obs

NUM_ENVS = 4
OBS_DIM = 8
OBS_GROUPS = {"actor": ["policy"], "critic": ["policy"]}


def test_multi_critic_model_returns_k_values() -> None:
    obs = make_obs(NUM_ENVS, OBS_DIM)
    critic = MultiCriticModel(
        obs,
        OBS_GROUPS,
        "critic",
        num_critics=3,
        trunk_hidden_dims=(32, 16),
        head_hidden_dims=(8,),
    )

    values = critic(obs)

    assert values.shape == (NUM_ENVS, 3)


def test_multi_critic_model_matches_reference_per_head_mlp() -> None:
    obs = make_obs(NUM_ENVS, OBS_DIM)
    critic = MultiCriticModel(
        obs,
        OBS_GROUPS,
        "critic",
        num_critics=3,
        trunk_hidden_dims=(32, 16),
        head_hidden_dims=(8, 4),
    )

    values = critic(obs)
    trunk_output = critic.trunk(critic.get_latent(obs))
    reference_outputs = []
    for critic_idx in range(critic.num_critics):
        head_output = trunk_output
        for layer_idx, (weight, bias) in enumerate(
            zip(critic.batched_heads.weights, critic.batched_heads.biases, strict=False)
        ):
            head_output = F.linear(head_output, weight[critic_idx], bias[critic_idx])
            if layer_idx < len(critic.batched_heads.weights) - 1:
                head_output = critic.batched_heads.activation(head_output)
        reference_outputs.append(head_output)
    reference = torch.cat(reference_outputs, dim=-1)

    torch.testing.assert_close(values, reference)


def test_multi_critic_model_supports_single_layer_trunk() -> None:
    obs = make_obs(NUM_ENVS, OBS_DIM)
    critic = MultiCriticModel(
        obs,
        OBS_GROUPS,
        "critic",
        num_critics=2,
        trunk_hidden_dims=(32,),
        head_hidden_dims=(8,),
    )

    values = critic(obs)

    assert values.shape == (NUM_ENVS, 2)


def test_multi_critic_model_updates_normalization() -> None:
    obs = make_obs(NUM_ENVS, OBS_DIM)
    critic = MultiCriticModel(
        obs,
        OBS_GROUPS,
        "critic",
        num_critics=2,
        trunk_hidden_dims=(32, 16),
        head_hidden_dims=(8,),
        obs_normalization=True,
    )
    critic.train()

    out_before = critic(obs).detach().clone()
    shifted_obs = make_obs(NUM_ENVS, OBS_DIM)
    shifted_obs["policy"] = shifted_obs["policy"] + 100.0
    for _ in range(50):
        critic.update_normalization(shifted_obs)

    out_after = critic(obs).detach()

    assert not torch.allclose(out_before, out_after, atol=1e-3)
