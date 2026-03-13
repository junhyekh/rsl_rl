from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.models import MLPModel, MultiCriticModel
from rsl_rl.storage import RolloutStorage
from tests.conftest import make_obs

NUM_ENVS = 2
NUM_STEPS = 1
OBS_DIM = 8
NUM_ACTIONS = 4
OBS_GROUPS = {"actor": ["policy"], "critic": ["policy"]}


def _make_actor(obs: TensorDict) -> MLPModel:
    return MLPModel(
        obs,
        OBS_GROUPS,
        "actor",
        NUM_ACTIONS,
        hidden_dims=[32, 32],
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    )


def _make_multi_critic(obs: TensorDict) -> MultiCriticModel:
    return MultiCriticModel(
        obs,
        OBS_GROUPS,
        "critic",
        num_critics=2,
        trunk_hidden_dims=(32, 16),
        head_hidden_dims=(8,),
    )


def _make_construct_algorithm_cfg(critic_class_name: object) -> dict:
    return {
        "num_steps_per_env": NUM_STEPS,
        "multi_gpu": None,
        "obs_groups": {"actor": ("policy",), "critic": ("policy",)},
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": (32, 32),
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        },
        "critic": {"class_name": critic_class_name, "hidden_dims": (32, 32)},
        "algorithm": {"class_name": "PPO", "schedule": "fixed"},
        "multi_critic": {
            "enabled": True,
            "num_critics": 2,
            "reward_group_indices": [[0], [1]],
            "trunk_hidden_dims": (32, 16),
            "head_hidden_dims": (8,),
            "advantage_normalization": "independent",
            "critic_weights": [1.0, 1.0],
        },
    }


def _make_multi_critic_ppo(**overrides: object) -> tuple[PPO, TensorDict]:
    obs = make_obs(NUM_ENVS, OBS_DIM)
    actor = _make_actor(obs)
    critic = _make_multi_critic(obs)
    storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS], num_critics=2)
    defaults = dict(
        num_learning_epochs=2,
        num_mini_batches=1,
        gamma=0.99,
        lam=0.95,
        schedule="fixed",
        num_critics=2,
        reward_group_indices=[[0, 1], [2, 3]],
        critic_weights=[2.0, 0.5],
    )
    defaults.update(overrides)
    return PPO(actor, critic, storage, **defaults), obs


def test_multi_critic_independent_normalization() -> None:
    ppo, obs = _make_multi_critic_ppo(advantage_normalization="independent")
    rewards = torch.tensor([[1.0, 2.0], [3.0, 6.0]])

    transition = RolloutStorage.Transition()
    transition.observations = obs
    transition.hidden_states = (None, None)
    transition.actions = torch.zeros(NUM_ENVS, NUM_ACTIONS)
    transition.values = torch.zeros(NUM_ENVS, 2)
    transition.actions_log_prob = torch.zeros(NUM_ENVS)
    transition.distribution_params = (torch.zeros(NUM_ENVS, NUM_ACTIONS), torch.ones(NUM_ENVS, NUM_ACTIONS))
    transition.rewards = rewards
    transition.dones = torch.zeros(NUM_ENVS)
    ppo.storage.add_transition(transition)

    original_forward = ppo.critic.forward
    ppo.critic.forward = lambda *args, **kwargs: torch.zeros(NUM_ENVS, 2)
    ppo.compute_returns(obs)
    ppo.critic.forward = original_forward

    expected_advantages = torch.zeros_like(rewards)
    for critic_idx, weight in enumerate((2.0, 0.5)):
        critic_rewards = rewards[:, critic_idx]
        expected_advantages[:, critic_idx] = weight * (
            (critic_rewards - critic_rewards.mean()) / (critic_rewards.std() + 1e-8)
        )

    torch.testing.assert_close(ppo.storage.returns[0], rewards)
    torch.testing.assert_close(ppo.storage.advantages[0], expected_advantages)
    torch.testing.assert_close(
        ppo.storage.aggregated_advantages[0],
        expected_advantages.sum(dim=-1, keepdim=True),
    )


def test_multi_critic_magnitude_preserved_normalization() -> None:
    ppo, obs = _make_multi_critic_ppo(advantage_normalization="magnitude_preserved")
    rewards = torch.tensor([[1.0, 2.0], [3.0, 6.0]])

    transition = RolloutStorage.Transition()
    transition.observations = obs
    transition.hidden_states = (None, None)
    transition.actions = torch.zeros(NUM_ENVS, NUM_ACTIONS)
    transition.values = torch.zeros(NUM_ENVS, 2)
    transition.actions_log_prob = torch.zeros(NUM_ENVS)
    transition.distribution_params = (torch.zeros(NUM_ENVS, NUM_ACTIONS), torch.ones(NUM_ENVS, NUM_ACTIONS))
    transition.rewards = rewards
    transition.dones = torch.zeros(NUM_ENVS)
    ppo.storage.add_transition(transition)

    original_forward = ppo.critic.forward
    ppo.critic.forward = lambda *args, **kwargs: torch.zeros(NUM_ENVS, 2)
    ppo.compute_returns(obs)
    ppo.critic.forward = original_forward

    centered = rewards - rewards.mean(dim=0, keepdim=True)
    pooled_std = torch.sqrt(centered.var(dim=0).sum() + 1e-8)
    expected = (centered / pooled_std).sum(dim=-1, keepdim=True)

    torch.testing.assert_close(ppo.storage.aggregated_advantages[0], expected)


def test_multi_critic_process_env_step_maps_rewards() -> None:
    ppo, obs = _make_multi_critic_ppo()

    ppo.act(obs)
    stored_values = ppo.transition.values.clone()
    extras = {
        "per_term_rewards": torch.tensor([[1.0, 2.0, 10.0, 20.0], [3.0, 4.0, 30.0, 40.0]]),
        "time_outs": torch.tensor([1.0, 0.0]),
    }
    ppo.process_env_step(
        obs,
        rewards=torch.zeros(NUM_ENVS),
        dones=torch.zeros(NUM_ENVS),
        extras=extras,
    )

    expected = torch.tensor([[3.0, 30.0], [7.0, 70.0]])
    expected[0] += ppo.gamma * stored_values[0]

    torch.testing.assert_close(ppo.storage.rewards[0], expected)


def test_multi_critic_load_rejects_single_critic_checkpoint() -> None:
    single_obs = make_obs(NUM_ENVS, OBS_DIM)
    single_actor = _make_actor(single_obs)
    single_critic = MLPModel(single_obs, OBS_GROUPS, "critic", 1, hidden_dims=[32, 32])
    single_storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, single_obs, [NUM_ACTIONS])
    single_ppo = PPO(single_actor, single_critic, single_storage, schedule="fixed")

    multi_ppo, _ = _make_multi_critic_ppo()
    checkpoint = single_ppo.save()

    with pytest.raises(ValueError, match="does not match current model"):
        multi_ppo.load(checkpoint, {"actor": False, "critic": True, "optimizer": False, "iteration": False}, True)


def test_construct_algorithm_rejects_non_mlp_multi_critic() -> None:
    obs = make_obs(NUM_ENVS, OBS_DIM)
    env = SimpleNamespace(num_actions=NUM_ACTIONS, num_envs=NUM_ENVS)
    cfg = _make_construct_algorithm_cfg("CNNModel")

    with pytest.raises(ValueError, match="only supports MLPModel critic"):
        PPO.construct_algorithm(obs, env, cfg, "cpu")


@pytest.mark.parametrize("critic_class_name", ["MLPModel", "rsl_rl.models.MLPModel", "rsl_rl.models:MLPModel", MLPModel])
def test_construct_algorithm_accepts_supported_mlp_specs(critic_class_name: object) -> None:
    obs = make_obs(NUM_ENVS, OBS_DIM)
    env = SimpleNamespace(num_actions=NUM_ACTIONS, num_envs=NUM_ENVS)
    cfg = _make_construct_algorithm_cfg(critic_class_name)

    alg = PPO.construct_algorithm(obs, env, cfg, "cpu")

    assert isinstance(alg.critic, MultiCriticModel)
