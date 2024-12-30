#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical, Independent
from rsl_rl.modules.util import CategoricalMasked, get_activation


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        use_layernorm=False,
        is_discrete=False,
        is_multi_discrete=False,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = get_activation(activation)

        self._is_discrete = is_discrete
        self._is_multi_discrete = is_multi_discrete
        self.is_continuous = (not (is_discrete or is_multi_discrete))
        if is_discrete:
            num_actions = num_actions[0]
        elif is_multi_discrete:
            max_len = max(list(num_actions))
            n_actions = len(num_actions)
            self._require_mask = len(set(num_actions))>1
            if self._require_mask:
                mask = torch.zeros(
                    n_actions, max_len,
                    dtype=torch.bool
                ) 
                for i, dim in enumerate(num_actions):
                    mask[i, :dim] = 1
                self._mask = nn.Parameter(mask[None],
                                        requires_grad=False
                                        )
            num_actions = max_len * n_actions
            self._action_shape = (n_actions, max_len)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                if use_layernorm:
                    actor_layers.append(nn.LayerNorm(actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                if use_layernorm:
                    critic_layers.append(nn.LayerNorm(critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        if is_discrete or is_multi_discrete:
            pass
        else:
            Normal.set_default_validate_args = False

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales=None):
        f = lambda l, idx: l[idx] if l is not None and idx < len(l) else 1.0
        [
            torch.nn.init.orthogonal_(module.weight, gain=f(scales, idx))
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def action_logit(self):
        return self.distribution.logits
    @property
    def action_prob(self):
        return self.distribution.probs

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        if self._is_discrete:
            logits = self.actor(observations)
            self.distribution = Categorical(logits=logits)
        elif self._is_multi_discrete:
            logits = self.actor(observations).view(-1, *self._action_shape)
            if self._require_mask:
                self.distribution = CategoricalMasked(logits=logits,
                                                    masks=self._mask)
            else:
                self.distribution = Categorical(logits=logits)
        else:
            mean = self.actor(observations)
            self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        if self._is_discrete:
            logits = self.actor(observations)
            return logits.amax(-1)
        elif self._is_multi_discrete:
            logits = self.actor(observations).view(-1, *self._action_shape)
            if self._require_mask:
                logits = torch.where(self._mask,
                                    logits,
                                    torch.tensor(-1e+8).to(observations.device))
            return logits.amax(-1)
        else:
            actions_mean = self.actor(observations)
            return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value


