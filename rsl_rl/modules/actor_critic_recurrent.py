# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.utils import resolve_nn_activation, unpad_trajectories


class ActorCriticRecurrent(ActorCritic):
    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_size=256,
        rnn_num_layers=1,
        init_noise_std=1.0,
        base_hidden_dims: Optional[List[int]] = None,
        use_layernorm: bool = False,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )

        super().__init__(
            num_actor_obs=rnn_hidden_size,
            num_critic_obs=rnn_hidden_size,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )

        if base_hidden_dims is not None:
            base_activation = resolve_nn_activation(activation)

            actor_base_layers = []
            critic_base_layers = []

            actor_base_layers.append(nn.Linear(num_actor_obs, base_hidden_dims[0]))
            actor_base_layers.append(base_activation)

            critic_base_layers.append(nn.Linear(num_critic_obs, base_hidden_dims[0]))
            critic_base_layers.append(base_activation)

            for layer_index in range(len(base_hidden_dims)):
                if layer_index == len(base_hidden_dims) - 1:
                    actor_base_layers.append(nn.Linear(base_hidden_dims[layer_index],
                                                rnn_hidden_size))
                    actor_base_layers.append(base_activation)
                    critic_base_layers.append(nn.Linear(base_hidden_dims[layer_index],
                                               rnn_hidden_size))
                    critic_base_layers.append(base_activation)
                else:
                    actor_base_layers.append(nn.Linear(base_hidden_dims[layer_index],
                                                base_hidden_dims[layer_index + 1]))
                    actor_base_layers.append(base_activation)
                    critic_base_layers.append(nn.Linear(base_hidden_dims[layer_index],
                                                base_hidden_dims[layer_index + 1]))
                    critic_base_layers.append(base_activation)
            self.actor_base = nn.Sequential(*actor_base_layers)
            self.critic_base = nn.Sequential(*critic_base_layers)
            memory_a_input_dim = rnn_hidden_size
            memory_c_input_dim = rnn_hidden_size

        else:
            self.actor_base = None
            self.critic_base = None
            memory_a_input_dim = num_actor_obs
            memory_c_input_dim = num_critic_obs

        self.memory_a = Memory(memory_a_input_dim, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        self.memory_c = Memory(memory_c_input_dim, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)

        if self.actor_base is not None:
            print(f"Actor Base: {self.actor_base}")
            print(f"Critic Base: {self.critic_base}")

        print(f"Actor RNN: {self.memory_a}")
        print(f"Critic RNN: {self.memory_c}")

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def act(self, observations, masks=None, hidden_states=None):
        input_a = self.memory_a(observations, masks, hidden_states)
        return super().act(input_a.squeeze(0))

    def act_inference(self, observations):
        input_a = self.memory_a(observations)
        return super().act_inference(input_a.squeeze(0))

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        input_c = self.memory_c(critic_observations, masks, hidden_states)
        return super().evaluate(input_c.squeeze(0))

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states


class Memory(torch.nn.Module):
    def __init__(self, input_size, type="lstm", num_layers=1, hidden_size=256):
        super().__init__()
        # RNN
        rnn_cls = nn.GRU if type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None

    def forward(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None
        if batch_mode:
            # batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(input, hidden_states)
            out = unpad_trajectories(out, masks)
        else:
            # inference mode (collection): use hidden states of last step
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
        return out

    def reset(self, dones=None):
        # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
        if self.hidden_states is None:
            return
        for hidden_state in self.hidden_states:
            hidden_state[..., dones == 1, :] = 0.0
