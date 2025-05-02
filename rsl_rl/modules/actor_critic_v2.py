from __future__ import annotations
from typing import List, Tuple

from dataclasses import MISSING

from icecream import ic

import torch as th
import torch.nn as nn

from isaaclab.utils import configclass

import rsl_rl.network.network_cfg as network_cfg
from torch.distributions import Normal, Categorical, Independent
from rsl_rl.network.util import CategoricalMasked

import numpy as np


@configclass
class NetworkConfig:
    feature_block_cfg: dict[str, network_cfg.FeatureExtractorConfig] = MISSING
    aggregation_block_cfg: dict[str, network_cfg.AggregationBlockConfig] = MISSING
    fuser_block_cfg: network_cfg.FuserBlockConfig = MISSING
    state_aggr_block_cfg: network_cfg.AggregationBlockConfig = MISSING
    

@configclass
class ActorCriticNetConfig:
    num_obs: dict[str, list[int]] = MISSING
    num_critic_obs: dict[str, list[int]]|None = None
    num_actions: int|list[int] = MISSING
    actor_cfg: NetworkConfig = MISSING
    critic_cfg: NetworkConfig|None = None
    is_discrete: bool = False
    is_multi_discrete: bool = False
    init_noise_std: float = 1.0
    
    def __post_init__(self):
        if self.critic_cfg is None:
            self.critic_cfg = self.actor_cfg.copy()
            # critic output dimension must be 1 for now (value dim)
            # TODO what if we have multiple value heads?
            self.critic_cfg.state_aggr_block_cfg.network_cfg.output_dim = [1]

class ActorCriticV2(nn.Module):
    is_recurrent = False
    def __init__(self,
                 cfg: ActorCriticNetConfig
                 ):
        super().__init__()
        self.cfg = cfg
        self._require_mask = False
        if self.cfg.num_critic_obs is None:
            self.cfg.num_critic_obs = self.cfg.num_obs

        if self.cfg.is_discrete:
            if isinstance(self.cfg.num_actions, int):
                n_bins = self.cfg.num_actions
                n_actions = 1
            else:
                n_bins = self.cfg.num_actions[0]
                n_actions = len(self.cfg.num_actions)
            self.cfg.actor_cfg.state_aggr_block_cfg.network_cfg.output_dim = [n_bins * n_actions]
            self._action_shape = (n_actions, n_bins)
            self._action_output_dim = [n_bins * n_actions]

        elif self.cfg.is_multi_discrete:
            if isinstance(self.cfg.num_actions, int):
                raise ValueError("num_actions must be a list for multi-discrete action space")
            else:
                n_bins = max(list(self.cfg.num_actions))
                n_actions = len(self.cfg.num_actions)
            self._require_mask = len(set(self.cfg.num_actions))>1
            if self._require_mask:
                mask = th.zeros(
                    n_actions, n_bins,
                    dtype=th.bool
                )   
                for i, dim in enumerate(self.cfg.num_actions):
                    mask[i, :dim] = 1
                self._mask = nn.Parameter(mask[None],
                                        requires_grad=False
                                        )
            self.cfg.actor_cfg.state_aggr_block_cfg.network_cfg.output_dim = [n_bins * n_actions]
            self._action_shape = (n_actions, n_bins)
            self._action_output_dim = [n_bins * n_actions]
        else:
            self._action_shape = self.cfg.num_actions
            self._action_output_dim = self.cfg.num_actions
            ic(self._action_shape, self._action_output_dim)
            self.std = nn.Parameter(self.cfg.init_noise_std * th.ones(self.cfg.num_actions))
            Normal.set_default_validate_args = False
   
        (actor_feature_blocks, 
         actor_aggregation_blocks, 
         actor_fuser_block, 
         actor_state_aggr_block) = self.__configure_network(self.cfg.actor_cfg,
                                                            self.cfg.num_obs)
        (critic_feature_blocks, 
         critic_aggregation_blocks, 
         critic_fuser_block, 
         critic_state_aggr_block) = self.__configure_network(self.cfg.critic_cfg,
                                                             self.cfg.num_critic_obs)
        
        # actor
        self.actor_feature_blocks = nn.ModuleDict(actor_feature_blocks)
        self.actor_aggregation_blocks = nn.ModuleDict(actor_aggregation_blocks)
        self.actor_fuser_block = actor_fuser_block
        self.actor_state_aggr_block = actor_state_aggr_block

        # critic
        self.critic_feature_blocks = nn.ModuleDict(critic_feature_blocks)
        self.critic_aggregation_blocks = nn.ModuleDict(critic_aggregation_blocks)
        self.critic_fuser_block = critic_fuser_block
        self.critic_state_aggr_block = critic_state_aggr_block

        # print network structure
        ic(self)
    
    def __configure_network(self,
                            cfg: NetworkConfig,
                            obs_dim: dict[str, list[int]]) -> tuple[dict[str, nn.Module], dict[str, nn.Module], nn.Module, nn.Module]:
        """
        In here, we sequentially update the network cfg and generate the network
        order:
         - feature block cfg 
         - aggregation block cfg
         - fuser block cfg
         - state aggregation block cfg
        1, we update the feature block cfg based on the input dimension and generate the feature block
        2, we update the aggregation block cfg based on the feature block output dimension
        3, we update the fuser block cfg based on the aggregation block output dimension
        4, we update the state aggregation block cfg based on the fuser block output dimension

        If the network_cfg is not noop (identity) we don't have to update the cfg
        """
        feature_blocks = {}
        aggregation_blocks = {}
        dummy_outputs = {}
        ic(obs_dim)
        # update feature block cfg
        for k, v in cfg.feature_block_cfg.items():
            v.network_cfg.input_dim = obs_dim[k]
            ic(k, v.network_cfg.input_dim, obs_dim[k])
            if isinstance(v.network_cfg.class_type, network_cfg.CNN2DConfig):
                assert len(obs_dim[k]) >=3, "CNN input dimension must be longer than 3"
                v.network_cfg.input_channels = obs_dim[k][-3]
            feature_blocks[k] = v.class_type(v)
            dummy_input = th.randn(v.network_cfg.input_dim)[None]
            dummy_output = feature_blocks[k](dummy_input)
            v.network_cfg.output_dim = list(dummy_output.shape[1:])
            dummy_outputs[k] = dummy_output
        # update aggregation block cfg
        for k, v in cfg.aggregation_block_cfg.items():
            v.network_cfg.input_dim = cfg.feature_block_cfg[k].network_cfg.output_dim
            if isinstance(v.network_cfg, network_cfg.CNN1DConfig):
                assert len(v.network_cfg.input_dim) >= 3, "CNN1D input dimension must be longer than 3 with history"
                v.network_cfg.input_channels = v.network_cfg.input_dim[1]
            aggregation_blocks[k] = v.class_type(v)
            dummy_output = aggregation_blocks[k](dummy_outputs[k])
            v.network_cfg.output_dim = list(dummy_output.shape)
            dummy_outputs[k] = dummy_output
        # update fuser block cfg
        fuser_block_cfg = cfg.fuser_block_cfg
        fuser_block_cfg.network_cfg.input_dim = np.array([v.network_cfg.output_dim 
                                                                        for v in cfg.aggregation_block_cfg.values()]).sum()
        fuser_block = fuser_block_cfg.class_type(fuser_block_cfg)
        dummy_output = fuser_block(dummy_outputs)
        fuser_block_cfg.network_cfg.output_dim = list(dummy_output.shape[1:])

        # update state aggregation block cfg
        state_aggr_block_cfg = cfg.state_aggr_block_cfg
        state_aggr_block_cfg.network_cfg.input_dim = fuser_block_cfg.network_cfg.output_dim
        state_aggr_block = state_aggr_block_cfg.class_type(state_aggr_block_cfg)
        dummy_output = state_aggr_block(dummy_output)
        assert len(dummy_output.shape) == 2, "State aggregation block output dimension must be 2"
        self.cfg.actor_cfg.state_aggr_block_cfg.network_cfg.output_dim = dummy_output.shape[1:]
        return feature_blocks, aggregation_blocks, fuser_block, state_aggr_block

            
    def forward(self):
        pass

    def reset(self, dones=None):
        pass
        # TODO have to handle rnn reset

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
    
    @property
    def is_continuous(self):
        return not (self.cfg.is_discrete or self.cfg.is_multi_discrete)
    
    def _forward_actor(self, obs:dict[str, th.Tensor]):
        obs = dict(obs)
        for k, v in self.actor_feature_blocks.items():
            obs[k] = v(obs[k])
        for k, v in self.actor_aggregation_blocks.items():
            obs[k] = v(obs[k])
        obs = self.actor_fuser_block(obs)
        obs = self.actor_state_aggr_block(obs)
        return obs
    
    def _forward_critic(self, obs:dict[str, th.Tensor]):
        obs = dict(obs)
        for k, v in self.critic_feature_blocks.items():
            obs[k] = v(obs[k])
        for k, v in self.critic_aggregation_blocks.items():
            obs[k] = v(obs[k])
        obs = self.critic_fuser_block(obs)
        obs = self.critic_state_aggr_block(obs)
        return obs
        
    def update_distribution(self, observations):
        if self.cfg.is_discrete:
            logits = self._forward_actor(observations).view(-1, *self._action_shape)
            self.distribution = Categorical(logits=logits)
        elif self.cfg.is_multi_discrete:
            logits = self._forward_actor(observations).view(-1, *self._action_shape)
            if self._require_mask:
                self.distribution = CategoricalMasked(logits=logits,
                                                    masks=self._mask)
            else:
                self.distribution = Categorical(logits=logits)
        else:
            mean = self._forward_actor(observations)
            self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_inference(self, observations):
        if self.cfg.is_discrete:
            logits = self._forward_actor(observations).view(-1, *self._action_shape)
            return logits.argmax(-1)
        elif self.cfg.is_multi_discrete:
            logits = self._forward_actor(observations).view(-1, *self._action_shape)
            if self._require_mask:
                logits = th.where(self._mask,
                                    logits,
                                    th.tensor(-1e+8).to(self._mask.device))
            return logits.argmax(-1)
        else:
            actions_mean = self._forward_actor(observations)
            return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self._forward_critic(critic_observations)
        return value
    


def test_actor_critic_v2():
    input_shape = {
        "a": [10, 5],
        "b": [10, 3, 64, 64],
        "c": [10, 7]
    }
    num_actions = [64, 3, 5]
    num_critic_obs = None
    feature_block_cfg = {
        "a": network_cfg.FeatureExtractorConfig(
            network_cfg=network_cfg.MLPConfig(
                output_dim=[64],
                hidden_dim=[128, 128],
                activation="relu"
            )
        ),
        "b": network_cfg.FeatureExtractorConfig(
            network_cfg=network_cfg.CNN2DConfig(
                input_channels=3,
                channels=[16, 16, 64],
                use_batchnorm=False,
                strides=[2, 2, 2],
                kernel_sizes=[3, 3, 3],
                paddings=[1, 1, 1],
                flatten_output=True,
                activation="relu",
                output_dim=None
            )
        ),
        "c": network_cfg.FeatureExtractorConfig(
            network_cfg=network_cfg.MLPConfig(
                output_dim=[64],
                hidden_dim=[128, 128],
                activation="relu"
            )
        )
    }
    aggregation_block_cfg = {
        "a": network_cfg.AggregationBlockConfig(
            network_cfg=network_cfg.NoopConfig()
        ),
        "b": network_cfg.AggregationBlockConfig(
            network_cfg=network_cfg.NoopConfig()
        ),
        "c": network_cfg.AggregationBlockConfig(
            network_cfg=network_cfg.NoopConfig()
        )
    }
    fuser_block_cfg = network_cfg.FuserBlockConfig(
        network_cfg=network_cfg.NoopConfig()
    )
    state_aggr_block_cfg = network_cfg.AggregationBlockConfig(
        network_cfg=network_cfg.MLPConfig(
            output_dim=[64],
            hidden_dim=[256, 128],
            activation="relu"
        )
    )
    cfg = ActorCriticNetConfig(
        num_obs=input_shape,    
        num_critic_obs=None,
        num_actions=num_actions,
        actor_cfg=NetworkConfig(
            feature_block_cfg=feature_block_cfg,
            aggregation_block_cfg=aggregation_block_cfg,
            fuser_block_cfg=fuser_block_cfg,
            state_aggr_block_cfg=state_aggr_block_cfg
        ),
        is_multi_discrete=True
    )
    actor_critic = ActorCriticV2(cfg.copy())
    # ic(actor_critic)
    obs={k: th.randn(input_shape[k]) for k in input_shape}
    # test multi-discrete
    print("multi-discrete")
    ic(actor_critic.act(obs).shape,
       actor_critic.act_inference(obs).shape)
    ic(actor_critic.evaluate(obs).shape)
    # test discrete
    print("discrete")
    cfg.is_discrete = True
    cfg.is_multi_discrete = False
    cfg.num_actions = [64]*3
    actor_critic = ActorCriticV2(cfg.copy())
    ic(actor_critic.act(obs).shape,
       actor_critic.act_inference(obs).shape)
    ic(actor_critic.evaluate(obs).shape)
    # test continuous
    print("continuous")
    cfg.is_discrete = False
    cfg.is_multi_discrete = False
    cfg.num_actions = [64]
    actor_critic = ActorCriticV2(cfg.copy())
    ic(actor_critic.act(obs).shape,
       actor_critic.act_inference(obs).shape)
    ic(actor_critic.evaluate(obs).shape)

    
if __name__ == "__main__":
    test_actor_critic_v2()