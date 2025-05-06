from __future__ import annotations

from typing import TYPE_CHECKING

from icecream import ic
import numpy as np

import torch as th
import torch.nn as nn

from rsl_rl.network import network_cfg

class NetworkWrapper(nn.Module):
    def __init__(self, cfg: network_cfg.NetworkConfig,
                 obs_dim: dict[str, list[int]]):
        super(NetworkWrapper, self).__init__()
        self.cfg = cfg
        self.configure_network(cfg, obs_dim)
 
    def configure_network(
        self,
        cfg: network_cfg.NetworkConfig,
        obs_dim: dict[str, list[int]]
    ) -> None:
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
        feature_blocks = nn.ModuleDict(feature_blocks)
        aggregation_blocks = nn.ModuleDict(aggregation_blocks)
        self.networks = nn.ModuleDict({
            "feature_blocks": feature_blocks,
            "aggregation_blocks": aggregation_blocks,
            "fuser_block": fuser_block,
            "state_aggr_block": state_aggr_block
        })

    def forward(self, x: dict[str, th.Tensor]) -> th.Tensor:
        x = dict(x)
        for k, v in self.networks['feature_blocks'].items():
            x[k] = v(x[k])
        for k, v in self.networks['aggregation_blocks'].items():
            x[k] = v(x[k])
        x = self.networks['fuser_block'](x)
        x = self.networks['state_aggr_block'](x)
        return x