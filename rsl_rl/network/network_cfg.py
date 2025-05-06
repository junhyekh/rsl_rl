from __future__ import annotations
from dataclasses import MISSING
from isaaclab.utils import configclass
from typing import Sequence

import torch as th
import torch.nn as nn


from .base import MLP, CNN2D, CNN1D, ResNet, RNNWrapper
from .block import (FeatureExtractor, 
                    AggregationBlock, 
                    RNNAggregationBlock,
                    HistoryAggregationBlock,
                    FuserBlock)

@configclass
class NetworkBaseConfig:
    class_type: type = MISSING
    input_dim: list[int] = MISSING
    output_dim: list[int] = MISSING

@configclass
class MLPConfig(NetworkBaseConfig):
    class_type = MLP
    hidden_dim: int | list[int] | None = MISSING
    activation: str | nn.Module = 'relu'
    use_layernorm: bool = False
    last_layer_activation: str | nn.Module | None = None

@configclass
class RNNWrapperConfig(NetworkBaseConfig):
    class_type = RNNWrapper
    hidden_size: int = MISSING
    num_layers: int = MISSING
    rnn_cls: str = MISSING

@configclass
class CNNBaseConfig(NetworkBaseConfig):
    class_type = MISSING
    input_channels: int = MISSING
    channels: list[int] = MISSING
    kernel_sizes: int | list[int] = MISSING
    strides: int | list[int] = MISSING
    paddings: int | list[int] = MISSING
    activation: str | nn.Module = MISSING
    use_batchnorm: bool = MISSING
    flatten_output: bool = MISSING
    projection_cfg: MLPConfig | None = None

    def __post_init__(self):
        # if isinstance(self.input_dim, Sequence):
        #     self.input_channels = self.input_dim[-2]
        # Convert scalar parameters to lists if needed
        if isinstance(self.kernel_sizes, int):
            self.kernel_sizes = [self.kernel_sizes] * len(self.channels)
        if isinstance(self.strides, int):
            self.strides = [self.strides] * len(self.channels)
        if isinstance(self.paddings, int):
            self.paddings = [self.paddings] * len(self.channels)
            

@configclass
class CNN2DConfig(CNNBaseConfig):
    class_type = CNN2D

@configclass
class CNN1DConfig(CNNBaseConfig):
    class_type = CNN1D

@configclass
class ResNetConfig(CNNBaseConfig):
    class_type = ResNet
    layers: list[int] = MISSING

#################### FEATURE EXTRACTOR CONFIGS ####################
    
@configclass
class FeatureExtractorConfig:
    class_type = FeatureExtractor
    network_cfg: NetworkBaseConfig = MISSING

#################### AGGREGATION BLOCK CONFIGS ####################

@configclass
class AggregationBlockConfig(FeatureExtractorConfig):
    class_type = AggregationBlock
    network_cfg: NetworkBaseConfig = MISSING

@configclass
class HistoryAggregationBlockConfig(AggregationBlockConfig):
    class_type = HistoryAggregationBlock
    flatten_history: bool = False

@configclass
class RNNAggregationBlockConfig(HistoryAggregationBlockConfig):
    class_type = RNNAggregationBlock
    network_cfg: RNNWrapperConfig = MISSING

#################### FEATURE FUSION BLOCK CONFIGS ####################

@configclass
class FuserBlockConfig(NetworkBaseConfig):
    class_type = FuserBlock
    network_cfg: NetworkBaseConfig = MISSING

#################### BASIC NETWORK CONFIGS ####################

@configclass
class NoopConfig(NetworkBaseConfig):
    class_type: type = nn.Identity

@configclass
class FlattenConfig(NetworkBaseConfig):
    class_type: type = nn.Flatten
    start_dim: int = 1
    end_dim: int = -1


@configclass
class NetworkConfig:
    feature_block_cfg: dict[str, FeatureExtractorConfig] = MISSING
    aggregation_block_cfg: dict[str, AggregationBlockConfig] = MISSING
    fuser_block_cfg: FuserBlockConfig = MISSING
    state_aggr_block_cfg: AggregationBlockConfig = MISSING
