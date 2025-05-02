from __future__ import annotations

import torch as th
import torch.nn as nn

from icecream import ic

import rsl_rl.network.network_cfg as network_cfg

#TODO add transformer block

#################### FEATURE EXTRACTOR BLOCKS ####################

class FeatureExtractor(nn.Module):
    def __init__(self, cfg: network_cfg.FeatureExtractorConfig):
        super(FeatureExtractor, self).__init__()
        self.cfg = cfg
        self.network = cfg.network_cfg.class_type(cfg.network_cfg)

    def forward(self, x: th.Tensor,
                ctx: dict[str, th.Tensor] | None = None) -> th.Tensor:
        if isinstance(self.network, nn.Identity):
            return self.network(x)
        else:
            return self.network(x, ctx)



#################### Aggregation Blocks (over history) ####################

class AggregationBlock(nn.Module):
    def __init__(self, cfg: network_cfg.AggregationBlockConfig):
        super(AggregationBlock, self).__init__()
        self.cfg = cfg
        self.network = cfg.network_cfg.class_type(cfg.network_cfg)

    def forward(self, x: th.Tensor,
                ctx: dict[str, th.Tensor] | None = None) -> th.Tensor:
        s = x.shape
        if len(s) > 2:
            # we have to flatten dimension
            x = x.reshape(s[0], -1)
        if isinstance(self.network, nn.Identity):
            return self.network(x)
        else:
            return self.network(x, ctx)
    

class HistoryAggregationBlock(AggregationBlock):
    cfg: network_cfg.HistoryAggregationBlockConfig
    def __init__(self, cfg: network_cfg.HistoryAggregationBlockConfig):
        super(HistoryAggregationBlock, self).__init__(cfg)

    def forward(self, x: th.Tensor,
                ctx: dict[str, th.Tensor] | None = None) -> th.Tensor:
        s = x.shape
        if len(s) > 3:
            x = x.reshape(*s[:2], -1)
        if isinstance(self.network, nn.Identity):
            x = self.network(x)
        else:
            x = self.network(x, ctx)
        if self.cfg.flatten_history:
            x = x.reshape(x.shape[0], -1)
        return x
        
        

class RNNAggregationBlock(HistoryAggregationBlock):
    def __init__(self, cfg: network_cfg.RNNAggregationBlockConfig):
        super(RNNAggregationBlock, self).__init__(cfg)
        
    def forward(self, x: th.Tensor,
                h: th.Tensor,
                ctx: dict[str, th.Tensor] | None = None) -> tuple[th.Tensor, th.Tensor]:
        s = x.shape
        if len(s) > 3:
            # we have to flatten dimension -> (B, H, D1, D2) -> (B, H, D1 * D2)
            x = x.reshape(*s[:2], -1)
        out, h = self.network(x, h, ctx)
        if self.cfg.flatten_history:
            # (B, H, D1) -> (B, H * D1)
            out = out.reshape(out.shape[0], -1)
        return out, h
    

#################### Feature Fusion Blocks ####################

class FuserBlock(nn.Module):
    def __init__(self, cfg: network_cfg.FuserBlockConfig):
        super(FuserBlock, self).__init__()
        self.cfg = cfg
        self.network = cfg.network_cfg.class_type(cfg.network_cfg)

    def forward(self, x: dict[str, th.Tensor]) -> th.Tensor:
        o = th.cat([v for v in x.values()], dim=-1)
        return self.network(o)
    
    
#test code for extractor block
def test_extractor_block():
    import torch as th
    from rsl_rl.network.network_cfg import (
        FeatureExtractorConfig, 
        MLPConfig, 
        CNN2DConfig
    )
    projection_cfg = MLPConfig(
        activation="relu",
        hidden_dim=[64],
    )
    
    # Test case 1: Feature extractor with CNN2D
    cnn_cfg = CNN2DConfig(
        input_channels=3,
        channels=[16, 32, 64],
        kernel_sizes=[3, 3, 3],
        strides=[2, 2, 2],
        paddings=[1, 1, 1],
        activation="relu",
        use_batchnorm=True,
        flatten_output=True,
        output_dim=[128],
        input_dim=[3, 64, 64],
        projection_cfg=projection_cfg
    )
    
    extractor_cfg = FeatureExtractorConfig(
        network_cfg=cnn_cfg
    )
    
    extractor = FeatureExtractor(extractor_cfg)
    # Input shape: (batch_size, 3, 64, 64)
    x = th.randn(4, 3, 64, 64)
    ctx = {}
    output = extractor(x, ctx)
    print(f"Extractor with CNN2D - Output shape: {output.shape}, Expected: torch.Size([4, 128])")
    assert output.shape == (4, 128)
    
    # Test case 2: Feature extractor with MLP
    mlp_cfg = MLPConfig(
        input_dim=[10],
        output_dim=[32],
        hidden_dim=[64, 128],
        activation="relu"
    )
    
    extractor_cfg = FeatureExtractorConfig(
        network_cfg=mlp_cfg
    )
    
    extractor = FeatureExtractor(extractor_cfg)
    # Input shape: (batch_size, 10)
    x = th.randn(4, 10)
    ctx = {}
    output = extractor(x, ctx)
    print(f"Extractor with MLP - Output shape: {output.shape}, Expected: torch.Size([4, 32])")
    assert output.shape == (4, 32)
    
    # Test case 3: Feature extractor with batch in history dimension
    mlp_cfg = MLPConfig(
        input_dim=[5, 10],  # History dim of 5, feature dim of 10
        output_dim=[32],
        hidden_dim=[64],
        activation="relu"
    )
    
    extractor_cfg = FeatureExtractorConfig(
        network_cfg=mlp_cfg
    )
    
    extractor = FeatureExtractor(extractor_cfg)
    # Input shape: (batch_size, history, features)
    x = th.randn(4, 5, 10)
    ctx = {}
    output = extractor(x, ctx)
    print(f"Extractor with history - Output shape: {output.shape}, Expected: torch.Size([4, 5, 32])")
    assert output.shape == (4, 5, 32)
    
    print("All FeatureExtractor tests passed!")

# test aggregation block
def test_aggregation_block():
    import torch as th
    from rsl_rl.network.network_cfg import (
        AggregationBlockConfig,
        HistoryAggregationBlockConfig,
        RNNAggregationBlockConfig,
        MLPConfig,
        CNN1DConfig,
        RNNWrapperConfig,
        NoopConfig
    )
    
    # Test case 1: Basic aggregation with MLP (no history)
    mlp_cfg = MLPConfig(
        input_dim=[50],  # Flattened features
        output_dim=[32],
        hidden_dim=[64],
        activation="relu"
    )
    
    agg_cfg = AggregationBlockConfig(
        network_cfg=mlp_cfg
    )
    
    agg_block = agg_cfg.class_type(agg_cfg)
    # Input shape: (batch_size, features)
    x = th.randn(4, 50)
    output = agg_block(x)
    print(f"AggregationBlock with MLP - Output shape: {output.shape}, Expected: torch.Size([4, 32])")
    assert output.shape == (4, 32)
    
    # Test case 2: Aggregation with Noop (identity)
    noop_cfg = NoopConfig(
        input_dim=[50],
        output_dim=[50]
    )
    
    agg_cfg = AggregationBlockConfig(
        network_cfg=noop_cfg
    )
    
    agg_block = agg_cfg.class_type(agg_cfg)
    # Input shape: (batch_size, features)
    x = th.randn(4, 50)
    output = agg_block(x)
    print(f"AggregationBlock with Noop - Output shape: {output.shape}, Expected: torch.Size([4, 50])")
    assert output.shape == (4, 50)
    
    # Test case 3: History aggregation with Conv1D
    projection_cfg = MLPConfig(
        activation="relu",
        hidden_dim=None,
    )
    cnn1d_cfg = CNN1DConfig(
        input_channels=10,
        channels=[16, 32, 64],
        kernel_sizes=[3, 3, 3],
        strides=[1, 1, 1],
        paddings=[1, 1, 1],
        activation="relu",
        use_batchnorm=True,
        flatten_output=True,
        output_dim=[128],
        input_dim=[10, 32],
        projection_cfg=projection_cfg
    )
    
    hist_agg_cfg = HistoryAggregationBlockConfig(
        network_cfg=cnn1d_cfg,
        flatten_history=True
    )
    
    hist_agg_block = hist_agg_cfg.class_type(hist_agg_cfg)
    # Input shape: (batch_size, history_len, features)
    x = th.randn(4, 10, 32)
    output = hist_agg_block(x)
    print(f"HistoryAggregationBlock with CNN1D - Output shape: {output.shape}, Expected: torch.Size([4, 128])")
    assert output.shape == (4, 128)
    
    # Test case 4: History aggregation with MLP without flattening
    mlp_cfg = MLPConfig(
        input_dim=[32],
        output_dim=[64],
        hidden_dim=[128],
        activation="relu"
    )
    
    hist_agg_cfg = HistoryAggregationBlockConfig(
        network_cfg=mlp_cfg,
        flatten_history=False
    )
    
    hist_agg_block = hist_agg_cfg.class_type(hist_agg_cfg)
    # Input shape: (batch_size, history_len, features)
    x = th.randn(4, 10, 32)
    output = hist_agg_block(x)
    print(f"HistoryAggregationBlock with MLP (no flatten) - Output shape: {output.shape}, Expected: torch.Size([4, 10, 64])")
    assert output.shape == (4, 10, 64)
    
    # Test case 5: RNN aggregation (GRU)
    rnn_cfg = RNNWrapperConfig(
        input_dim=32,
        output_dim=64,
        hidden_size=128,
        num_layers=2,
        rnn_cls="gru"
    )
    
    rnn_agg_cfg = RNNAggregationBlockConfig(
        network_cfg=rnn_cfg,
        flatten_history=False,
    )
    
    rnn_agg_block = rnn_agg_cfg.class_type(rnn_agg_cfg)
    # Input shape: (batch_size, history_len, features)
    x = th.randn(4, 10, 32)
    h = th.zeros(2, 4, 128)  # num_layers, batch_size, hidden_dim
    output, h_out = rnn_agg_block(x, h)
    print(f"RNNAggregationBlock (GRU) - Output shape: {output.shape}, Expected: torch.Size([4, 10, 128])")
    assert output.shape == (4, 10, 128)
    assert h_out.shape == (2, 4, 128)
    
    # Test case 6: RNN aggregation (LSTM) with flattened history
    rnn_cfg = RNNWrapperConfig(
        input_dim=[10, 32],
        output_dim=[64],
        hidden_size=128,
        num_layers=2,
        rnn_cls="lstm"
    )
    
    rnn_agg_cfg = RNNAggregationBlockConfig(
        network_cfg=rnn_cfg,
        flatten_history=True,
    )
    
    rnn_agg_block = RNNAggregationBlock(rnn_agg_cfg)
    # Input shape: (batch_size, history_len, features)
    x = th.randn(4, 10, 32)
    h = (th.zeros(2, 4, 128), th.zeros(2, 4, 128))  # LSTM state: (h, c)
    output, h_out = rnn_agg_block(x, h)
    print(f"RNNAggregationBlock (LSTM) with flatten - Output shape: {output.shape}, Expected: torch.Size([4, 1280])")
    assert output.shape == (4, 1280)  # batch_size, history_len * hidden_dim = 4, 10 * 128
    assert len(h_out) == 2  # LSTM returns tuple of (h, c)
    assert h_out[0].shape == (2, 4, 128)
    assert h_out[1].shape == (2, 4, 128)
    
    print("All AggregationBlock tests passed!")

# test fuser block
def test_fuser_block():
    import torch as th
    from rsl_rl.network.network_cfg import (
        FuserBlockConfig,
        MLPConfig,
        NoopConfig
    )
    
    # Test case 1: Feature fusion with MLP
    mlp_cfg = MLPConfig(
        input_dim=[128],  # Combined feature dimension
        output_dim=[64],
        hidden_dim=[256, 128],
        activation="relu"
    )
    
    fuser_cfg = FuserBlockConfig(
        network_cfg=mlp_cfg,
        input_dim=[128],
        output_dim=[64]
    )
    
    fuser_block = FuserBlock(fuser_cfg)
    # Create dictionary of features to fuse
    features = {
        "visual": th.randn(4, 64),
        "proprioception": th.randn(4, 32),
        "task": th.randn(4, 32)
    }
    # Total feature dimension: 64 + 32 + 32 = 128
    output = fuser_block(features)
    print(f"FuserBlock with MLP - Output shape: {output.shape}, Expected: torch.Size([4, 64])")
    assert output.shape == (4, 64)
    
    # Test case 2: Feature fusion with Noop (identity)
    noop_cfg = NoopConfig(
        input_dim=[128],
        output_dim=[128]
    )
    
    fuser_cfg = FuserBlockConfig(
        network_cfg=noop_cfg,
        input_dim=[128],
        output_dim=[128]
    )
    
    fuser_block = FuserBlock(fuser_cfg)
    # Create dictionary of features to fuse
    features = {
        "visual": th.randn(4, 64),
        "proprioception": th.randn(4, 64)
    }
    # Total feature dimension: 64 + 64 = 128
    output = fuser_block(features)
    print(f"FuserBlock with Noop - Output shape: {output.shape}, Expected: torch.Size([4, 128])")
    assert output.shape == (4, 128)
    
    # Test case 3: Feature fusion with multiple dimensions (batch and sequence)
    mlp_cfg = MLPConfig(
        input_dim=[96],  # Combined feature dimension
        output_dim=[32],
        hidden_dim=[64],
        activation="relu"
    )
    
    fuser_cfg = FuserBlockConfig(
        network_cfg=mlp_cfg,
        input_dim=[96],
        output_dim=[32]
    )
    
    fuser_block = FuserBlock(fuser_cfg)
    # Create dictionary of features to fuse with batch and sequence dimensions
    features = {
        "sensor1": th.randn(4, 5, 32),  # (batch, seq, feat1)
        "sensor2": th.randn(4, 5, 64)   # (batch, seq, feat2)
    }
    # Total feature dimension: 32 + 64 = 96
    output = fuser_block(features)
    print(f"FuserBlock with sequence dim - Output shape: {output.shape}, Expected: torch.Size([4, 5, 32])")
    assert output.shape == (4, 5, 32)
    
    print("All FuserBlock tests passed!")

if __name__ == "__main__":
    print("Running FeatureExtractor tests...")
    test_extractor_block()
    
    print("\nRunning AggregationBlock tests...")
    test_aggregation_block()
    
    print("\nRunning FuserBlock tests...")
    test_fuser_block()
    
    print("\nAll tests completed successfully!")