from __future__ import annotations

import numpy as np

from typing import TYPE_CHECKING
from icecream import ic

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union, Type
if TYPE_CHECKING:
    from rsl_rl.network.network_cfg import (MLPConfig, 
                          CNNBaseConfig,
                          ResNetConfig,
                          RNNWrapperConfig)
from rsl_rl.network.util import get_activation, merge_shapes

class MLP(nn.Module):
    def __init__(self, cfg: MLPConfig):
        """
        Multilayer Perceptron network.
        
        Args:
            cfg: Configuration object from network_cfg.py containing:
                - input_dim: Input dimension
                - output_dim: Output dimension
                - hidden_dim: Hidden layer dimensions (int or list of ints)
                - last_layer_activation: Activation for the last layer (optional)
                - activation: Activation for hidden layers (default: "relu")
                - use_layernorm: Whether to use layer normalization (default: False)
        """
        super(MLP, self).__init__()
        
        self.cfg = cfg
        
        assert self.cfg.input_dim is not None, "input_dim must be provided in cfg"
        assert self.cfg.output_dim is not None, "output_dim must be provided in cfg"
        

        if not isinstance(self.cfg.activation, nn.Module):
            activation = get_activation(self.cfg.activation)
            if activation is None:  # Default to ReLU if invalid activation provided
                activation = nn.ReLU()
        
        if isinstance(self.cfg.hidden_dim, int):
            hidden_dim = [self.cfg.hidden_dim]
        elif self.cfg.hidden_dim is None:
            hidden_dim = []
        else:
            hidden_dim = self.cfg.hidden_dim
        
        layers = []
        current_dim = self.cfg.input_dim[-1]
        for i in range(len(hidden_dim)):
            layers.append(nn.Linear(current_dim, hidden_dim[i]))
            if self.cfg.use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim[i]))
            layers.append(activation)
            current_dim = hidden_dim[i]
        layers.append(nn.Linear(current_dim, self.cfg.output_dim[-1]))
        if self.cfg.last_layer_activation is not None:
            if not isinstance(self.cfg.last_layer_activation, nn.Module):
                last_layer_activation = get_activation(self.cfg.last_layer_activation)
                if last_layer_activation is None:
                    last_layer_activation = nn.Identity()
            if self.cfg.use_layernorm:
                layers.append(nn.LayerNorm(self.cfg.output_dim[-1]))
            layers.append(last_layer_activation)
        self.model = nn.Sequential(*layers)

    def forward(self, x: th.Tensor,
                ctx: dict[str, th.Tensor]|None = None) -> th.Tensor:
        return self.model(x)
    
# Base CNN class for shared code between CNN1D and CNN2D
class CNNBase(nn.Module):
    def __init__(self, cfg: CNNBaseConfig, conv_type: str = "2d"):
        """
        Base CNN architecture for 1D and 2D CNNs.
        
        Args:
            cfg: Configuration object from network_cfg.py containing:
                - input_channels: Number of input channels
                - channels: List of channel sizes for each convolutional layer
                - output_dim: Output dimension (optional, None if no FC layer is needed)
                - kernel_sizes: Kernel sizes for each layer (default: 3)
                - strides: Strides for each layer (default: 1)
                - paddings: Paddings for each layer (default: 1)
                - activation: Activation function (default: "relu")
                - use_batchnorm: Whether to use batch normalization (default: False)
                - flatten_output: Whether to flatten the output (default: True)
                - input_dim: Spatial dimension + channel dimension (default: None)
            conv_type: Type of convolution ("1d" or "2d")
        """
        super(CNNBase, self).__init__()
        
        self.cfg = cfg
        
        assert cfg.input_channels is not None, "input_channels must be provided in cfg"
        assert cfg.channels is not None, "channels must be provided in cfg"
        
        # Set convolution and batch normalization types based on dimensionality
        if conv_type == "1d":
            self.conv_layer = nn.Conv1d
            self.bn_layer = nn.BatchNorm1d
        elif conv_type == "2d":
            self.conv_layer = nn.Conv2d
            self.bn_layer = nn.BatchNorm2d
        else:
            raise ValueError(f"Invalid conv_type: {conv_type}. Expected '1d' or '2d'.")
        
        self.conv_type = conv_type
        
        # Convert activation to module if it's a string
        if not isinstance(cfg.activation, nn.Module):
            activation = get_activation(cfg.activation)
            if activation is None:  # Default to ReLU if invalid activation provided
                activation = nn.ReLU()
            
        # Ensure all parameter lists have correct lengths
        assert len(cfg.kernel_sizes) == len(cfg.channels)
        assert len(cfg.strides) == len(cfg.channels)
        assert len(cfg.paddings) == len(cfg.channels)
        
        # Build convolutional layers
        layers = []
        in_channels = cfg.input_channels
        
        for i, out_channels in enumerate(cfg.channels):
            layers.append(self.conv_layer(
                in_channels, 
                out_channels,
                kernel_size=cfg.kernel_sizes[i],
                stride=cfg.strides[i],
                padding=cfg.paddings[i]
            ))
            
            if cfg.use_batchnorm:
                layers.append(self.bn_layer(out_channels))
                
            layers.append(activation)
            in_channels = out_channels
        
        self.cnn_layers = nn.Sequential(*layers)
        
        # Calculate output spatial dimensions if input_size is provided
        self.output_size = self._calculate_output_size()
        
        # Fully connected layer if needed
        if cfg.flatten_output and cfg.output_dim is not None:
            flattened_dim = in_channels * np.prod(self.output_size)
            assert cfg.projection_cfg is not None, "project_MLP must be provided in cfg"
            cfg.projection_cfg.input_dim = [flattened_dim]
            cfg.projection_cfg.output_dim = cfg.output_dim
            self.projection = self.cfg.projection_cfg.class_type(self.cfg.projection_cfg)
        else:
            self.projection = None
    
    def _calculate_output_size(self):
        """
        Calculate the output spatial dimensions after applying convolutional layers.
        
        Args:
            input_size: Tuple with input spatial dimensions (H, W) for 2D or (L,) for 1D
            kernel_sizes: List of kernel sizes for each layer
            strides: List of strides for each layer
            paddings: List of paddings for each layer
            
        Returns:
            Tuple with output spatial dimensions
        """ 
            
        current_size = list(self.cfg.input_dim[-1:] if self.conv_type == "1d" else self.cfg.input_dim[-2:])
        # Calculate the number of convolutional layers
        num_conv_layers = 0
        for module in self.cnn_layers:
            if isinstance(module, (nn.Conv1d, nn.Conv2d)):
                num_conv_layers += 1
                
        # Apply the formula for each dimension and each layer
        layer_idx = 0
        for i, module in enumerate(self.cnn_layers):
            if not isinstance(module, (nn.Conv1d, nn.Conv2d)):
                continue
                
            if self.conv_type == "1d":
                # For 1D convolution, update only length
                k_size = self.cfg.kernel_sizes[layer_idx] if layer_idx < len(self.cfg.kernel_sizes) else 3
                s_size = self.cfg.strides[layer_idx] if layer_idx < len(self.cfg.strides) else 1
                p_size = self.cfg.paddings[layer_idx] if layer_idx < len(self.cfg.paddings) else 1
                
                current_size[0] = int((current_size[0] - k_size + 2 * p_size) / s_size) + 1
            else:  # 2d
                # For 2D convolution, update both height and width
                for j in range(len(current_size)):
                    k_size = self.cfg.kernel_sizes[layer_idx] if layer_idx < len(self.cfg.kernel_sizes) else 3
                    s_size = self.cfg.strides[layer_idx] if layer_idx < len(self.cfg.strides) else 1
                    p_size = self.cfg.paddings[layer_idx] if layer_idx < len(self.cfg.paddings) else 1
                    
                    # Handle tuple kernel_sizes, strides, and paddings
                    if isinstance(k_size, (tuple, list)):
                        k_size = k_size[j] if j < len(k_size) else k_size[0]
                    if isinstance(s_size, (tuple, list)):
                        s_size = s_size[j] if j < len(s_size) else s_size[0]
                    if isinstance(p_size, (tuple, list)):
                        p_size = p_size[j] if j < len(p_size) else p_size[0]
                    
                    current_size[j] = int((current_size[j] - k_size + 2 * p_size) / s_size) + 1
            
            layer_idx += 1
        return tuple(current_size)
    
    def forward(self, x: th.Tensor,
                ctx: dict[str, th.Tensor]|None = None) -> th.Tensor:
        x = self.cnn_layers(x)
        if self.cfg.flatten_output:
            x = x.flatten(start_dim=1)
            if self.projection is not None:
                x = self.projection(x)
        
        return x

# CNN with Conv2d
class CNN2D(CNNBase):
    def __init__(self, cfg: CNNBaseConfig):
        """
        Basic 2D CNN architecture.
        
        Args:
            cfg: Configuration object from network_cfg.py containing:
                - input_channels: Number of input channels
                - channels: List of channel sizes for each convolutional layer
                - output_dim: Output dimension (optional)
                - kernel_sizes: Kernel sizes for each layer (default: 3)
                - strides: Strides for each layer (default: 1)
                - paddings: Paddings for each layer (default: 1)
                - activation: Activation function (default: "relu")
                - use_batchnorm: Whether to use batch normalization (default: False)
                - flatten_output: Whether to flatten the output (default: True)
        """
        super(CNN2D, self).__init__(cfg=cfg, conv_type="2d")
    
    def forward(self, x: th.Tensor,
                ctx: dict[str, th.Tensor]|None = None) -> th.Tensor:
        s = x.shape
        if len(s) > 4:
            # we have (B, history, H, W, C)
            # convert to (B * history, H, W, C)
            x = x.reshape(-1, *s[2:])
        # (B * history, H, W, C) -> (B * history, C, H, W)
        x = super().forward(x, ctx)
        
        if self.cfg.flatten_output:
            # (B * history, D1) -> (B, history, D1)
            x = x.reshape(*s[:-3], -1)
        else:
            # (B * history, C, H, W) -> (B, history, C, H, W)
            x = x.reshape(*s[:-3], *x.shape[-3:])
        return x

# CNN with Conv1d
class CNN1D(CNNBase):
    def __init__(self, cfg: CNNBaseConfig):
        """
        Basic 1D CNN architecture.
        """
        
        super(CNN1D, self).__init__(cfg=cfg, conv_type="1d")

    def forward(self, x: th.Tensor,
                ctx: dict[str, th.Tensor]|None = None) -> th.Tensor:
        # (B, H, C) -> (B, C, H)
        # x = x.transpose(1, 2)
        s = x.shape
        if len(s) > 3:
            # we have (B, history, D1, D2)
            # convert to (B * history, D1, D2)
            x = x.reshape(*s[:2], -1)
        x = super(CNN1D, self).forward(x, ctx)
        return x

# basic residual block for Resnet
class ResidualBlock(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 stride: int = 1, 
                 downsample: nn.Module = None, 
                 activation: Union[str, nn.Module] = "relu", 
                 use_batchnorm: bool = True):
        """
        Basic residual block for ResNet architectures.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for the first convolutional layer
            downsample: Optional downsampling layer for skip connection
            activation: Activation function
            use_batchnorm: Whether to use batch normalization
        """
        super(ResidualBlock, self).__init__()
        
        if not isinstance(activation, nn.Module):
            activation = get_activation(activation)
            if activation is None:  # Default to ReLU if invalid activation provided
                activation = nn.ReLU()
        self.activation: nn.Module = activation
        # Conv1 block
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1, 
            bias=not use_batchnorm
        )
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        
        # Conv2 block
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=not use_batchnorm
        )
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        
        # Downsample for residual connection if needed
        self.downsample = downsample
    
    def forward(self, x: th.Tensor) -> th.Tensor:
        identity = x
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply downsample to identity if needed
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add residual connection
        out += identity
        
        out = self.activation(out)
            
        return out

# Resnet based on residual block
class ResNet(nn.Module):
    def __init__(self, cfg: ResNetConfig):
        """
        ResNet architecture based on residual blocks.
        
        Args:
            cfg: Configuration object from network_cfg.py containing:
                - input_channels: Number of input channels
                - output_dim: Output dimension
                - layers: List of number of blocks in each layer (default: [2, 2, 2, 2])
                - channels: List of channel dimensions for each layer (default: [64, 128, 256, 512])
                - activation: Activation function (default: "relu")
                - use_batchnorm: Whether to use batch normalization (default: True)
                - zero_init_residual: Whether to initialize residual block batch norm to zero (default: False)
                - flatten_output: Whether to flatten the output (default: True)
        """
        super(ResNet, self).__init__()
        
        self.cfg = cfg
        
        assert self.cfg.input_channels is not None, "input_channels must be provided in cfg"
        assert self.cfg.output_dim is not None, "output_dim must be provided in cfg"
        
        # Get activation type
        if not isinstance(self.cfg.activation, nn.Module):
            activation = get_activation(self.cfg.activation)
            if activation is None:
                activation = nn.ReLU()
        self.activation: nn.Module = activation
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(self.cfg.input_channels, self.cfg.channels[0], kernel_size=7, stride=2, padding=3, bias=not self.cfg.use_batchnorm)
        self.bn1 = nn.BatchNorm2d(self.cfg.channels[0]) if self.cfg.use_batchnorm else nn.Identity()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(ResidualBlock, 
                                       self.cfg.channels[0], 
                                       self.cfg.channels[0], 
                                       self.cfg.layers[0], 
                                       stride=self.cfg.strides[0], 
                                       activation=self.cfg.activation, 
                                       use_batchnorm=self.cfg.use_batchnorm)
        self.layer2 = self._make_layer(ResidualBlock, 
                                      self.cfg.channels[0], 
                                      self.cfg.channels[1], 
                                      self.cfg.layers[1], 
                                      stride=self.cfg.strides[1], 
                                      activation=self.cfg.activation, 
                                      use_batchnorm=self.cfg.use_batchnorm)
        self.layer3 = self._make_layer(ResidualBlock, 
                                      self.cfg.channels[1], 
                                      self.cfg.channels[2], 
                                      self.cfg.layers[2], 
                                      stride=self.cfg.strides[2], 
                                      activation=self.cfg.activation, 
                                      use_batchnorm=self.cfg.use_batchnorm)
        self.layer4 = self._make_layer(ResidualBlock, 
                                      self.cfg.channels[2], 
                                      self.cfg.channels[3], 
                                      self.cfg.layers[3], 
                                      stride=self.cfg.strides[3], 
                                      activation=self.cfg.activation, 
                                      use_batchnorm=self.cfg.use_batchnorm)
        
        # Global average pooling and final fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten_output = self.cfg.flatten_output
        if self.cfg.flatten_output:
            self.fc = nn.Linear(self.cfg.channels[3], self.cfg.output_dim[-1])
        else:
            self.fc = None
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # # Zero-initialize the last BN in each residual branch
        # if self.cfg.zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, ResidualBlock):
        #             for n, c in m.named_children():
        #                 if n == 'bn2':
        #                     nn.init.constant_(c.weight, 0)
    
    def _make_layer(self, 
                    block_type: Type[nn.Module], 
                    in_channels: int, 
                    out_channels: int, 
                    blocks: int, 
                    stride: int = 1, 
                    activation: Union[str, nn.Module] = "relu", 
                    use_batchnorm: bool = True) -> nn.Sequential:
        """
        Create a layer of blocks for the ResNet.
        
        Args:
            block_type: Block module class to use
            in_channels: Input channels
            out_channels: Output channels
            blocks: Number of blocks in the layer
            stride: Stride for the first block
            activation: Activation function
            use_batchnorm: Whether to use batch normalization
        """
        downsample = None
        
        # Create downsample path if dimensions change
        if stride != 1 or in_channels != out_channels:
            downsample_layers = []
            downsample_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=not use_batchnorm))
            if use_batchnorm:
                downsample_layers.append(nn.BatchNorm2d(out_channels))
            downsample = nn.Sequential(*downsample_layers)
        
        # Create layer with blocks
        layers = []
        # First block may have stride and downsample
        layers.append(block_type(in_channels, out_channels, stride, downsample, activation, use_batchnorm))
        
        # Remaining blocks are simpler
        for _ in range(1, blocks):
            layers.append(block_type(out_channels, out_channels, 1, None, activation, use_batchnorm))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: th.Tensor,
                ctx: dict[str, th.Tensor]|None = None) -> th.Tensor:
        # Initial conv and pooling
        x = self.conv1(x)
        x = self.bn1(x)
        
        x = self.activation(x)
        
            
        x = self.maxpool(x)
        
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Final processing
        x = self.avgpool(x)
        if self.fc is not None:
            x = th.flatten(x, 1)
            x = self.fc(x)
        
        return x
    

class RNNWrapper(nn.Module):
    def __init__(self, cfg: RNNWrapperConfig):
        super(RNNWrapper, self).__init__()
        self.cfg = cfg
        rnn_cls = nn.GRU if cfg.rnn_cls.lower() == "gru" else nn.LSTM
        if isinstance(cfg.input_dim, list):
            input_dim = cfg.input_dim[-1]
        else:
            input_dim = cfg.input_dim
        self.rnn = rnn_cls(input_size=input_dim,
                          hidden_size=cfg.hidden_size,
                          num_layers=cfg.num_layers,
                          batch_first=True)
    def forward(self, x: th.Tensor,
                h: th.Tensor|None = None,
                ctx: dict[str, th.Tensor]|None = None) -> tuple[th.Tensor, th.Tensor]:
        return self.rnn(x, h)
        
        
# test code for MLP
def test_mlp():
    import torch as th
    from rsl_rl.network.network_cfg import MLPConfig
    
    # Test case 1: Basic MLP with single hidden layer
    cfg = MLPConfig(
        input_dim=[10],
        output_dim=[5],
        hidden_dim=32,
        activation="relu"
    )
    mlp = MLP(cfg)
    x = th.randn(2, 10)  # Batch size 2, 10 features
    ctx = {}
    output = mlp(x, ctx)
    print(mlp)
    print(f"MLP test 1 - Output shape: {output.shape}, Expected: torch.Size([2, 5])")
    assert output.shape == (2, 5)
    
    # Test case 2: MLP with multiple hidden layers
    cfg = MLPConfig(
        input_dim=[10],
        output_dim=[5],
        hidden_dim=[32, 16],
        activation="relu"
    )
    mlp = MLP(cfg)
    x = th.randn(2, 10)
    output = mlp(x, ctx)
    print(mlp)
    print(f"MLP test 2 - Output shape: {output.shape}, Expected: torch.Size([2, 5])")
    assert output.shape == (2, 5)
    
    # Test case 3: MLP with layer normalization and custom final activation
    cfg = MLPConfig(
        input_dim=[10],
        output_dim=[5],
        hidden_dim=[32, 16],
        use_layernorm=True,
        activation="relu",
        last_layer_activation="tanh"
    )
    mlp = MLP(cfg)
    x = th.randn(2, 10)
    output = mlp(x, ctx)
    print(mlp)
    print(f"MLP test 3 - Output shape: {output.shape}, Expected: torch.Size([2, 5])")
    assert output.shape == (2, 5)
    assert th.all(output >= -1.0) and th.all(output <= 1.0)  # Check tanh bounds
    
    # Test case 4: MLP with history dimension
    cfg = MLPConfig(
        input_dim=[7, 10],
        output_dim=[5],
        hidden_dim=[32, 16],
        activation="relu"
    )
    mlp = MLP(cfg)
    x = th.randn(2, 7, 10)
    output = mlp(x, ctx)
    print(mlp)
    print(f"MLP test 4 - Output shape: {output.shape}, Expected: torch.Size([2, 7,5])")
    assert output.shape == (2, 7, 5)

    print("All MLP tests passed!")

# test code for CNN1D
def test_cnn1d():
    import torch as th
    import numpy as np
    from rsl_rl.network.network_cfg import CNNBaseConfig, MLPConfig, CNN1DConfig
    
    # Test case 1: With flatten_output and projection_cfg
    projection_cfg = MLPConfig(
        activation="relu",
        hidden_dim=[64],
    )
    
    cfg = CNN1DConfig(
        input_channels=32,
        channels=[16, 32, 64],
        kernel_sizes=[3, 3, 5],
        strides=[1, 1, 1],
        paddings=[1, 1, 1],
        activation="relu",
        use_batchnorm=True,
        flatten_output=True,
        output_dim=[10],
        input_dim=(32, 12),  # 1D input of length 32
        projection_cfg=projection_cfg
    )
    print(cfg)
    cnn1d = CNN1D(cfg)
    print(cnn1d)
    # Input: (batch_size, sequence_length, channels)
    x = th.randn(2, 32, 12)
    ctx = {}
    output = cnn1d(x, ctx)
    print(f"CNN1D test 1 - Output shape: {output.shape}, Expected: torch.Size([2, 10])")
    assert output.shape == (2, 10)
        
    # Test case 2: Without flatten_output
    cfg = CNNBaseConfig(
        input_channels=32,
        channels=[16, 32, 64],
        kernel_sizes=[3, 3, 5],
        strides=[1, 1, 1],
        paddings=[1, 1, 1],
        activation="relu",
        use_batchnorm=True,
        flatten_output=False,
        output_dim=[64],    
        input_dim=(32,12)
    )
    
    cnn1d = CNN1D(cfg)
    x = th.randn(2, 32, 12)
    output = cnn1d(x, ctx)
    
    # Expected output channels = last channel size (64)
    # Expected output sequence length determined by calculation
    expected_length = 32  # Same as input since stride=1 and padding=1
    print(f"CNN1D test 2 - Output shape: {output.shape}, Expected: with 64 channels")
    assert output.shape[1] == 64  # Check channels
    
    print("All CNN1D tests passed!")

# test code for CNN2D
def test_cnn2d():
    import torch as th
    import numpy as np
    from rsl_rl.network.network_cfg import CNNBaseConfig, MLPConfig
    
    # Test case 1: With flatten_output and projection_cfg
    projection_cfg = MLPConfig(
        activation="relu",
        hidden_dim=[64],
    )
    
    cfg = CNNBaseConfig(
        input_channels=3,
        channels=[16, 32, 64],
        kernel_sizes=[3, 3, 3],
        strides=[2, 2, 2],
        paddings=[1, 1, 1],
        activation="relu",
        use_batchnorm=True,
        flatten_output=True,
        output_dim=[10],
        input_dim=(3, 64, 64),  # 64x64 image
        projection_cfg=projection_cfg
    )
    
    cnn2d = CNN2D(cfg)
    print(cnn2d)
    # Input: (batch_size, channel, height, width)
    x = th.randn(2, *cfg.input_dim)
    ctx = {}
    output = cnn2d(x, ctx)
    print(f"CNN2D test 1 - Output shape: {output.shape}, Expected: torch.Size([2, 10])")
    assert output.shape == (2, 10)
    
    # Test case 2: With history dimension
    cfg = CNNBaseConfig(
        input_channels=3,
        channels=[16, 32, 64],
        kernel_sizes=[3, 3, 3],
        strides=[2, 2, 2],
        paddings=[1, 1, 1],
        activation="relu",
        use_batchnorm=True,
        flatten_output=True,
        output_dim=[10],
        input_dim=(3, 64, 64),
        projection_cfg=projection_cfg
    )
    
    cnn2d = CNN2D(cfg)
    # Input: (batch_size, channels, history, height, width)
    x = th.randn(2, 4, *cfg.input_dim)  # Batch size 2, history length 4
    output = cnn2d(x, ctx)
    print(f"CNN2D test 2 - Output shape: {output.shape}, Expected: torch.Size([2, 4, 10])")
    assert output.shape == (2, 4, 10)
    
    # Test case 3: Without flatten_output
    cfg = CNNBaseConfig(
        input_channels=3,
        channels=[16, 32, 64],
        kernel_sizes=[3, 3, 3],
        strides=[2, 2, 2],
        paddings=[1, 1, 1],
        activation="relu",
        use_batchnorm=True,
        flatten_output=False,
        input_dim=(3, 64, 64)
    )
    
    cnn2d = CNN2D(cfg)
    x = th.randn(2, *cfg.input_dim)
    output = cnn2d(x, ctx)
    
    # Expected shape: (batch_size, channels, height, width)
    # Calculate expected output dimensions
    h_out = 8  # 64 -> 32 -> 16 -> 8 with strides of 2
    w_out = 8  # 64 -> 32 -> 16 -> 8 with strides of 2
    c_out = 64  # last channel size
    
    print(f"CNN2D test 3 - Output shape: {output.shape}, Expected: torch.Size([2, 64, 8, 8])")
    assert output.shape[1] == c_out
    assert output.shape[2] == h_out
    assert output.shape[3] == w_out
    
    # Test case 4: With history dim and without flatten_output
    cfg = CNNBaseConfig(
        input_channels=3,
        channels=[16, 32, 64],
        kernel_sizes=[3, 3, 3],
        strides=[2, 2, 2],
        paddings=[1, 1, 1],
        activation="relu",
        use_batchnorm=True,
        flatten_output=False,
        input_dim=(4, 3, 64, 64)
    )
    
    cnn2d = CNN2D(cfg)
    # Input: (batch_size, channels, history, height, width)
    x = th.randn(2, *cfg.input_dim)
    output = cnn2d(x, ctx)
    
    print(f"CNN2D test 4 - Output shape: {output.shape}, Expected: torch.Size([2, 4, 64, 8, 8])")
    assert output.shape == (2, 4, 64, 8, 8)
    
    print("All CNN2D tests passed!")

# Test Resnet
def test_resnet():
    import torch as th
    from rsl_rl.network.network_cfg import ResNetConfig
    
    # Test case 1: Basic ResNet
    cfg = ResNetConfig(
        input_channels=3,
        output_dim=[10],
        flatten_output=True,
        layers=[2, 2, 2, 2],
        channels=[64, 128, 256, 512],
        kernel_sizes=[3, 3, 3, 3],
        strides=[1, 1, 1, 1],
        paddings=[1, 1, 1, 1],
        activation="relu",
        use_batchnorm=False,
        input_dim=(3, 64, 64)
    )
    
    resnet = ResNet(cfg)
    print(resnet)
    # Input: (batch_size, channels, height, width)
    x = th.randn(2, *cfg.input_dim)
    ctx = {}
    output = resnet(x, ctx)
    print(f"ResNet test 1 - Output shape: {output.shape}, Expected: torch.Size([2, 10])")
    assert output.shape == (2, 10)
    
    # Test case 2: ResNet without flatten_output
    cfg = ResNetConfig(
        input_channels=1,
        output_dim=[10],
        flatten_output=False,
        layers=[1, 1, 1, 1],
        channels=[16, 128, 256, 512],
        kernel_sizes=[3, 3, 3, 3],
        strides=[1, 2,2,2],
        paddings=[1, 1, 1, 1],
        activation="relu",
        use_batchnorm=False,
        input_dim=(1, 64, 64)
    )
    
    resnet = ResNet(cfg)
    x = th.randn(2, *cfg.input_dim)
    output = resnet(x, ctx)
    print(resnet)
    # Expected shape: (batch_size, channels, 1, 1) due to the adaptive pooling
    print(f"ResNet test 2 - Output shape: {output.shape}")
    assert output.shape[1] == 512  # Default final channel count
    assert output.shape[2] == 1 and output.shape[3] == 1  # Due to adaptive pooling
    
    # Test case 3: ResNet with custom layers and channels
    cfg = ResNetConfig(
        input_channels=3,
        output_dim=[10],
        layers=[1, 1, 1, 1],
        channels=[32, 64, 128, 256],
        flatten_output=True,
        input_dim=(3, 64, 64),
        strides=1,
        paddings=1,
        activation="relu",
        use_batchnorm=False,
    )
    
    resnet = ResNet(cfg)
    x = th.randn(2, *cfg.input_dim)
    output = resnet(x, ctx)
    print(f"ResNet test 3 - Output shape: {output.shape}, Expected: torch.Size([2, 10])")
    assert output.shape == (2, 10)
    
    print("All ResNet tests passed!")

if __name__ == "__main__":
    print("Running MLP tests...")
    test_mlp()
    
    print("\nRunning CNN1D tests...")
    test_cnn1d()
    
    print("\nRunning CNN2D tests...")
    test_cnn2d()
    
    print("\nRunning ResNet tests...")
    test_resnet()
    
    print("\nAll tests completed successfully!")