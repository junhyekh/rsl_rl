#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from typing import Optional, List, Tuple, Dict

import torch as th
import torch.nn as nn

from rsl_rl.modules.actor_critic import get_activation
from rsl_rl.modules.util import merge_shapes, MLP
from opt_einsum import contract_expression


class ActorSubnet(nn.Module):

    # @dataclass
    # class ActionValueSubnetConfig(FeatureBase.Config):
    #     dim_in: Tuple[int, ...] = ()
    #     dim_out: int = 20 + 20 + 1

    #     action_dim: int = 20
    #     value_dim: int = 1
    #     squeeze: bool = True
    #     log_std_init: float = 0.0
    #     hidden: Tuple[int, ...] = (64, 128, 64,)
    #     act_cls: str = 'tanh'
    #     use_bn: bool = False
    #     use_ln: bool = True
    #     affine: bool = True
    #     output_ls: bool = True
    #     share_backbone: bool = False

    #     def __post_init__(self):
    #         if self.output_ls:
    #             self.dim_out = (self.action_dim * 2 +
    #                             self.value_dim)
    #         else:
    #             self.dim_out = (self.action_dim + self.value_dim)

    # Config = ActionValueSubnetConfig

    def __init__(self,
                 dim_in: int,
                 dim_out: int,
                 hidden: Tuple[int, ...] = (64, 128, 64,),
                 act_cls: str|nn.Module = 'elu',
                 use_bn: bool = False,
                 use_ln: bool = True
                 ):
        super().__init__()
        if isinstance(act_cls, str):
            act_cls = get_activation(act_cls)
        
        dims_actor = merge_shapes(dim_in,
                                hidden,
                                dim_out)
        self.action_center = MLP(dims_actor, act_cls, False, use_bn,
                                use_ln=use_ln, affine=False)

    def forward(self, state: th.Tensor,
                ctx: Optional[Dict[str, th.Tensor]] = None):
        
        return self.action_center(state)

class CriticSubnet(nn.Module):

    def __init__(self,
                 dim_in: int,
                 dim_out: int = 1,
                 hidden: Tuple[int, ...] = (64, 128, 64,),
                 act_cls: str|nn.Module = 'elu',
                 use_bn: bool = False,
                 use_ln: bool = True
                 ):
        super().__init__()
        if isinstance(act_cls, str):
            act_cls = get_activation(act_cls)
        
        dims_critic = merge_shapes(dim_in,
                                hidden,
                                dim_out)
        self.value = MLP(dims_critic, act_cls, False, use_bn,
                                use_ln=use_ln, affine=False)

    def forward(self, state: th.Tensor,
                ctx: Optional[Dict[str, th.Tensor]] = None):
        
        return self.value(state)
    
class GateModLinear(nn.Module):
    def __init__(self,
                 d_i: int,
                 d_o: int,
                 act_cls='elu',
                 norm_cls='layernorm',
                 gate: bool = True):
        super().__init__()
        self.d_i = d_i
        self.d_o = d_o
        if act_cls == 'elu':
            self.act = nn.ELU()
        else:
            self.act = nn.Identity()
        if norm_cls == 'layernorm':
            self.norm = nn.LayerNorm(d_o, elementwise_affine=False)
        else:
            self.norm = nn.Identity()

        b = 1024 * 10  # batch size
        i = d_o  # output dim
        m = 8  # num modules
        j = d_i  # input dim
        self.gate = gate
        if gate:
            self.f_Wx = contract_expression(
                'bi,bm,mij,bj->bi',
                (b, i),
                (b, m),
                (m, i, j),
                (b, j),
                #    (b,i),
                #    backend='torch'
            )
            self.f_b = contract_expression(
                'bj,bm,mj->bj',
                (b, j),
                (b, m),
                (m, j),
                #    (b,j),
                #    backend='torch'
            )
        else:
            self.f_Wx = contract_expression(
                'bm,mij,bj->bi',
                (b, m),
                (m, i, j),
                (b, j),
            )
            self.f_b = contract_expression(
                'bm,mj->bj',
                (b, m),
                (m, j)
            )

    def extra_repr(self):
        return F'{self.d_i}->{self.d_o}'

    def forward(self,
                x: th.Tensor,
                # weight/bias (modules)
                Ws: th.Tensor,
                bs: th.Tensor,

                # module selection probabilities
                pW: th.Tensor,
                pb: th.Tensor,

                # output gates
                gW: Optional[th.Tensor] = None,
                gb: Optional[th.Tensor] = None):
        if self.gate:
            Wx = self.f_Wx(gW, pW, Ws, x)
            b = self.f_b(gb, pb, bs)
        else:
            Wx = self.f_Wx(pW, Ws, x)
            b = self.f_b(pb, bs)
        # Wx = th.einsum('...i, ...m, mij, ...j -> ...i', gW, pW, Ws, x)
        # b = th.einsum('...j, ...m, mj -> ...j', gb, pb, bs)
        return self.act(self.norm(Wx + b))

class GateNormModLinear(nn.Module):
    def __init__(self, d_i: int, d_o: int,
                 act_cls='tanh',
                 norm_cls='layernorm',
                 gate: bool = True):
        super().__init__()
        self.d_i = d_i
        self.d_o = d_o
        if act_cls == 'elu':
            self.act = nn.ELU()
        else:
            self.act = nn.Identity()
        if norm_cls == 'layernorm':
            self.norm = nn.LayerNorm(d_o, elementwise_affine=False)
        else:
            self.norm = nn.Identity()

        b = 1024 * 10  # batch size
        i = d_o  # output dim
        m = 8  # num modules
        j = d_i  # input dim
        self.f_Wx = contract_expression('bm,mij,bj->bi',
                                        (b, m),
                                        (m, i, j),
                                        (b, j),
                                        #    (b,i),
                                        #    backend='torch'
                                        )
        self.f_b = contract_expression('bm,mj->bj',
                                       (b, m),
                                       (m, j),
                                       #    (b,j),
                                       #    backend='torch'
                                       )

    def extra_repr(self):
        return F'{self.d_i}->{self.d_o}'

    def forward(self,
                x: th.Tensor,
                # weight/bias (modules)
                Ws: th.Tensor,
                bs: th.Tensor,

                # module selection probabilities
                pW: th.Tensor,
                pb: th.Tensor,

                # output gates
                gW: th.Tensor,
                gb: th.Tensor):
        Wx = self.f_Wx(pW, Ws, x)
        b = self.f_b(pb, bs)
        # Wx = th.einsum('...i, ...m, mij, ...j -> ...i', gW, pW, Ws, x)
        # b = th.einsum('...j, ...m, mj -> ...j', gb, pb, bs)
        # gb is centered at 1
        return self.act(gW*self.norm(Wx + b)+(gb-1))
    
class GateModMLP(nn.Module):
    def __init__(self,
                 dims: Tuple[int, ...],
                 gate: bool = True,
                 after_norm:bool=False):
        super().__init__()
        print(F'GateModMLP got dims = {dims}')
        num_layer: int = len(dims) - 1
        last_idx: int = num_layer - 1
        layer_cls = GateModLinear if not after_norm else GateNormModLinear
        self.gate = gate
        self.layers = nn.ModuleList([
            layer_cls(d_i, d_o,
                          act_cls='elu' if (l != last_idx) else 'none',
                          norm_cls='layernorm' if (l != last_idx) else 'none',
                          gate=gate
                          )
            for l, (d_i, d_o) in enumerate(zip(dims[:-1], dims[1:]))
        ])

    def forward(self, x: th.Tensor,
                # weight/bias (modules)
                # shape: Lx(M,Di,Do)
                Ws: List[th.Tensor],
                # shape: Lx(M,Do)
                bs: List[th.Tensor],

                # module selection probabilities
                # shape: (...,L,M)
                pWs: th.Tensor,
                # shape: (...,L,M)
                pbs: th.Tensor,

                # output gates
                # shape: Lx(..., Do)
                gWs: Optional[th.Tensor] = None,
                # shape: Lx(..., Do)
                gbs: Optional[th.Tensor] = None):
        if self.gate:
            for layer, W, b, pW, pb, gW, gb in zip(self.layers,
                                                   Ws, bs, pWs, pbs, gWs, gbs):
                x = layer(x, W, b, pW, pb, gW.squeeze(dim=-1), gb)
        else:
            for layer, W, b, pW, pb in zip(self.layers, Ws, bs, pWs, pbs):
                x = layer(x, W, b, pW, pb)
        return x