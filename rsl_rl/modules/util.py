from __future__ import annotations
from typing import Tuple, List, Iterable

import torch as th
import torch.nn as nn
import numpy as np
from flash_attn.modules.mha import MHA
import einops

from torch.distributions import Categorical, Independent 

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None

def merge_shapes(*dims) -> Tuple[int, ...]:
    """
    Concatenate multiple dims into one.
    Args:
        dims: either an int or a list of ints.
    Returns:
        concatenated dims.
    """
    out: List[int] = []
    for d in dims:
        if isinstance(d, Iterable):
            out.extend(d)
        else:
            out.append(d)
    return tuple(out)

class LinearBn(nn.Module):
    """ Linear layer with optimal batch normalization. """

    def __init__(self, dim_in: int, dim_out: int,
                 use_bn: bool = True,
                 use_ln: bool = False, **kwds):
        super().__init__()

        if use_bn and use_ln:
            raise ValueError('use_bn and use_ln cannot both be true!')

        affine = kwds.pop('affine', True)
        if use_bn or use_ln:
            if use_bn:
                kwds['bias'] = False
            self.linear = nn.Linear(dim_in, dim_out, **kwds)
            if use_ln:
                self.bn = nn.LayerNorm(dim_out, elementwise_affine=affine)
            else:
                self.bn = nn.BatchNorm1d(dim_out, affine=affine)
        else:
            self.linear = nn.Linear(dim_in, dim_out, **kwds)
            self.bn = nn.Identity()

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.linear(x)
        s = x.shape
        x = x.reshape(-1, s[-1])
        x = self.bn(x)
        x = x.reshape(s)
        return x

class MLP(nn.Module):
    """ Generic multilayer perceptron. """

    def __init__(self, dims: Tuple[int, ...],
                 act_cls: nn.Module = nn.LeakyReLU,
                 activate_output: bool = False,
                 use_bn: bool = True,
                 bias: bool = True,
                 use_ln: bool = False,
                 pre_ln_bias: bool = True,
                 affine: bool = True):
        super().__init__()
        assert (len(dims) >= 2)

        if isinstance(act_cls, str):
            act_cls = get_activation(act_cls)

        layers = []
        for d0, d1 in zip(dims[:-2], dims[1:-1]):
            # FIXME(ycho): incorrect `bias` logic
            if not use_ln:
                layer_bias = bias
            else:
                layer_bias = pre_ln_bias
            layers.extend(
                (LinearBn(
                    d0,
                    d1,
                    use_bn=use_bn,
                    bias=layer_bias,
                    use_ln=use_ln,
                    affine=affine),
                    act_cls,
                 ))
        if activate_output:
            layers.extend((
                LinearBn(
                    dims[-2],
                    dims[-1],
                    use_bn=use_bn, bias=bias, use_ln=use_ln,
                    affine=affine),
                act_cls))
        else:
            # FIXME(ycho): not much I can do here except
            # hardcoding... for now
            layers.extend((
                nn.Linear(dims[-2], dims[-1], bias=bias),)
            )
        self.model = nn.Sequential(*layers)

    def forward(self, x: th.Tensor):
        return self.model(x)
    
class SplitDim(nn.Module):
    def __init__(self,
                 sizes:Tuple[int,...],
                 dim:int=-1):
        super().__init__()
        self.dim=dim
        self.splits = np.cumsum(sizes)[:-1].tolist()

    def forward(self, x:th.Tensor):
        return th.tensor_split(x, self.splits, dim=self.dim)
    
class MHAWrapper(MHA):
    def forward(self, q, m):
        s = q.shape
        q = einops.rearrange(q, '... s d -> (...) s d')
        m = einops.rearrange(m, '... s d -> (...) s d')
        o = super().forward(q, m)
        o = o.reshape(*s[:-2], *o.shape[-2:])
        return o

class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=None):
        self.masks = masks
        if masks is None:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.device = self.masks.device
            logits = th.where(self.masks, logits, th.tensor(-1e+8).to(self.device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
    
    def rsample(self):
        u = th.distributions.Uniform(low=th.zeros_like(self.logits, device = self.logits.device),
                                     high=th.ones_like(self.logits, device = self.logits.device)).sample()
        #print(u.size(), self.logits.size())
        rand_logits = self.logits -(-u.log()).log()
        return th.max(rand_logits, axis=-1)[1]

    def entropy(self):
        if self.masks is None:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = th.where(self.masks, p_log_p, th.tensor(0.0).to(self.device))
        return -p_log_p.sum(-1)