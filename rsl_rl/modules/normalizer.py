#  Copyright (c) 2020 Preferred Networks, Inc.
#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from collections.abc import Mapping

import torch
from torch import nn


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, shape, eps=1e-2, until=None,
                 keys: list[str] | None = None):
        """Initialize EmpiricalNormalization module.

        Args:
            shape (int or tuple of int): Shape of input values except batch axis.
            eps (float): Small value for stability.
            until (int or None): If this arg is specified, the link learns input values until the sum of batch sizes
            exceeds it.
            keys (list[str] or None): If this arg is specified, the link learns input values only for the keys.
        """
        super().__init__()
        self.eps = eps
        self.until = until
        if keys is not None:
            assert isinstance(shape, Mapping), "shape must be a mapping if keys is specified"
        if isinstance(shape, Mapping):
            self.__keys = []
            for k,v in shape.items():
                if keys is not None and k not in keys:
                    continue
                self.__keys.append(k)
                if isinstance(v, int):
                    v = [v]
                self.register_buffer(f"{k}_mean", torch.zeros(*v).unsqueeze(0))
                self.register_buffer(f"{k}_var", torch.ones(*v).unsqueeze(0))
                self.register_buffer(f"{k}_std", torch.ones(*v).unsqueeze(0))
        else:
            self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
            self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
            self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        self.count = 0

    @property
    def mean(self):
        if hasattr(self, '_mean'):
            return self._mean.squeeze(0).clone()
        else:
            return {k: getattr(self, f"{k}_mean").squeeze(0).clone()
                    for k in self.__keys}

    @property
    def std(self):
        if hasattr(self, '_std'):
            return self._std.squeeze(0).clone()
        else:
            return {k: getattr(self, f"{k}_std").squeeze(0).clone()
                    for k in self.__keys}

    def forward(self, x):
        """Normalize mean and variance of values based on empirical values.

        Args:
            x (ndarray or Variable): Input values

        Returns:
            ndarray or Variable: Normalized output values
        """
        if isinstance(x, Mapping):
            x = dict(x)
            if self.training:
                self.update(x)
            # for k,v in x.items():
            for k in self.__keys:
                v = x[k].clone()
                _mean = getattr(self, f'{k}_mean')
                _std = getattr(self, f'{k}_std')
                x[k] = (v - _mean) / (_std + self.eps)
            return x
        else:
            if self.training:
                self.update(x)
            return (x - self._mean) / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x):
        """Learn input values without computing the output values of them"""

        if self.until is not None and self.count >= self.until:
            return
        if isinstance(x, dict):
            count_x = next(iter(x.values())).shape[0]
            self.count += count_x
            rate = count_x / self.count
            # for k,v in x.items():
            for k in self.__keys:
                v = x[k]
                var_x = torch.var(v, dim=0, unbiased=False, keepdim=True)
                mean_x = torch.mean(v, dim=0, keepdim=True)

                _mean = getattr(self, f'{k}_mean')
                _var = getattr(self, f'{k}_var')
                delta_mean = mean_x - _mean
                _mean += rate * delta_mean
                _var += rate * (var_x - _var + delta_mean * (mean_x - _mean))
                setattr(self, f'{k}_mean', _mean)
                setattr(self, f'{k}_var', _var)
                setattr(self, f'{k}_std', torch.sqrt(_var))
        else:
            count_x = x.shape[0]
            self.count += count_x
            rate = count_x / self.count

            var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
            mean_x = torch.mean(x, dim=0, keepdim=True)

            delta_mean = mean_x - self._mean
            self._mean += rate * delta_mean
            self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
            self._std = torch.sqrt(self._var)

    @torch.jit.unused
    def inverse(self, y):
        if isinstance(y, Mapping):
            return {k: y[k] * (getattr(self, f"{k}_std") + self.eps) 
                    + getattr(self, f"{k}_mean")
                    for k in self.__keys}
        else:
            return y * (self._std + self.eps) + self._mean

class EmpiricalDiscountedVariationNormalization(nn.Module):
    """Reward normalization from Pathak's large scale study on PPO.

    Reward normalization. Since the reward function is non-stationary, it is useful to normalize
    the scale of the rewards so that the value function can learn quickly. We did this by dividing
    the rewards by a running estimate of the standard deviation of the sum of discounted rewards.
    """

    def __init__(self, shape, eps=1e-2, gamma=0.99, until=None):
        super().__init__()

        self.emp_norm = EmpiricalNormalization(shape, eps, until)
        self.disc_avg = DiscountedAverage(gamma)

    def forward(self, rew):
        if self.training:
            # update discounected rewards
            avg = self.disc_avg.update(rew)

            # update moments from discounted rewards
            self.emp_norm.update(avg)

        if self.emp_norm._std > 0:
            return rew / self.emp_norm._std
        else:
            return rew
        

class DiscountedAverage:
    r"""Discounted average of rewards.

    The discounted average is defined as:

    .. math::

        \bar{R}_t = \gamma \bar{R}_{t-1} + r_t

    Args:
        gamma (float): Discount factor.
    """

    def __init__(self, gamma):
        self.avg = None
        self.gamma = gamma

    def update(self, rew: torch.Tensor) -> torch.Tensor:
        if self.avg is None:
            self.avg = rew
        else:
            self.avg = self.avg * self.gamma + rew
        return self.avg