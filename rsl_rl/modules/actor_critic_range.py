#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.distributions import Normal
from functorch import (make_functional_with_buffers, vmap)

from rsl_rl.modules.actor_critic import get_activation
from rsl_rl.modules.util import merge_shapes, MLP, SplitDim, MHAWrapper
from rsl_rl.modules.network import ActorSubnet, CriticSubnet, GateModMLP



class ModulationAndGateNet(nn.Module):
    def __init__(self,
                 dim_in: int,
                 num_param: int,
                 num_module: int,
                 target_hiddens: Tuple[int, ...],
                 hidden: Tuple[int, ...],
                 init_std: float = 0.008,
                 fused:bool=False,
                 temperature: Optional[float] = None
                 ):
        super().__init__()
        self.num_param = num_param
        self.num_module = num_module
        self.fused=fused

        self.temperature = None
        if temperature is not None:
            self.temperature = temperature

        dims = merge_shapes(dim_in, hidden)
        self.bb = MLP(dims,
                          act_cls=nn.GELU(),
                          activate_output = True,
                          use_bn=False,
                          use_ln=True)
        self.logits = nn.Linear(hidden[-1], num_param * num_module)
        if fused:
            self.target_hiddens = target_hiddens
            h_out = sum(target_hiddens)
            self.scale_header = nn.Linear(hidden[-1], h_out)
            self.split_header = SplitDim(self.target_hiddens)
            with torch.no_grad():
                l = self.scale_header
                nn.init.uniform_(l.weight, -init_std, init_std)
                nn.init.zeros_(l.bias)
        else:
            self.scale_header = nn.ModuleList(
                                            [nn.Linear(hidden[-1],
                                            t)
                                            for t in target_hiddens])
            for l in self.scale_header:
                nn.init.uniform_(l.weight, -init_std, init_std)
                nn.init.zeros_(l.bias)

    def forward(self, x: torch.Tensor):
        z = self.bb(x)
        y = self.logits(z).reshape(*x.shape[:-1],
                                   self.num_param,
                                   self.num_module)
        if self.temperature is not None:
            p = torch.softmax(y / self.temperature, dim=-1)
        else:
            p = torch.softmax(y, dim=-1)
        # ic(p.std(dim=0).mean(dim=-1))
        if self.fused:
            scale = 1 + self.scale_header(z)
            scale = self.split_header(scale)
        else:
            scale = [(1+l(z)) for l in self.scale_header]
        return p, scale


class ActorCriticRange(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        actor_obs,
        critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        n_modules=8,
        emb_dim =128,
        inter_emb_dim=512,
        num_head=4,
        wnet_dims=(512, 512),
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = get_activation(activation)

        a_paramss = []
        c_paramss = []
        print(actor_obs)
        num_actor_obs = sum([emb_dim if 'embed' in k else sum(d)
                   for k,d in actor_obs.items()])
        num_critic_obs = sum([emb_dim if 'embed' in k else sum(d)
                   for k,d in critic_obs.items()])
        for i in range(n_modules):
            params = []
            actor_base = ActorSubnet(num_actor_obs,
                                 num_actions,
                                 actor_hidden_dims,
                                 act_cls=activation)
            func, a_params, bufs = make_functional_with_buffers(actor_base)
            a_params = nn.ParameterList(a_params)
            critic_base = CriticSubnet(num_critic_obs,
                                       1,
                                       critic_hidden_dims,
                                       act_cls=activation)
            func, c_params, bufs = make_functional_with_buffers(critic_base)
            c_params = nn.ParameterList(c_params)
            a_paramss.append(a_params)
            c_paramss.append(c_params)

        stacked_params = [torch.stack(p, dim=0) for p in zip(*a_paramss)]
        self.a_params = nn.ParameterList(stacked_params)
        stacked_params = [torch.stack(p, dim=0) for p in zip(*c_paramss)]
        self.c_params = nn.ParameterList(stacked_params)

        dims_actor = merge_shapes(num_actor_obs,
                            actor_hidden_dims,
                            num_actions)
        dims_value = merge_shapes(num_critic_obs,
                        critic_hidden_dims,
                        1)

        self.func_a = GateModMLP(dims_actor)
        self.func_v = GateModMLP(dims_value)

        self.query_token_mod = nn.Parameter(torch.zeros(1, inter_emb_dim),
                                            requires_grad=True)
        self.mhca_mod = MHAWrapper(inter_emb_dim,
                            num_head,
                            cross_attn = True,
                            use_flash_attn = True)
        
        self.kv_mod = MLP((emb_dim, 
                          (emb_dim + inter_emb_dim)//2,
                          inter_emb_dim),
                        nn.GELU(),
                        True,
                        False,
                        use_ln=True
                    )
        dim_proj = merge_shapes(inter_emb_dim,
                                ((emb_dim + inter_emb_dim)//2,
                                 emb_dim))
        self.proj_mod = MLP(dim_proj,
                        nn.GELU(),
                        use_bn=False,
                        use_ln=True
                        )
        
        self.query_token_sub = nn.Parameter(torch.zeros(1, inter_emb_dim),
                                            requires_grad=True)
        self.mhca_sub = MHAWrapper(inter_emb_dim,
                            num_head,
                            cross_attn = True,
                            use_flash_attn = True)
        
        self.kv_sub = MLP((emb_dim, 
                          (emb_dim + inter_emb_dim)//2,
                          inter_emb_dim),
                        nn.GELU(),
                        True,
                        False,
                        use_ln=True
                    )
        self.proj_sub = MLP(dim_proj,
                        nn.GELU(),
                        use_bn=False,
                        use_ln=True
                        )

        print(f"Actor MLP: {actor_base}")
        print(f"Critic MLP: {critic_base}")

        dim_ctx = [emb_dim if 'embed' in k else sum(d)
                   for k,d in actor_obs.items()]
        scale_target = list(actor_hidden_dims)
        scale_target = [s for s in scale_target for _ in (0, 1)]
        self.actor_modulate = ModulationAndGateNet(sum(dim_ctx),
                                            len(a_params),
                                            n_modules,
                                            tuple(scale_target), 
                                            wnet_dims,
                                            fused=True)
        
        dim_ctx = [emb_dim if 'embed' in k else sum(d)
                   for k,d in critic_obs.items()]
        scale_target = list(critic_hidden_dims)
        scale_target = [s for s in scale_target for _ in (0, 1)]
        self.critic_modulate = ModulationAndGateNet(sum(dim_ctx),
                                            len(c_params),
                                            n_modules,
                                            tuple(scale_target), 
                                            wnet_dims,
                                            fused=True)
        # padding for last layer
        self.register_buffer('last_actor_scale', torch.ones(num_actions))
        self.register_buffer('last_critic_scale', torch.ones(1))

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
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
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    def _actor(self, observations):
        ctx = dict(observations)
        for k, v in ctx.items():
            if 'embed' in k:
                with torch.cuda.amp.autocast(True,
                                            torch.float16):
                    kv = self.kv_mod(v)
                    q = self.query_token_mod[None].broadcast_to(*kv.shape[:-2], 1, -1)
                    ctx[k] = self.proj_mod(self.mhca_mod(
                            q,
                            kv).squeeze(dim=-2))
        flat_ctx = torch.cat([v for v in ctx.values()], dim=-1)
        weights, scales = self.actor_modulate(flat_ctx)
        scales=list(scales)
        weights = weights.unbind(dim=-2)
         
        for _ in range(2):
            scales.append(self.last_actor_scale[None])
        
        ss = []
        for p, s in zip(self.a_params, scales):
            if len(p.shape) == 2:
                # bias
                ss.append(s)
            else:
                # weight
                ss.append(s[..., None])
        
        local_ctx = dict(observations)
        for k, v in local_ctx.items():
            if 'embed' in k:
                with torch.cuda.amp.autocast(True,
                                            torch.float16):
                    kv = self.kv_sub(v)
                    q = self.query_token_sub[None].broadcast_to(*kv.shape[:-2], 1, -1)
                    local_ctx[k] = self.proj_mod(self.mhca_sub(
                            q,
                            kv).squeeze(dim=-2))
        flat_state = torch.cat([v for v in local_ctx.values()], dim=-1)
        act = self.func_a(flat_state,
                        self.a_params[::2],
                        self.a_params[1::2],
                        weights[::2],
                        weights[1::2],
                        ss[::2],
                        ss[1::2]
                        )
        return act  
    
    def _critic(self, critic_observations):
        ctx = dict(critic_observations)
        for k, v in ctx.items():
            if 'embed' in k:
                with torch.cuda.amp.autocast(True,
                                            torch.float16):
                    kv = self.kv_mod(v)
                    q = self.query_token_mod[None].broadcast_to(*kv.shape[:-2], 1, -1)
                    ctx[k] = self.proj_mod(self.mhca_mod(
                            q,
                            kv).squeeze(dim=-2))
        flat_ctx = torch.cat([v for v in ctx.values()], dim=-1)
        weights, scales = self.critic_modulate(flat_ctx)
        scales=list(scales)
        weights = weights.unbind(dim=-2)
         
        for _ in range(2):
            scales.append(self.last_critic_scale[None])
        
        ss = []
        for p, s in zip(self.a_params, scales):
            if len(p.shape) == 2:
                # bias
                ss.append(s)
            else:
                # weight
                ss.append(s[..., None])
        
        local_ctx = dict(critic_observations)
        for k, v in local_ctx.items():
            if 'embed' in k:
                with torch.cuda.amp.autocast(True,
                                            torch.float16):
                    kv = self.kv_sub(v)
                    q = self.query_token_sub[None].broadcast_to(*kv.shape[:-2], 1, -1)
                    local_ctx[k] = self.proj_mod(self.mhca_sub(
                            q,
                            kv).squeeze(dim=-2))
        flat_state = torch.cat([v for v in local_ctx.values()], dim=-1)
        value = self.func_v(flat_state,
                        self.c_params[::2],
                        self.c_params[1::2],
                        weights[::2],
                        weights[1::2],
                        ss[::2],
                        ss[1::2]
                        )
        return value  

    def update_distribution(self, observations):
        mean = self._actor(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self._actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self._critic(critic_observations)
        return value
    

if __name__ == '__main__':
    device ="cuda:0"
    i =  {'a': torch.rand(10, 6, device=device),
          'b': torch.rand(10, 16, device=device),
          'embed': torch.rand(10,16,128, device=device)}
    net = ActorCriticRange(
       actor_obs=i,
       critic_obs=i,
       num_actions=6,
       actor_hidden_dims=[512,128],
       critic_hidden_dims=[512,128]
    ).to(device)
    net.eval()
    print(net)
    print(net.act_inference(i).shape)
    print(net.evaluate(i).shape)