#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations
from collections.abc import Sequence

import torch
import copy

from rsl_rl.utils import split_and_pad_trajectories

from icecream import ic

def _copy(t, s, i):
    if isinstance(t, dict):
        for k,v in s.items():
            if isinstance(i, int):
                t[k][i].copy_(v)
            elif isinstance(i, torch.Tensor):
                t[k].index_copy_(0, i, v)
    else:
        if isinstance(i, int):
            t[i].copy_(s)
        elif isinstance(i, torch.Tensor):
            t.index_copy_(0, i, s)

def _copy2(t, s, ti, ei):
    if isinstance(t, dict):
        for k,v in s.items():
            t[k].index_put_((ti, ei), s[ei])
    else:
        t.index_put_((ti, ei), s[ei])

def _flatten(s):
    if isinstance(s, dict):
        out = dict(s)
        for k,v in out.items():
            out[k] = v.flatten(0, 1)
        return out
    else:
        return s.flatten(0, 1)
    
def _get(s, i):
    if isinstance(s, dict):
        out = {}
        for k,v in s.items():
            out[k] = v[i]
        return out
    else:
        return s[i]
    

class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None
            self.action_logit = None
            self.update_indices = None

        def clear(self):
            self.__init__()

    class Local_Transition:
        def __init__(self, num_envs: int,
                     obs_shape: int|dict[str,int],
                     actions_shape: int,
                     logit_dim: Sequence[int]|None = None,
                     device: str|torch.device = 'cuda:0'
                     ):
            if isinstance(obs_shape, dict):
                self.observations = {k: torch.zeros(
                                                num_envs,
                                                *v,
                                                device=device)
                                                for k,v in obs_shape.items()}
            else:
                self.observations = torch.zeros(num_envs, *obs_shape, device=device)
            # FIXME no state input for critic for now
            self.critic_observations = None
            self.rewards = torch.zeros(num_envs, device=device)
            # self.dones = torch.zeros(num_envs, 1, device=device).byte()
            self.values = torch.zeros(num_envs, 1, device=device)
            self.actions_log_prob = torch.zeros(num_envs, device=device)
            if logit_dim is None:
                self._is_discrete = False
                self.action_mean = torch.zeros(num_envs, 
                                       actions_shape, device=device)
                self.action_sigma = torch.zeros(num_envs, 
                                       actions_shape, device=device)
            else:
                self._is_discrete = True
                logit_max = max(list(logit_dim))
                self.action_logit = torch.zeros(num_envs, actions_shape,
                                               logit_max, device=device)
            dtype = torch.long if self._is_discrete else torch.float
            self.actions = torch.zeros(num_envs, 
                                       actions_shape,
                                       dtype=dtype, 
                                       device=device)

            # FIXME no support for RNN
            self.hidden_states = None
            # buffer is initialized (start recording)
            # when the first action_mask is set
            self.activated = torch.zeros(num_envs, dtype=torch.bool, device=device)

        def add(self, transition: RolloutStorage.Transition):
            activated = torch.nonzero(transition.update_indices).ravel()
            done = (transition.dones>0)

            to_be_flushed = transition.update_indices | done
            to_be_flushed &= self.activated
            # accumulate reward for termination
            termination_update = (done & self.activated)
            if termination_update.any():
                self.rewards[termination_update] += transition.rewards[termination_update]
            
            # for attr_name, attr_value in transition.__dict__.items():
            #     if isinstance(attr_value, torch.Tensor):
            #         print(f"Attribute '{attr_name}' is a tensor with shape {attr_value.shape}")

            if to_be_flushed.any():
                # get copy of transition if we have to return
                transition_buff = copy.deepcopy(transition)
                transition_buff.rewards[to_be_flushed] = self.rewards[to_be_flushed]
                transition_buff.actions[to_be_flushed] = self.actions[to_be_flushed]
                transition_buff.values[to_be_flushed] = self.values[to_be_flushed]
                transition_buff.actions_log_prob[to_be_flushed] = self.actions_log_prob[to_be_flushed]
                # once flushed reward have to be 0
                self.rewards[to_be_flushed] = 0.0
                
                # _copy(transition_buff.observations, self.observations[to_be_flushed],
                #       to_be_flushed)
                transition_buff.observations[to_be_flushed] = self.observations[to_be_flushed]
                if self._is_discrete:
                    transition_buff.action_logit[to_be_flushed] = self.action_logit[to_be_flushed]
                else:
                    transition_buff.action_mean[to_be_flushed] = self.action_mean[to_be_flushed]
                    transition_buff.action_sigma[to_be_flushed] = self.action_sigma[to_be_flushed]
            else:
                transition_buff = transition

            self.activated[activated] = 1
            # terminated environment is not activate until next action come
            self.activated[done] = 0

            # initialize the buf with current transition
            if self.activated.any():
                self.rewards[self.activated] += transition.rewards[self.activated]
                # _copy(self.observations, transition.observations[activated], activated)
                self.observations[self.activated] = transition.observations[self.activated]
                self.values[self.activated] = transition.values[self.activated]
                self.actions[self.activated] = transition.actions[self.activated]
                self.actions_log_prob[self.activated] = transition.actions_log_prob[self.activated]
                if self._is_discrete:
                    self.action_logit[self.activated] = transition_buff.action_logit[self.activated]
                else:
                    self.action_mean[self.activated] = transition_buff.action_mean[self.activated]
                    self.action_sigma[self.activated] = transition_buff.action_sigma[self.activated]

            transition_buff.update_indices = to_be_flushed
            return transition_buff
            

        def clear(self):
            self.activated[:] = 0
            self.rewards[:] = 0

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, actions_shape,
                 is_discrete,
                 device="cpu"):
        self.device = device

        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self._is_discrete = is_discrete
        self.actions_shape = actions_shape

        self._is_debug = False
        ic(actions_shape)
        self._local_transition = self.Local_Transition(num_envs,
                                                       self.obs_shape,
                                                       len(actions_shape) if is_discrete else actions_shape,
                                                       actions_shape if is_discrete else None,
                                                       self.device
                                                       )

        # Core
        if isinstance(obs_shape, dict):
            self.observations = {k: torch.zeros(num_transitions_per_env,
                                            num_envs,
                                            *v,
                                            device=self.device)
                                            for k,v in obs_shape.items()}
        else:
            self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        if ((isinstance(privileged_obs_shape, Sequence)
             and privileged_obs_shape[0] is not None) 
            or (isinstance(privileged_obs_shape, dict)
                and next(iter(privileged_obs_shape)) is not None)):
            if isinstance(privileged_obs_shape, dict):
                self.privileged_observations= {k: torch.zeros(num_transitions_per_env,
                                            num_envs,
                                            *v,
                                            device=self.device)
                                            for k,v in privileged_obs_shape.items()}
            else:
                self.privileged_observations = torch.zeros(
                    num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device
                )
        else:
            self.privileged_observations = None
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        if is_discrete:
            self.actions = torch.zeros(num_transitions_per_env, num_envs, 
                                       len(actions_shape), device=self.device)
        else:
            self.actions = torch.zeros(num_transitions_per_env, num_envs, 
                                       actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        if is_discrete:
            if len(actions_shape) ==1:
                self.logit = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            else:
                max_len = max(list(actions_shape))
                actions_shape = (len(actions_shape), max_len)
                self.logit = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        else:
            self.mu = torch.zeros(num_transitions_per_env, num_envs, actions_shape, device=self.device)
            self.sigma = torch.zeros(num_transitions_per_env, num_envs, actions_shape, device=self.device)
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # rnn
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        self.step = torch.zeros(num_envs, dtype=torch.long,
                                device=self.device)
        

    def add_transitions_v2(self, transition: Transition):

        if transition.update_indices is None:
            s = self.step
            ei = torch.arange(self.num_envs, device=self.device)
        else:
            #First add to the local buffer
            transition = self._local_transition.add(transition)
            # everything is updated based on the output of the local buffer
            s = self.step[transition.update_indices]
            ei = transition.update_indices
        _copy2(self.observations, transition.observations, s, ei)
        if self.privileged_observations is not None:
            _copy2(self.privileged_observations,
                  transition.critic_observations, s, ei)
        _copy2(self.actions, transition.actions.float(), s, ei)
        _copy2(self.rewards, transition.rewards.view(-1, 1), s, ei)
        _copy2(self.dones, transition.dones.view(-1, 1).byte(), s, ei)
        _copy2(self.values, transition.values, s, ei)
        _copy2(self.actions_log_prob, transition.actions_log_prob.view(-1, 1), s, ei)
        if self._is_discrete:
            _copy2(self.logit, transition.action_logit, s, ei)
        else:
            _copy2(self.mu, transition.action_mean, s, ei)
            _copy2(self.sigma, transition.action_sigma, s, ei)
        
        if self._is_debug:
            ic(s, ei, self.step)
            # ic(self.observations[s, ei], transition.observations,
            #    transition.observations[ei])
            if False:
                ic(self.actions[s, ei], transition.actions,
                transition.actions[ei])
                ic(self.rewards[s, ei], transition.rewards,
                transition.rewards[ei])
                ic(self.dones[s, ei], transition.dones,
                transition.dones[ei])
                ic(self.values[s, ei], transition.values,
                transition.values[ei])

        self._save_hidden_states(transition.hidden_states)
        if transition.update_indices is None:
            self.step += 1
        else:
            # ic(transition.update_indices, self.step)
            self.step[transition.update_indices] += 1
            # ic(self.step)

    def add_transitions_v1(self, transition: Transition):
        _copy(self.observations, transition.observations, self.step)
        if self.privileged_observations is not None:
            _copy(self.privileged_observations,
                  transition.critic_observations, self.step)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        if self._is_discrete:
            self.logit[self.step].copy_(transition.action_logit)
        else:
            self.mu[self.step].copy_(transition.action_mean)
            self.sigma[self.step].copy_(transition.action_sigma)
        self._save_hidden_states(transition.hidden_states)
        self.step += 1

    def add_transitions(self, transition: Transition):
        if (self.step >= self.num_transitions_per_env).any():
            raise AssertionError("Rollout buffer overflow")
        self.add_transitions_v2(transition)

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states == (None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)

        # initialize if needed
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [
                torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))
            ]
            self.saved_hidden_states_c = [
                torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))
            ]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self):
        self.step[:] = 0
        self.values[:] = 0
        self.returns[:] = 0
        self.rewards[:] = 0
        self.dones[:] = 0
        self._local_transition.clear()

    def compute_returns(self, last_values, gamma, lam):
        advantage = torch.zeros(self.num_envs, 1, device=self.device)
        end_ = self.step.amax().item()
        ic(self.step, end_)
        ic(self.values[end_], self.values[self.step.amin().item()-1])
        for step in reversed(range(self.step.amin().item()-1)):
            if False:
                use_next_val = self.step> step+1
                use_last_val = self.step == step +1
                valid = torch.logical_or(use_last_val,
                                        use_next_val)
                next_values = self.values[step].clone()
                # ic(step, valid)
                if len(use_next_val)>0:
                    next_values[use_next_val] = self.values[step + 1, use_next_val]
                    # ic(step, use_next_val)
                if len(use_last_val)>0:
                    # ic(step, use_last_val)
                    next_values[use_last_val] = last_values[use_last_val]
                next_is_not_terminal = 1.0 - self.dones[step, valid].float()
                delta = self.rewards[step, valid] + next_is_not_terminal * gamma * next_values[valid] - self.values[step, valid]
                advantage[valid] = delta + next_is_not_terminal * gamma * lam * advantage[valid]
                # ic(step, advantage[valid].shape)
                self.returns[step, valid] = advantage[valid] + self.values[step, valid]
            else:
                next_values = self.values[step + 1]
                next_is_not_terminal = 1.0 - self.dones[step].float()
                delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
                advantage = delta + next_is_not_terminal * gamma * lam * advantage
                self.returns[step] = advantage + self.values[step]
        self.advantages = self.returns - self.values
        mask = torch.arange(self.num_transitions_per_env, device=self.device)[..., None]
        if False:
            mask = (mask<self.step).float()[..., None]
        else:
            l = torch.full_like(self.step, self.step.amin().item()-1)
            mask = (mask<l).float()[..., None]
        # for iii, sss in enumerate(self.step):
        #     ic(sss,self.values[:end_+1, iii].ravel(),
        #        self.returns[:end_+1, iii].ravel(),
        #        self.rewards[:end_+1, iii].ravel(),
        #        self.dones[:end_+1, iii].ravel(),
        #        mask[:, iii].ravel()
        #        )
        sum = (self.advantages * mask).sum()
        count = mask.sum()
        mean = sum / count
        var = ((self.advantages - mean) ** 2 * mask).sum() /(count-1)
        std = var.sqrt()
        if self._is_debug:
            before_norm = self.advantages.clone()
        self.advantages = (self.advantages - mean) / (std + 1e-8)

        if self._is_debug:
            adv_dbg = 0
            return_dbg = self.returns.clone()
            for step in reversed(range(end_)):
                if step == self.num_transitions_per_env - 1:
                    next_values = last_values
                else:
                    next_values = self.values[step + 1]
                next_is_not_terminal = 1.0 - self.dones[step].float()
                delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
                adv_dbg = delta + next_is_not_terminal * gamma * lam * adv_dbg
                return_dbg[step] = adv_dbg + self.values[step]

            # Compute and normalize the advantages
            adv_dbg_t = return_dbg - self.values
            ic(before_norm[:end_], adv_dbg_t[:end_])
            ic(count, sum, mean, std)
            ic(adv_dbg_t[:end_].sum(), adv_dbg_t[:end_].mean(), adv_dbg_t[:end_].std())
            adv_dbg_t = (adv_dbg_t - adv_dbg_t[:end_].mean()) / (adv_dbg_t[:end_].std() + 1e-8)
            ic(self.returns[:end_], return_dbg[:end_])
            ic(self.advantages[:end_], adv_dbg_t[:end_])
            
    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat(
            (flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0])
        )
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        end_ = self.step.amin().item()-1
        ic(self.step.amin().item(), end_)
        batch_size = self.num_envs * end_
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = _flatten(self.observations)
        if self.privileged_observations is not None:
            critic_observations = _flatten(self.privileged_observations)
        else:
            critic_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        if self._is_discrete:
            old_logit = self.logit.flatten(0, 1)
        else:
            old_mu = self.mu.flatten(0, 1)
            old_sigma = self.sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                # obs_batch = observations[batch_idx]
                obs_batch = _get(observations, batch_idx)
                critic_observations_batch = _get(critic_observations, batch_idx)
                # critic_observations_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx] if not self._is_discrete else None
                old_sigma_batch = old_sigma[batch_idx] if not self._is_discrete else None
                old_logit_batch = (old_logit[batch_idx] if self._is_discrete else None)
                yield obs_batch, critic_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    None,
                    None,
                ), None, old_logit_batch

    # for RNNs only
    def reccurent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
        if self.privileged_observations is not None:
            padded_critic_obs_trajectories, _ = split_and_pad_trajectories(self.privileged_observations, self.dones)
        else:
            padded_critic_obs_trajectories = padded_obs_trajectories

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                critic_obs_batch = padded_critic_obs_trajectories[:, first_traj:last_traj]

                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop] if not self._is_discrete else None
                old_sigma_batch = self.sigma[:, start:stop] if not self._is_discrete else None
                old_logit_batch = (self.logit[:, start:stop] if self._is_discrete else None)
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
                # then take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                last_was_done = last_was_done.permute(1, 0)
                hid_a_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_a
                ]
                hid_c_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_c
                ]
                # remove the tuple for GRU
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else hid_a_batch
                hid_c_batch = hid_c_batch[0] if len(hid_c_batch) == 1 else hid_c_batch

                yield obs_batch, critic_obs_batch, actions_batch, values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    hid_a_batch,
                    hid_c_batch,
                ), masks_batch, old_logit_batch

                first_traj = last_traj
