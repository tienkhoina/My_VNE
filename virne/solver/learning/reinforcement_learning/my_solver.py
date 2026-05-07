from virne.solver.learning.rl_core.instance_agent import InstanceAgent
from virne.solver.learning.rl_core.rl_solver import PPOSolver
from virne.solver.learning.rl_core.policy_builder import PolicyBuilder
from virne.solver import SolverRegistry

from virne.solver.learning.reinforcement_learning.my_env import MyInstanceEnv
from virne.solver.learning.rl_core.tensor_convertor import TensorConvertor

from virne.solver.learning.utils import apply_mask_to_logit
from virne.solver.debug_obs import debug_obs

import torch
from torch.distributions import Categorical
@SolverRegistry.register(solver_name="my_dual_solver", solver_type="r_learning")
class MyPPOSolver(InstanceAgent, PPOSolver):

    def __init__(self, controller, recorder, counter, logger, config, **kwargs):

        # ===== InstanceAgent (env) =====
        InstanceAgent.__init__(
            self,
            MyInstanceEnv
        )

        # ===== PPO Solver =====
        PPOSolver.__init__(
            self,
            controller=controller,
            recorder=recorder,
            counter=counter,
            logger=logger,
            config=config,
            make_policy=PolicyBuilder.build_my_dual_policy,  # 🔥 model của bạn
            obs_as_tensor=TensorConvertor.obs_as_tensor_myself,
            **kwargs
        )

        # ===== custom flag =====
        self.is_hierarchical = True


    def select_action(self, observation, sample=True):

        with torch.no_grad():

            # =========================
            # 🔥 MASK (V, P)
            # =========================
            mask = observation["action_mask"]   # (V, P)
            mask = mask.bool()

            V, P = mask.shape
            device = mask.device

            # =========================
            # 🔥 HIGH: chọn v
            # =========================
            # policy trả (1, V)
            high_logits = self.policy(observation, actor_high=True)

            if high_logits.dim() == 2:
                high_logits = high_logits.squeeze(0)   # (V,)

            # v hợp lệ nếu có ít nhất 1 p hợp lệ
            mask_v = (mask.sum(dim=1) > 0)   # (V,)

            if mask_v.sum() == 0:
                mask_v[0] = True

            high_logits = apply_mask_to_logit(high_logits, mask_v)

            dist_high = Categorical(logits=high_logits / self.softmax_temp)

            if sample:
                v = dist_high.sample()   # scalar
            else:
                v = high_logits.argmax(dim=-1)

            log_prob_v = dist_high.log_prob(v)

            # 🔥 FIX: đảm bảo v có batch dim cho policy
            if v.dim() == 0:
                v_batch = v.unsqueeze(0)   # (1,)
            else:
                v_batch = v

            # =========================
            # 🔥 LOW: chọn p | v
            # =========================
            # policy trả (1, P)
            low_logits = self.policy(
                observation,
                actor_low=True,
                high_level_action=v_batch
            )

            if low_logits.dim() == 2:
                low_logits = low_logits.squeeze(0)   # (P,)

            # mask theo v đã chọn
            v_idx = v.item() if v.dim() == 0 else v_batch.item()
            mask_p = mask[v_idx]   # (P,)

            if mask_p.sum() == 0:
                mask_p[0] = True

            low_logits = apply_mask_to_logit(low_logits, mask_p)

            dist_low = Categorical(logits=low_logits / self.softmax_temp)

            if sample:
                p = dist_low.sample()
            else:
                p = low_logits.argmax(dim=-1)

            log_prob_p = dist_low.log_prob(p)

            # =========================
            # 🔥 COMBINE ACTION
            # =========================
            p_val = p.item()
            v_val = v.item()

            action = p_val * V + v_val          # int
            log_prob = (log_prob_v + log_prob_p).item()

            import numpy as np
            log_prob = np.array([log_prob])     # chỉ wrap log_prob

            return action, log_prob
    
    

    def evaluate_actions(self, old_observations, old_actions, return_others=False):

        mask = old_observations["action_mask"]

        if isinstance(mask, list):
            B = len(mask)
            V = max(m.shape[0] for m in mask)
            P = mask[0].shape[1]

            padded = torch.zeros((B, V, P), device=mask[0].device)

            for i, m in enumerate(mask):
                padded[i, :m.shape[0]] = m

            mask = padded

        mask = mask.bool()
        B, V, P = mask.shape
        device = mask.device

        # =========================
        # 🔥 DECODE ACTION
        # =========================
        real_V = old_observations["v_net_size"].long()   # (B,)

        v = old_actions % real_V
        p = old_actions // real_V
        # =========================
        # 🔥 HIGH (π(v))
        # =========================
        high_logits = self.policy(old_observations, actor_high=True)  # (B, V)

        mask_v = (mask.sum(dim=2) > 0)  # (B, V)

        invalid_batch = (mask_v.sum(dim=1) == 0)
        if invalid_batch.any():
            mask_v[invalid_batch, 0] = True

        high_logits = apply_mask_to_logit(high_logits, mask_v)

        dist_high = Categorical(logits=high_logits / self.softmax_temp)

        log_prob_v = dist_high.log_prob(v)
        entropy_v = dist_high.entropy()

        # =========================
        # 🔥 LOW (π(p|v))
        # =========================
        low_logits = self.policy(
            old_observations,
            actor_low=True,
            high_level_action=v
        )  # (B, P)

        idx = torch.arange(B, device=device)
        mask_p = mask[idx, v]   # (B, P)

        invalid_batch = (mask_p.sum(dim=1) == 0)
        if invalid_batch.any():
            mask_p[invalid_batch, 0] = True

        low_logits = apply_mask_to_logit(low_logits, mask_p)

        dist_low = Categorical(logits=low_logits / self.softmax_temp)

        log_prob_p = dist_low.log_prob(p)
        entropy_p = dist_low.entropy()

        # =========================
        # 🔥 COMBINE
        # =========================
        action_logprobs = log_prob_v + log_prob_p
        dist_entropy = entropy_v + entropy_p

        # =========================
        # 🔥 VALUE
        # =========================
        values = self.policy.evaluate(old_observations).squeeze(-1)

        # print("action:", old_actions[:10])
        # print("mask valid count:", mask.sum(dim=-1)[:10])
        # =========================
        # 🔥 CHECK ACTION VALIDITY
        # =========================
        idx = torch.arange(B, device=device)

        valid_p = mask[idx, v, p]   # 🔥 KEY LINE

        # print("v:", v[:10])
        # print("p:", p[:10])
        # print("valid_p:", valid_p[:10])

        if return_others:
            return values, action_logprobs, dist_entropy, {}

        return values, action_logprobs, dist_entropy
    
    def solve(self, instance):

        v_net, p_net = instance['v_net'], instance['p_net']

        instance_env = self.InstanceEnv(
            p_net,
            v_net,
            self.controller,
            self.recorder,
            self.counter,
            self.logger,
            self.config
        )

        obs = instance_env.get_observation()
        done = False

       

        while not done:
            tensor_obs = self.preprocess_obs(obs, device=self.device)

            action, _ = self.select_action(tensor_obs, sample=False)
            # print(f"Selected action: {action}")
            # debug_obs("Selected Action",obs,p_net,v_net)

            obs, reward, done, info = instance_env.step(action)

            if done:
                return instance_env.solution