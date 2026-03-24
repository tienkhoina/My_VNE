import torch
from torch.distributions import Categorical

from virne.solver.learning.rl_core.instance_agent import InstanceAgent
from virne.solver.learning.rl_core.rl_solver import PPOSolver

from virne.solver.learning.rl_core.tensor_convertor_flag import (
    obs_as_tensor_for_flag,
)

from virne.solver.learning.utils import apply_mask_to_logit

from virne.solver.learning.reinforcement_learning.flag_env import (
    FlagInstanceEnv,
)

from virne.solver.learning.rl_core.policy_builder import (
    PolicyBuilder,
)

from virne.solver import SolverRegistry


@SolverRegistry.register(
    solver_name="ppo_flag_real",
    solver_type="r_learning",
)
class PPOFlagRealSolver(
    InstanceAgent,
    PPOSolver,
):

    def __init__(
        self,
        controller,
        recorder,
        counter,
        logger,
        config,
        **kwargs,
    ):

        InstanceAgent.__init__(
            self,
            FlagInstanceEnv,
        )

        self.softmax_temp = 1.0

        self.use_bidirectional_action = True

        PPOSolver.__init__(
            self,
            controller,
            recorder,
            counter,
            logger,
            config,
            PolicyBuilder.build_flag_real_policy,
            obs_as_tensor_for_flag,
            **kwargs,
        )

        # =========================
        # FLAG paper required attrs
        # =========================

        self.gamma = config.rl.gamma

        self.gae_lambda = config.rl.gae_lambda

        self.repeat_times = config.rl.repeat_times

        self.eps_clip = config.rl.eps_clip

        self.training_epoch_id = 0

    # ===================================
    # PAPER SELECT ACTION
    # ===================================

    def select_action(
        self,
        observation,
        sample=True,
    ):

        print("\n=== SELECT ACTION ===")

        mask = observation["action_mask"]

        v_size_tensor = observation["v_net_size"]

        v_net_size = int(
            v_size_tensor.item()
        )

        batch_size = observation[
            "curr_v_node_id"
        ].shape[0]

        # -------------------------
        # reshape mask
        # -------------------------

        mask = mask.reshape(
            batch_size,
            v_net_size,
            -1,
        ).permute(
            0,
            2,
            1,
        )

        # =========================
        # HIGH LEVEL
        # =========================

        if self.use_bidirectional_action:

            with torch.no_grad():

                high_logits = self.policy.forward(
                    observation,
                    actor_high=True,
                )

            high_mask = (
                mask.sum(1) != 0
            ).float()

            high_logits = apply_mask_to_logit(
                high_logits,
                high_mask,
            )

            high_dist = Categorical(
                logits=high_logits
                / self.softmax_temp
            )

            if sample:
                high_action = high_dist.sample()
            else:
                high_action = high_logits.argmax(
                    dim=-1
                )

            high_logprob = high_dist.log_prob(
                high_action
            )

        else:

            high_action = observation[
                "curr_v_node_id"
            ]

            high_logprob = torch.zeros_like(
                high_action,
                dtype=torch.float32,
            )

        # =========================
        # LOW LEVEL
        # =========================

        idx = torch.arange(
            mask.shape[0],
            device=mask.device,
        )

        low_mask = mask[
            idx,
            :,
            high_action,
        ]

        with torch.no_grad():

            low_logits = self.policy.forward(
                observation,
                actor_low=True,
                high_level_action=high_action,
            )

        low_logits = apply_mask_to_logit(
            low_logits,
            low_mask,
        )

        low_dist = Categorical(
            logits=low_logits
            / self.softmax_temp
        )

        if sample:
            low_action = low_dist.sample()
        else:
            low_action = low_logits.argmax(
                dim=-1
            )

        low_logprob = low_dist.log_prob(
            low_action
        )

        # =========================
        # decode action
        # =========================

        action = (
            v_size_tensor
            * low_action
            + high_action
        )

        action_logprob = (
            high_logprob
            + low_logprob
        )

        # =========================
        # format
        # =========================

        if torch.numel(action) == 1:

            action = action.item()

        else:

            action = (
                action.reshape(-1)
                .cpu()
                .detach()
                .numpy()
            )

        return (
            action,
            action_logprob
            .cpu()
            .detach()
            .numpy(),
        )