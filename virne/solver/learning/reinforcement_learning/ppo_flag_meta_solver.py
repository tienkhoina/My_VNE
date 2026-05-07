import copy
import torch
from torch.distributions import Categorical
import numpy as np
from virne.solver.learning.utils import (
    apply_mask_to_logit,
)

import torchopt

from virne.solver.learning.rl_core.buffer import RolloutBuffer

from virne.solver.learning.rl_core.instance_agent import InstanceAgent
from virne.solver.learning.rl_core.rl_solver import PPOSolver

from virne.solver.learning.rl_core.tensor_convertor_flag import (
    obs_as_tensor_for_flag,
)

from virne.solver.learning.reinforcement_learning.flag_env import (
    FlagInstanceEnv,
)

from virne.solver.learning.rl_core.policy_builder import (
    PolicyBuilder,
)

from virne.solver import SolverRegistry


@SolverRegistry.register(
    solver_name="ppo_flag_meta",
    solver_type="r_learning",
)
class PPOFlagMetaSolver(
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

        print("\n===== INIT META SOLVER =====")

        InstanceAgent.__init__(
            self,
            FlagInstanceEnv,
        )

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

        print("=== META EXTRA INIT ===")

        print("instance_dict init")
        print("training_task_id_list init")

        self.instance_dict = {}

        self.training_task_id_list = []

        

        

        self.use_curriculum = True

        self.inner_repeat_times = config.rl.repeat_times
        self.outer_repeat_times = 1
        self.init_inner_kl_penalty = 1e-3
        self.target_steps = config.rl.target_steps
        self.weight_decay = config.rl.weight_decay
        self.lr_actor = config.rl.learning_rate.actor

        self.repeat_times = config.rl.repeat_times
        self.repeat_times *= 10

        self.use_bidirectional_action = True
        self.softmax_temp = 1.0

        # --------------------------
        # FLAG meta params
        # --------------------------

        self.use_meta_learning = True
        self.infer_with_meta_policy = False

        self.gamma = config.rl.gamma
        self.gae_lambda = config.rl.gae_lambda
        self.repeat_times = config.rl.repeat_times*10
        self.eps_clip = config.rl.eps_clip

        self.training_epoch_id = 0
        self.num_meta_learning_epochs = kwargs.get('num_meta_learning_epochs', 40)
        self.policy_entropy_threshold = kwargs.get(
            "policy_entropy_threshold",
            2.0,
        )
        self.use_curriculum = True
        

        # --------------------------
        # META POLICY
        # --------------------------

        self.meta_policy = copy.deepcopy(
            self.policy
        ).to(self.device)

        self.meta_optimizer = torch.optim.AdamW(
            self.meta_policy.parameters(),
            lr=config.rl.learning_rate.actor,
            weight_decay=self.weight_decay,
        )

        self.task_policies = {}
        self.task_optimizers = {}

        self.instance_dict = {}

        self.norm_advantage = getattr(
            config.rl,
            "norm_advantage",
            True,
        )

        self.coef_critic_loss = getattr(
            config.rl,
            "coef_critic_loss",
            0.5,
        )

        self.coef_entropy_loss = getattr(
            config.rl,
            "coef_entropy_loss",
            0.0,
        )

        self.max_grad_norm = getattr(
            config.rl,
            "max_grad_norm",
            1.0,
        )

        print("meta_policy:", type(self.meta_policy))
        print("buffer:", type(self.buffer))
        print("policy:", type(self.policy))


    def _stats_task_dist(self, buffer):

        v_sizes = np.array([
            obs["v_net_size"]
            for obs in buffer.observations
        ])

        counter = {}

        for v in v_sizes:
            counter[v] = counter.get(v, 0) + 1

        task_dist = dict(
            sorted(
                counter.items(),
                key=lambda x: x[0],
            )
        )

        return task_dist


    def _stats_instance_dist(self):

        instance_dist = {}

        for k in self.instance_dict:

            instance_dist[k] = len(
                self.instance_dict[k]
            )

        instance_dist = dict(
            sorted(
                instance_dist.items(),
                key=lambda x: x[0],
            )
        )

        return instance_dist

    def select_action(
        self,
        observation,
        sample=True,
    ):
        debug = False

        if debug:
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
        if debug:
            print("high", high_logits.shape)
            print("low", low_logits.shape)
            print("mask", mask.shape)

        # =========================
        # decode action (FIX)
        # =========================

        v_size = v_net_size

        action = (
            v_size
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
    # =========================================
    # override update
    # =========================================

    def _register_instance(self, instance):

        task_id = instance["v_net"].num_nodes

        if task_id not in self.instance_dict:

            self.instance_dict[task_id] = []

        self.instance_dict[task_id].append(instance)

        print("register instance", task_id)
    def learn_with_instance(self, instance):

        v_net = instance["v_net"]
        p_net = instance["p_net"]

        # -------------------------------------------------
        # no meta → normal PPO
        # -------------------------------------------------

        if not self.use_meta_learning:
            return super().learn_with_instance(instance)

        # -------------------------------------------------
        # task id = v_net size
        # -------------------------------------------------

        v_net_size = v_net.num_nodes
        task_id = v_net_size

        # -------------------------------------------------
        # create task policy if needed
        # -------------------------------------------------

        if task_id not in self.task_policies:

            self._init_task_policy_and_task_optimizer(task_id)

        # -------------------------------------------------
        # switch to task policy
        # -------------------------------------------------

        self.policy = self.task_policies[task_id]
        self.optimizer = self.task_optimizers[task_id]

        # -------------------------------------------------
        # register instance (ONLY in meta phase)
        # -------------------------------------------------

        if (
            self.use_meta_learning
            and self.training_epoch_id < self.num_meta_learning_epochs
        ):

            if task_id in self.instance_dict:

                if len(self.instance_dict[task_id]) >= 100:

                    self.instance_dict[task_id].pop(0)

                self.instance_dict[task_id].append(
                    copy.deepcopy(instance)
                )

            else:

                self.instance_dict[task_id] = [
                    copy.deepcopy(instance)
                ]

        # -------------------------------------------------
        # normal learn
        # -------------------------------------------------

        return super().learn_with_instance(instance)
    
    def _init_task_policy(self, task_id):

        self.task_policies[task_id] = \
            copy.deepcopy(self.meta_policy)

        self.task_optimizers[task_id] = \
            torch.optim.AdamW(
                self.task_policies[task_id].parameters(),
                lr=self.meta_optimizer.param_groups[0]["lr"],
                weight_decay=self.weight_decay,
            )
    def _init_task_policy_and_task_optimizer(self, task_id):

        self.task_policies[task_id] = copy.deepcopy(
            self.meta_policy
        )

        self.task_optimizers[task_id] = torch.optim.AdamW(
            self.task_policies[task_id].parameters(),
            lr=self.lr_actor,
            weight_decay=self.weight_decay,
        )

        print(f"New task policy created for {task_id}")
        
    def collect_new_task_buffer(
        self,
        policy,
        instance_set,
        max_num_instances=float("inf"),
        max_num_experiences=float("inf"),
    ):

        # ------------------------------------
        # use given policy
        # ------------------------------------

        self.policy = policy

        task_specific_buffer = RolloutBuffer()

        # ------------------------------------
        # collect data
        # ------------------------------------

        for i, instance in enumerate(instance_set):

            solution, instance_buffer, last_value = \
                super().learn_with_instance(instance)

            # ------------------------------------
            # merge experience
            # ------------------------------------

            self.merge_instance_experience(
                instance,
                solution,
                instance_buffer,
                last_value,
            )

            # ------------------------------------
            # compute returns
            # ------------------------------------

            instance_buffer.compute_returns_and_advantages(
                last_value,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                method=self.compute_advantage_method,
            )

            # ------------------------------------
            # merge buffer
            # ------------------------------------

            task_specific_buffer.merge(
                instance_buffer
            )

            # ------------------------------------
            # stop condition
            # ------------------------------------

            if (
                task_specific_buffer.size()
                >= max_num_experiences
                or i >= max_num_instances
            ):
                break

        return task_specific_buffer
    

    def _calculate_ppo_loss(
        self,
        observations,
        actions,
        old_action_logprobs,
        returns,
        clip_loss=False,
        use_ppo=True,
    ):

        values, action_logprobs, dist_entropy = \
            self.evaluate_actions_flag(
                observations,
                actions,
            )

        advantages = returns - values.detach()

        if self.norm_advantage and values.numel() != 0:

            advantages = (
                advantages - advantages.mean()
            ) / (advantages.std() + 1e-9)

        if use_ppo:

            ratio = torch.exp(
                action_logprobs
                - old_action_logprobs
            )

            surr1 = ratio * advantages

            surr2 = torch.clamp(
                ratio,
                1.0 - self.eps_clip,
                1.0 + self.eps_clip,
            ) * advantages

            actor_loss = (
                -torch.min(surr1, surr2).mean()
                if clip_loss
                else -(surr1).mean()
            )

            kl_div = (
                (torch.exp(ratio) - 1)
                - ratio
            ).mean()

        else:

            actor_loss = -(
                action_logprobs
                * advantages
            ).mean()

            kl_div = torch.zeros(1).to(
                self.device
            ).mean()

        critic_loss = self.criterion_critic(
            returns,
            values,
        )

        entropy_loss = dist_entropy.mean()

        loss = (
            actor_loss
            + self.coef_critic_loss
            * critic_loss
            - self.coef_entropy_loss
            * entropy_loss
        )

        return loss, (
            actor_loss,
            critic_loss,
            entropy_loss,
            values,
            action_logprobs,
            advantages,
            kl_div,
        )
    
    def _preprocess_buffer(
        self,
        buffer,
    ):

        if (
            not hasattr(buffer, "returns")
            or len(buffer.returns) == 0
        ):

            buffer.compute_returns_and_advantages(
                0,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
            )

        obs = self.preprocess_obs(
            buffer.observations,
            self.device,
        )

        actions = torch.LongTensor(
            np.array(buffer.actions)
        ).to(self.device)

        old_logprobs = torch.FloatTensor(
            np.concatenate(
                buffer.logprobs,
                axis=0,
            )
        ).to(self.device)

        rewards = torch.FloatTensor(
            buffer.rewards
        ).to(self.device)

        returns = torch.FloatTensor(
            buffer.returns
        ).to(self.device)

        if returns.numel() > 1:

            returns = (
                returns - returns.mean()
            ) / (returns.std() + 1e-9)

        return (
            obs,
            actions,
            old_logprobs,
            rewards,
            returns,
        )

    def update(self):

        if not self.use_meta_learning:
            return super().update()

        if self.training_epoch_id < self.num_meta_learning_epochs:

            return self._meta_learning_update()

        else:

            return self._fine_tuning_update()
        
    def _fine_tuning_update(self):

        meta_buffer = self.buffer

        # -------------------------
        # task distribution
        # -------------------------

        task_dist = self._stats_task_dist(
            self.buffer
        )

        # -------------------------
        # init missing task policy
        # -------------------------

        for task_id in task_dist:

            if task_id not in self.task_policies:

                self.task_policies[task_id] = \
                    copy.deepcopy(
                        self.meta_policy
                    )

                self.task_optimizers[task_id] = \
                    torch.optim.AdamW(
                        self.task_policies[
                            task_id
                        ].parameters(),
                        lr=self.lr_actor,
                        weight_decay=self.weight_decay,
                    )

        # -------------------------
        # split buffer
        # -------------------------

        task_buffers = self._split_buffer(
            self.buffer
        )

        # -------------------------
        # update per task
        # -------------------------

        for task_id in task_buffers:

            self.policy = self.task_policies[
                task_id
            ]

            self.optimizer = \
                self.task_optimizers[
                    task_id
                ]

            self.buffer = task_buffers[
                task_id
            ]

            super().update()

        # -------------------------
        # clear buffer
        # -------------------------

        meta_buffer.clear()

    # =========================================
    # meta learning update (debug)
    # =========================================

    def evaluate_actions_flag(
        self,
        obs,
        actions,
    ):

        mask = obs["action_mask"]

        v_size = int(
            obs["v_net_size"][0].item()
        )

        batch_size = obs[
            "curr_v_node_id"
        ].shape[0]

        mask = mask.reshape(
            batch_size,
            v_size,
            -1,
        ).permute(
            0,
            2,
            1,
        )

        # ------------------------
        # decode action
        # ------------------------

        high_action = actions % v_size
        low_action = actions // v_size

        # ========================
        # HIGH
        # ========================

        high_logits = self.policy.forward(
            obs,
            actor_high=True,
        )

        high_mask = mask.sum(1) != 0

        high_logits = apply_mask_to_logit(
            high_logits,
            high_mask,
        )

        high_dist = Categorical(
            logits=high_logits / self.softmax_temp
        )

        high_logprob = high_dist.log_prob(
            high_action
        )

        # ========================
        # LOW
        # ========================

        idx = torch.arange(
            mask.shape[0],
            device=mask.device,
        )

        low_mask = mask[
            idx,
            :,
            high_action,
        ]

        low_logits = self.policy.forward(
            obs,
            actor_low=True,
            high_level_action=high_action,
        )

        low_logits = apply_mask_to_logit(
            low_logits,
            low_mask,
        )

        low_dist = Categorical(
            logits=low_logits / self.softmax_temp
        )

        low_logprob = low_dist.log_prob(
            low_action
        )

        # ========================
        # value
        # ========================

        values = self.policy.evaluate(obs)

        if values.dim() > 1:
            values = values.squeeze(-1)

        logprob = high_logprob + low_logprob

        entropy = (
            high_dist.entropy()
            + low_dist.entropy()
        )

        return values, logprob, entropy

    def _meta_learning_update(self):

        import torchopt

        self.policy = self.meta_policy

        meta_buffer = self.buffer

        task_buffers = self._split_buffer(
            meta_buffer
        )

        task_dist = self._stats_task_dist(
            meta_buffer
        )

        instance_dist = self._stats_instance_dist()

        print(
            "Experience distribution:",
            task_dist,
        )

        print(
            "Instance distribution:",
            instance_dist,
        )

        if self.use_curriculum:

            if len(
                self.training_task_id_list
            ) == 0:

                self.training_task_id_list.append(
                    min(task_dist.keys())
                )

                print(
                    "Add task",
                    min(task_dist.keys()),
                )

            training_tasks_list = (
                self.training_task_id_list
            )

        else:

            training_tasks_list = list(
                task_buffers.keys()
            )

        num_tasks = len(task_buffers)

        self.outer_repeat_times = 1

        for _ in range(
            self.outer_repeat_times
        ):

            total_meta_loss = 0
            total_ppo_loss = 0
            total_kl_loss = 0

            task_policy_entropy_dict = {}

            inner_opt = torchopt.MetaSGD(
                self.meta_policy,
                lr=self.lr_actor,
                weight_decay=self.weight_decay,
            )

            policy_state_dict = \
                torchopt.extract_state_dict(
                    self.meta_policy
                )

            optim_state_dict = \
                torchopt.extract_state_dict(
                    inner_opt
                )

            for task_id in training_tasks_list:

                if task_id not in task_buffers:
                    continue

                if task_id not in self.instance_dict:
                    continue

                buffer = task_buffers[task_id]

                inner_repeat_times = int(
                    (num_tasks
                    * self.repeat_times)
                    / len(
                        training_tasks_list
                    )
                )

                inner_repeat_times = 10

                for step in range(
                    inner_repeat_times
                ):

                    buffer.split_with_instance()

                    (
                        observations,
                        actions,
                        old_action_logprobs,
                        rewards,
                        returns,
                    ) = self._preprocess_buffer(
                        buffer
                    )

                    loss, (
                        actor_loss,
                        critic_loss,
                        entropy_loss,
                        values,
                        action_logprobs,
                        advantages,
                        kl_div,
                    ) = self._calculate_ppo_loss(
                        observations,
                        actions,
                        old_action_logprobs,
                        returns,
                        clip_loss=True,
                    )

                    inner_opt.step(loss)

                if task_id not in self.instance_dict:
                    continue

                task_specific_buffer = \
                    self.collect_new_task_buffer(
                        self.meta_policy,
                        self.instance_dict[
                            task_id
                        ],
                        max_num_experiences=64,
                    )

                (
                    observations,
                    actions,
                    old_action_logprobs,
                    rewards,
                    returns,
                ) = self._preprocess_buffer(
                    task_specific_buffer
                )

                ppo_loss, (
                    actor_loss,
                    critic_loss,
                    entropy_loss,
                    values,
                    action_logprobs,
                    advantages,
                    kl_div,
                ) = self._calculate_ppo_loss(
                    observations,
                    actions,
                    old_action_logprobs,
                    returns,
                    clip_loss=True,
                    use_ppo=False,
                )

                kl_loss = torch.zeros(
                    1
                ).to(
                    self.device
                ).mean()

                meta_loss = (
                    ppo_loss
                    + kl_loss
                )

                meta_loss.backward()

                torchopt.recover_state_dict(
                    self.meta_policy,
                    policy_state_dict,
                )

                torchopt.recover_state_dict(
                    inner_opt,
                    optim_state_dict,
                )

                total_meta_loss += \
                    meta_loss.detach().cpu().numpy()

                total_ppo_loss += \
                    ppo_loss.detach().cpu().numpy()

                total_kl_loss += \
                    kl_loss.detach().cpu().numpy()

                task_policy_entropy_dict[
                    task_id
                ] = entropy_loss.detach(
                ).cpu().numpy()

            torch.nn.utils.clip_grad_norm_(
                self.meta_policy.parameters(),
                self.max_grad_norm,
            )

            self.meta_optimizer.step()

            self.meta_optimizer.zero_grad()

            if self.use_curriculum:

                most_complex = \
                    self.training_task_id_list[-1]

                if most_complex < max(task_dist.keys()):

                    if most_complex not in task_policy_entropy_dict:
                        continue

                    entropy = \
                        task_policy_entropy_dict[
                            most_complex
                        ]

                    if entropy < \
                    self.policy_entropy_threshold:

                        self.training_task_id_list.append(
                            most_complex + 1
                        )

                        print(
                            "Add task",
                            most_complex + 1,
                        )
        for task_id in task_buffers:

            self.task_policies[
                task_id
            ].load_state_dict(
                self.meta_policy.state_dict()
            )

        meta_buffer.clear()

        self.buffer = meta_buffer

        self.instance_dict = {}

        return None

    # =========================================
    # split buffer (same as paper)
    # =========================================

    def _split_buffer(
        self,
        buffer,
    ):

        task_buffers = {}

        v_sizes = np.array([
            obs["v_net_size"]
            for obs in buffer.observations
        ])

        tasks = sorted(
            list(set(v_sizes))
        )

        for task_id in tasks:

            idx = np.where(
                v_sizes == task_id
            )[0]

            tb = type(buffer)()

            tb.observations = [
                buffer.observations[i]
                for i in idx
            ]

            tb.actions = list(
                np.array(buffer.actions)[idx]
            )

            tb.logprobs = list(
                np.array(buffer.logprobs)[idx]
            )

            tb.rewards = list(
                np.array(buffer.rewards)[idx]
            )

            tb.returns = list(
                np.array(buffer.returns)[idx]
            )

            tb.values = list(
                np.array(buffer.values)[idx]
            )

            tb.dones = list(
                np.array(buffer.dones)[idx]
            )

            task_buffers[task_id] = tb

        return task_buffers

    def solve(self, instance):

        v_net = instance["v_net"]
        p_net = instance["p_net"]

        v_net_size = v_net.num_nodes

        # ------------------------------------
        # choose policy (FLAG logic)
        # ------------------------------------

        if self.use_meta_learning:

            if (
                self.infer_with_meta_policy
                or v_net_size not in self.task_policies
            ):

                self.policy = self.meta_policy

            else:

                self.policy = self.task_policies[v_net_size]

        else:

            self.policy = self.policy

        # ------------------------------------
        # create env
        # ------------------------------------

        env = self.InstanceEnv(
            p_net,
            v_net,
            self.controller,
            self.recorder,
            self.counter,
            self.logger,
            self.config,
        )

        obs = env.get_observation()


        done = False

        # ------------------------------------
        # rollout
        # ------------------------------------

        while not done:
            # print("curr_v_node_id =", obs["curr_v_node_id"])

            tensor_obs = self.preprocess_obs(
                obs,
                device=self.device,
            )

            action, _ = self.select_action(
                tensor_obs,
                sample=False,
            )
            # print("action v_node_id =", action%v_net_size)

            obs, reward, done, info = env.step(
                action
            )

            if done:

                return env.solution

        raise RuntimeError("solve failed")