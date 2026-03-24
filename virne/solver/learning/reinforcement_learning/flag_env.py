from virne.solver.learning.rl_core.instance_rl_environment import (
    NodePairStepInstanceRLEnv,
)


class FlagInstanceEnv(NodePairStepInstanceRLEnv):

    def __init__(
        self,
        p_net,
        v_net,
        controller,
        recorder,
        counter,
        logger,
        config,
        **kwargs,
    ):

        print("FLAG ENV INIT")

        # paper dùng ranking
        kwargs["node_ranking_method"] = "nrm"

        super().__init__(
            p_net,
            v_net,
            controller,
            recorder,
            counter,
            logger,
            config,
            **kwargs,
        )

    # =========================
    # paper: next_unplaced_v_node_id
    # =========================

    @property
    def next_unplaced_v_node_id(self):

        placed = len(self.solution["node_slots"])

        if placed >= self.v_net.num_nodes:
            return 0

        # paper dùng ranked_nodes
        return self.v_net.ranked_nodes[placed]

    # =========================
    # observation (paper style)
    # =========================

    def get_observation(self):

        obs = self.feature_constructor.construct(
            self.p_net,
            self.v_net,
            self.solution,
            self.curr_v_node_id,  # dùng cho feature thôi
            self.controller,
        )

        # mask từ NodePair (paper cũng vậy)
        mask = self.generate_action_mask().flatten()

        obs["action_mask"] = mask

        # paper dùng next_unplaced
        obs["curr_v_node_id"] = self.next_unplaced_v_node_id

        obs["v_net_size"] = self.v_net.num_nodes

        print("\nFLAG OBS OK")
        print("p_net", obs["p_net_x"].shape)
        print("v_net", obs["v_net_x"].shape)
        print("mask", obs["action_mask"].shape)
        print("curr", obs["curr_v_node_id"])

        return obs

    # =========================
    # reward (paper)
    # =========================

    def compute_reward(self, solution):

        weight = 1.0 / self.v_net.num_nodes

        if solution["result"]:

            reward = solution["v_net_r2c_ratio"]

        elif solution["place_result"] and solution["route_result"]:

            reward = weight

        else:

            reward = -weight

        self.solution["v_net_reward"] += reward

        return reward