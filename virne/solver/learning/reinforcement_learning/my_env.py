from virne.solver.learning.rl_core.instance_rl_environment import (
    NodePairStepInstanceRLEnv,)

class MyInstanceEnv(NodePairStepInstanceRLEnv):

    def __init__(self, p_net, v_net, controller, recorder, counter, logger, config, **kwargs):
        super().__init__(p_net, v_net, controller, recorder, counter, logger, config, **kwargs)
        self.debug = False

        # 🔥 đảm bảo ranking tồn tại
        if not hasattr(self.v_net, "ranked_nodes"):
            from ...rank.node_rank import rank_nodes
            rank_nodes(self.v_net, self.config.solver.node_ranking_method)

    @property
    def heuristic_v_node_id(self):
        if self.num_placed_v_net_nodes == self.v_net.num_nodes:
            return 0
        return self.v_net.ranked_nodes[self.num_placed_v_net_nodes]

    def get_observation(self):

        obs = self.feature_constructor.construct(
            self.p_net,
            self.v_net,
            self.solution,
            self.heuristic_v_node_id
        )

        mask = self.generate_action_mask()  # (V, P)

        obs["action_mask"] = mask
        obs["curr_v_node_id"] = int(self.heuristic_v_node_id)
        obs["v_net_size"] = int(self.v_net.num_nodes)

        if self.debug:
            print("p_net", obs["p_net_x"].shape)
            print("v_net", obs["v_net_x"].shape)
            print("mask", obs["action_mask"].shape)
            print("heuristic", obs["curr_v_node_id"])

        return obs

    def compute_reward(self, solution):

        if self.debug:
            print("calculate reward")

        weight = 1.0 / self.v_net.num_nodes

        if solution["result"]:

            reward = solution["v_net_r2c_ratio"]

        elif solution["place_result"] and solution["route_result"]:

            reward = weight

        else:

            reward = -weight

        self.solution["v_net_reward"] += reward

        if self.debug:
            print("reward", reward)

        return reward