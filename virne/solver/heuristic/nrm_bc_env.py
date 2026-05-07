# =========================================================
# NRM BC (ENV-EXACT + LOCAL STATE CLEAN)
# =========================================================

import os
import json
import torch
import networkx as nx
import numpy as np

from virne.solver.base_solver import Solver, SolverRegistry
from virne.core import Solution
from virne.solver.heuristic.nrm_par_solver import NRMNodeRank
from virne.solver.learning.rl_core.feature_constructor import FeatureConstructorRegistry
from virne.solver.learning.rl_core.instance_rl_environment import rank_nodes
from virne.solver.learning.rl_core.tensor_convertor import TensorConvertor
from virne.solver.debug_obs import debug_obs

@SolverRegistry.register('nrm_bc', solver_type='heuristic')
class NRMBCSolver(Solver):

    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        super().__init__(controller, recorder, counter, logger, config, **kwargs)

        self.shortest_method = config.shortest_method
        self.node_rank = NRMNodeRank()

        self.save_data = True
        self.save_path = "/app/results/bc_dataset.jsonl"

        if self.save_data:
            os.makedirs("/app/results", exist_ok=True)
            open(self.save_path, "w").close()

    # =====================================================
    # ===== UTIL: PyG → dict (save được)
    # =====================================================

    def pyg_to_dict(self, x):
        import torch
        import numpy as np

        # tensor
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy().tolist()   # 🔥 FIX

        # numpy
        if isinstance(x, np.ndarray):
            return x.tolist()                 # 🔥 FIX

        # PyG Data / Batch
        if hasattr(x, "keys") and callable(x.keys):
            out = {}
            for key in x.keys():
                val = x[key]
                if val is not None:
                    out[key] = self.pyg_to_dict(val)
            return out

        # dict
        if isinstance(x, dict):
            return {k: self.pyg_to_dict(v) for k, v in x.items()}

        # list
        if isinstance(x, list):
            return [self.pyg_to_dict(v) for v in x]

        return x

    # =====================================================
    # ===== MASK (dynamic)
    # =====================================================

    def generate_action_mask(self, v_net, p_net, solution):
        mask = np.zeros([v_net.num_nodes, p_net.num_nodes])

        selected_p_nodes = list(solution['node_slots'].values())

        for v_node_id in range(v_net.num_nodes):

            if v_node_id in solution['node_slots']:
                continue

            candidate_nodes = self.controller.find_candidate_nodes(
                v_net,
                p_net,
                v_node_id,
                filter=selected_p_nodes,
                check_node_constraint=True,
                check_link_constraint=True
            )

            mask[v_node_id][candidate_nodes] = 1

        # remove used
        for v_node_id, p_id in solution['node_slots'].items():
            mask[:, p_id] = 0
            mask[v_node_id, :] = 0

        if mask.sum() == 0:
            mask[0][0] = 1

        return mask

    # =====================================================
    # ===== OBS (env-style)
    # =====================================================

    def get_observation(self, v_net, p_net, solution, feature_constructor):

        num_placed = len(solution['node_slots'])

        if num_placed == v_net.num_nodes:
            curr_v = 0
        else:
            curr_v = v_net.ranked_nodes[num_placed]

        obs = feature_constructor.construct(
            p_net,
            v_net,
            solution,
            curr_v
        )

        mask = self.generate_action_mask(v_net, p_net, solution)

        obs["action_mask"] = mask
        obs["curr_v_node_id"] = int(curr_v)
        obs["v_net_size"] = int(v_net.num_nodes)

        return obs

    # =====================================================
    # ===== SAVE
    # =====================================================

    def save_sample(self, obs, action):

        tensor_obs = TensorConvertor.obs_as_tensor_myself(
            obs,
            device=torch.device("cpu")
        )

        sample = {
            "obs": self.pyg_to_dict(tensor_obs),
            "action": int(action)
        }

        with open(self.save_path, "a") as f:
            f.write(json.dumps(sample) + "\n")

    # =====================================================
    # SOLVE
    # =====================================================

    def solve(self, instance):

        v_net = instance['v_net']
        p_net = instance['p_net']

        # ===== rank (static)
        rank_nodes(v_net, self.config.solver.node_ranking_method)
        rank_nodes(p_net, self.config.solver.node_ranking_method)

        solution = Solution.from_v_net(v_net)

        # ===== feature constructor
        fc_name = self.config.rl.feature_constructor.name
        feature_constructor = FeatureConstructorRegistry.get(fc_name)(
            p_net,
            v_net,
            self.config
        )

        # ===== heuristic label
        v_rank = self.node_rank(v_net)
        p_rank = self.node_rank(p_net)

        sorted_v_nodes = list(v_rank)

        # =====================================================
        # MAIN LOOP
        # =====================================================

        for v_node in sorted_v_nodes:

            placed = False
            selected_p_nodes = list(solution['node_slots'].values())

            p_candidates = self.controller.find_candidate_nodes(
                v_net,
                p_net,
                v_node,
                filter=selected_p_nodes,
                check_node_constraint=True,
                check_link_constraint=True
            )

            if len(p_candidates) == 0:
                solution['result'] = False
                return solution

            # ===== score heuristic
            scores = {}

            for p_node in p_candidates:

                s_value = p_rank[p_node]

                if len(selected_p_nodes) == 0:
                    dist_penalty = 1.0
                else:
                    dist_sum = 0
                    for used_p in selected_p_nodes:
                        try:
                            dist = nx.shortest_path_length(p_net, p_node, used_p)
                        except:
                            dist = 1e6
                        dist_sum += dist

                    dist_penalty = 1.0 / (dist_sum + 1e-6)

                scores[p_node] = s_value * dist_penalty

            sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)

            # =====================================================
            # TRY PLACE
            # =====================================================

            for p_node, _ in sorted_candidates:

                obs = self.get_observation(
                    v_net,
                    p_net,
                    solution,
                    feature_constructor
                )
                

                num_v = v_net.num_nodes
                action = p_node * num_v + v_node

                if self.save_data:
                    self.save_sample(obs, action)

                # print(f"Selected action: {action}")
                # debug_obs("Selected Action", obs, p_net, v_net)

                result = self.controller.place_and_route(
                    v_net,
                    p_net,
                    v_node_id=v_node,
                    p_node_id=p_node,
                    solution=solution,
                    shortest_method=self.shortest_method
                )

                if result:
                    placed = True
                    break

            if not placed:
                solution['result'] = False
                return solution

        solution['result'] = True
        return solution