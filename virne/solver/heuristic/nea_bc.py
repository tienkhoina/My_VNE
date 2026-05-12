# =========================================================
# NEA BC (ENV-EXACT + LOCAL STATE CLEAN)
# =========================================================

import os
import json
import torch
import networkx as nx
import numpy as np

from virne.solver.base_solver import Solver, SolverRegistry
from virne.core import Solution

from ..rank.node_rank import NRMNodeRank

from virne.utils import path_to_links

from virne.solver.learning.rl_core.feature_constructor import (
    FeatureConstructorRegistry
)

from virne.solver.learning.rl_core.instance_rl_environment import (
    rank_nodes
)

from virne.solver.learning.rl_core.tensor_convertor import (
    TensorConvertor
)

from virne.solver.debug_obs import debug_obs


@SolverRegistry.register('nea_bc', solver_type='heuristic')
class NEABCSolver(Solver):

    def __init__(self, controller, recorder, counter, logger, config, **kwargs):
        super().__init__(controller, recorder, counter, logger, config, **kwargs)

        self.shortest_method = config.shortest_method
        self.node_rank = NRMNodeRank()

        self.save_data = True
        self.save_path = "/app/results/bc_dataset.jsonl"
        self.debug = False

        if self.save_data:
            os.makedirs("/app/results", exist_ok=True)
            open(self.save_path, "a").close()

    # =====================================================
    # ===== UTIL: PyG → dict
    # =====================================================

    def pyg_to_dict(self, x):

        import torch
        import numpy as np

        if isinstance(x, torch.Tensor):
            return x.cpu().numpy().tolist()

        if isinstance(x, np.ndarray):
            return x.tolist()

        if hasattr(x, "keys") and callable(x.keys):
            out = {}
            for key in sorted(x.keys()):
                val = x[key]
                if val is not None:
                    out[key] = self.pyg_to_dict(val)
            return out

        if isinstance(x, dict):
            return {k: self.pyg_to_dict(v) for k, v in x.items()}

        if isinstance(x, list):
            return [self.pyg_to_dict(v) for v in x]

        return x

    # =====================================================
    # ===== MASK
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
    # ===== OBS
    # =====================================================

    def get_observation(
        self,
        v_net,
        p_net,
        solution,
        feature_constructor
    ):

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
            f.write(json.dumps(sample, sort_keys=True) + "\n")

    # =====================================================
    # ===== SOLVE
    # =====================================================

    def solve(self, instance):

        v_net = instance['v_net']
        p_net = instance['p_net']

        # ===== rank (env-style)
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

        # =====================================================
        # STEP 1: NODE RANKING
        # =====================================================

        v_rank = self.node_rank(v_net)
        p_rank = self.node_rank(p_net)

        sorted_v_nodes = list(v_rank)
        sorted_p_nodes = list(p_rank)

        # =====================================================
        # STEP 2: GREEDY PLACE + ROUTE
        # =====================================================

        for step_id, v_node in enumerate(sorted_v_nodes):

            placed = False

            # =====================================================
            # 1. LẤY CANDIDATE NODE
            # =====================================================

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
                solution['place_result'] = False
                solution['result'] = False
                return solution

            shortest_path_length_dict = dict(nx.shortest_path_length(p_net))
            shortest_path_dict = nx.shortest_path(p_net)
            p_node_degree_dict = dict(p_net.degree())

            # multi-resource
            p_adj_link_resources = p_net.get_adjacency_attrs_data(
                p_net.get_link_attrs(['resource'])
            )

            # =====================================================
            # 2. RANK LOCAL
            # =====================================================

            p_candidate_scores = {}
            debug_list = []

            for p_node in p_candidates:

                # (1) node resource
                s_value = p_rank[p_node]

                # (2) distance penalty
                if len(selected_p_nodes) == 0:
                    dist_penalty = 1.0
                else:
                    dist_sum = sum(
                        shortest_path_length_dict[p_node].get(u, 1e6)
                        for u in selected_p_nodes
                    )

                    dist_penalty = 1.0 / (dist_sum + 1)

                # (3) degree
                degree = p_node_degree_dict[p_node]

                # (4) path quality
                path_score = 0.0

                for u in selected_p_nodes:

                    try:

                        path = shortest_path_dict[p_node][u]

                        links = path_to_links(path)

                        if len(links) == 0:
                            continue

                        total_resource = 0.0

                        for adj in p_adj_link_resources:
                            total_resource += sum(
                                adj[i][j] for i, j in links
                            )

                        path_score += (
                            total_resource / (len(links) + 1e-6)
                        )

                    except:
                        continue

                # (5) FINAL SCORE
                score = (
                    degree
                    * s_value
                    * dist_penalty
                    * (2 + path_score)
                )

                p_candidate_scores[p_node] = score

                if self.debug:
                    debug_list.append(
                        (
                            p_node,
                            s_value,
                            dist_penalty,
                            path_score,
                            score
                        )
                    )

            # =====================================================
            # SORT
            # =====================================================

            sorted_candidates = sorted(
                p_candidate_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )

            # =====================================================
            # DEBUG
            # =====================================================

            if self.debug:

                print("\n=== DEBUG STEP", step_id, "===")
                print("v_node:", v_node)
                print("selected_p_nodes:", selected_p_nodes[:10])

                print("TOP 5 candidates:")

                for p_node, _ in sorted_candidates[:5]:

                    for d in debug_list:

                        if d[0] == p_node:

                            print(
                                f"p={p_node:3d} | "
                                f"NRM={d[1]:.4f} | "
                                f"dist={d[2]:.4f} | "
                                f"path={d[3]:.4f} | "
                                f"score={d[4]:.4f}"
                            )

                print("BEST:", sorted_candidates[0][0])

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
                solution['place_result'] = False
                solution['result'] = False
                return solution

        # =====================================================
        # SUCCESS
        # =====================================================

        solution['result'] = True
        return solution