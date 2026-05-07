# =========================================================
# NRM + Greedy Place-And-Route (CORRECT VERSION + DEBUG)
# =========================================================

from virne.solver.base_solver import Solver, SolverRegistry
from virne.core import Solution

from ..rank.node_rank import NRMNodeRank
import networkx as nx
from virne.utils import path_to_links


@SolverRegistry.register('nea_par', solver_type='heuristic')
class NEAParSolver(Solver):

    def __init__(self, controller, recorder, counter, logger, config, debug=False, **kwargs):
        super().__init__(controller, recorder, counter, logger, config, **kwargs)

        self.shortest_method = config.shortest_method
        self.node_rank = NRMNodeRank()
        self.debug = debug

    def solve(self, instance):

        v_net = instance['v_net']
        p_net = instance['p_net']

        solution = Solution.from_v_net(v_net)

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

            # multi-resource (giống NEA)
            p_adj_link_resources = p_net.get_adjacency_attrs_data(
                p_net.get_link_attrs(['resource'])
            )

            # =====================================================
            # 2. RANK LOCAL
            # =====================================================
            p_candidate_scores = {}
            debug_list = []

            for p_node in p_candidates:

                # (1) node resource (giữ NRM)
                s_value = p_rank[p_node]

                # (2) distance penalty (giữ nguyên)
                if len(selected_p_nodes) == 0:
                    dist_penalty = 1.0
                else:
                    dist_sum = sum(
                        shortest_path_length_dict[p_node].get(u, 1e6)
                        for u in selected_p_nodes
                    )
                    dist_penalty = 1.0 / (dist_sum + 1)

                # (3) degree (NEA)
                degree = p_node_degree_dict[p_node]

                # (4) path quality (NEA core)
                path_score = 0.0

                for u in selected_p_nodes:
                    try:
                        path = shortest_path_dict[p_node][u]
                        links = path_to_links(path)

                        if len(links) == 0:
                            continue

                        total_resource = 0.0

                        # multi-resource aggregation
                        for adj in p_adj_link_resources:
                            total_resource += sum(adj[i][j] for i, j in links)

                        path_score += total_resource / (len(links) + 1e-6)

                    except:
                        continue



                # (5) FINAL SCORE (NEA-style + NRM)
                score = degree * s_value * dist_penalty * (2 + path_score)

                p_candidate_scores[p_node] = score

                if self.debug:
                    debug_list.append((p_node, s_value, dist_penalty, path_score, score))

            # sort candidate
            sorted_candidates = sorted(
                p_candidate_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )

            # ================= DEBUG =================
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
            # =========================================

            # =====================================================
            # 3. TRY PLACE + ROUTE
            # =====================================================
            for p_node, _ in sorted_candidates:

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