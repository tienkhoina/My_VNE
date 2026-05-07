# script_test_full_pipeline.py

import hydra
from omegaconf import DictConfig

from virne.system import BaseSystem
from virne.utils.config import add_simulation_into_config, generate_run_id

from ortools.linear_solver import pywraplp


# =========================
# ILP GENERIC (LP relaxation)
# =========================
def solve_ilp_path(p_net, v_net, v_link):
    source, target = v_link

    solver = pywraplp.Solver.CreateSolver('SCIP')

    # directed edges
    directed_edges = []
    for (u, v) in p_net.edges:
        directed_edges.append((u, v))
        directed_edges.append((v, u))

    # variables (continuous)
    x = {}
    for (u, v) in directed_edges:
        x[(u, v)] = solver.NumVar(0, 1, f'x_{u}_{v}')

    # objective: shortest path
    solver.Minimize(sum(x[e] for e in directed_edges))

    # flow constraints
    for node in p_net.nodes:
        out_flow = sum(x[(node, v)] for v in p_net.neighbors(node))
        in_flow  = sum(x[(u, node)] for u in p_net.neighbors(node))

        if node == source:
            solver.Add(out_flow - in_flow == 1)
        elif node == target:
            solver.Add(in_flow - out_flow == 1)
        else:
            solver.Add(out_flow - in_flow == 0)

    # resource constraints (generic)
    resource_attrs = [
        attr for attr in p_net.link_attrs.values()
        if attr.type == 'resource'
    ]

    for (u, v) in p_net.edges:
        for attr in resource_attrs:
            demand = attr.get(v_net, v_link)
            capacity = attr.get(p_net, (u, v))

            solver.Add(demand * x[(u, v)] <= capacity)
            solver.Add(demand * x[(v, u)] <= capacity)

    status = solver.Solve()

    if status != pywraplp.Solver.OPTIMAL:
        return None

    # reconstruct path
    path = [source]
    current = source
    visited = set()

    while current != target:
        visited.add(current)
        found = False

        for neighbor in p_net.neighbors(current):
            if x[(current, neighbor)].solution_value() > 0.5:
                if neighbor in visited:
                    continue
                path.append(neighbor)
                current = neighbor
                found = True
                break

        if not found:
            return None

    return path


# =========================
# PRINT EDGE STATE
# =========================
def print_edge_state(p_net, path, title):
    print(f"\n=== {title} ===")

    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]

        print(f"Edge ({u},{v})")

        for attr in p_net.link_attrs.values():
            val = attr.get(p_net, (u, v))
            print(f"  {attr.name}: {val}")


# =========================
# MAIN
# =========================
@hydra.main(config_path="settings", config_name="main", version_base=None)
def main(config: DictConfig):

    print("\n=== INIT SYSTEM ===")

    if config.experiment.run_id == 'auto':
        config.experiment.run_id = generate_run_id()

    add_simulation_into_config(config)

    system = BaseSystem.from_config(config)

    # 🔥 MUST RESET
    system.env.reset(config.experiment.seed)

    p_net = system.env.p_net
    v_net = system.env.v_net_simulator.v_nets[0]
    controller = system.controller
    updator = controller.resource_updator
    constraint_checker = controller.constraint_checker

    # =========================
    # PICK LINK
    # =========================
    v_link = list(v_net.links)[0]
    print("\n=== VIRTUAL LINK ===")
    print(v_link, v_net.links[v_link])

    # =========================
    # SOLVE ILP
    # =========================
    path = solve_ilp_path(p_net, v_net, v_link)

    print("\n=== ILP PATH ===")
    print(path)

    if path is None:
        print("❌ No feasible path")
        return

    # =========================
    # BEFORE UPDATE
    # =========================
    print_edge_state(p_net, path, "BEFORE UPDATE")

    # =========================
    # CHECK CONSTRAINT BEFORE
    # =========================
    print("\n=== CONSTRAINT BEFORE ===")

    for i in range(len(path) - 1):
        p_link = (path[i], path[i+1])

        result, info = constraint_checker.check_link_level_constraints(
            v_net, p_net, v_link, p_link
        )

        print(p_link, "->", result, info)

    # =========================
    # UPDATE PATH
    # =========================
    print("\n=== APPLY UPDATE ===")

    updator.update_path_resources(
        v_net,
        p_net,
        v_link,
        path,
        operator='-',
        safe=True
    )

    # =========================
    # AFTER UPDATE
    # =========================
    print_edge_state(p_net, path, "AFTER UPDATE")

    # =========================
    # CHECK CONSTRAINT AFTER
    # =========================
    print("\n=== CONSTRAINT AFTER ===")

    for i in range(len(path) - 1):
        p_link = (path[i], path[i+1])

        result, info = constraint_checker.check_link_level_constraints(
            v_net, p_net, v_link, p_link
        )

        print(p_link, "->", result, info)


if __name__ == "__main__":
    main()