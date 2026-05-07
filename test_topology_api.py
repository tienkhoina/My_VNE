import sys
sys.path.append("/app")

import random
from omegaconf import OmegaConf

from virne.network.dataset_generator import Generator
from virne.core.controller.controller import Controller


# =========================
# LOAD CONFIG
# =========================
def load_config():
    cfg = OmegaConf.load("settings/main.yaml")

    cfg.solver.solver_name = "ppo_flag_meta"
    cfg.v_sim_setting = OmegaConf.load(
        "settings/v_sim_setting/v_sim_setting_flagvne.yaml"
    )
    cfg.p_net_setting = OmegaConf.load(
        "settings/p_net_setting/p_net_setting_flagvne.yaml"
    )

    return cfg


# =========================
# BUILD CONTROLLER (CHUẨN FRAMEWORK)
# =========================
def build_controller(config):
    return Controller(
        config.v_sim_setting['node_attrs_setting'],
        config.v_sim_setting['link_attrs_setting'],
        config.v_sim_setting.get('graph_attrs_setting', {}),
        config
    )


# =========================
# UTILS
# =========================
def path_hop(path):
    return len(path) - 1 if path else float("inf")


def compare_paths(paths_avail, paths_ilp):
    # =========================
    # BOTH FAIL → MATCH
    # =========================
    if not paths_avail and not paths_ilp:
        return "MATCH"

    # =========================
    # ONE FAIL → REAL DIFF
    # =========================
    if not paths_avail or not paths_ilp:
        return "REAL_DIFF"

    # =========================
    # BOTH HAVE PATH
    # =========================
    p1 = paths_avail[0]
    p2 = paths_ilp[0]

    hop1 = path_hop(p1)
    hop2 = path_hop(p2)

    if p1 == p2:
        return "MATCH"

    if hop1 == hop2:
        return "SAME_COST"

    return "REAL_DIFF"


# =========================
# MAIN TEST
# =========================
def main():
    config = load_config()

    print("\n🔥 Generating dataset...")
    p_net, v_sim = Generator.generate_dataset(config)

    controller = build_controller(config)
    analyzer = controller.topology_analyzer

    NUM_VNET = min(50, len(v_sim.v_nets))
    NUM_PAIR = 10

    match = 0
    same_cost = 0
    diff = 0
    fail = 0
    total = 0

    print("\n🚀 START STRESS TEST")

    for i in range(NUM_VNET):
        v_net = v_sim.v_nets[i]

        for v_link in v_net.edges:
            for _ in range(NUM_PAIR):
                source = random.choice(list(p_net.nodes))
                target = random.choice(list(p_net.nodes))

                if source == target:
                    continue

                total += 1

                try:
                    paths_avail = analyzer.find_shortest_paths(
                        v_net, p_net, v_link, (source, target),
                        method="available_shortest"
                    )

                    paths_ilp = analyzer.find_shortest_paths(
                        v_net, p_net, v_link, (source, target),
                        method="ilp_shortest"
                    )

                    result = compare_paths(paths_avail, paths_ilp)

                    if result == "MATCH":
                        match += 1

                    elif result == "SAME_COST":
                        same_cost += 1

                    elif result == "REAL_DIFF":
                        diff += 1

                        print("\n🔥 REAL MISMATCH")
                        print("v_link:", v_link)
                        print("pair:", (source, target))

                        if paths_avail:
                            print("avail:", paths_avail,
                                  "hop =", path_hop(paths_avail[0]))
                        else:
                            print("avail: None")

                        if paths_ilp:
                            print("ilp  :", paths_ilp,
                                  "hop =", path_hop(paths_ilp[0]))
                        else:
                            print("ilp  : None")

                    elif result == "FAIL":
                        fail += 1

                except Exception as e:
                    fail += 1

        print(f"Progress: {i+1}/{NUM_VNET}")

    print("\n========== RESULT ==========")
    print("Total     :", total)
    print("Match     :", match)
    print("Same cost :", same_cost)
    print("Real diff :", diff)
    print("Fail      :", fail)

    if total > 0:
        print("\nMatch %     :", match / total)
        print("Same cost % :", same_cost / total)
        print("Real diff % :", diff / total)
        print("Fail %      :", fail / total)


if __name__ == "__main__":
    main()