def debug_obs(tag, obs, p_net, v_net):
    import numpy as np

    print("\n" + "=" * 60)
    print(f"🔍 DEBUG OBS [{tag}]")
    print("=" * 60)

    # =========================
    # GENERAL
    # =========================
    print("curr_v_node_id:", obs["curr_v_node_id"])
    print("v_net_size:", obs["v_net_size"])

    # =========================
    # MASK
    # =========================
    mask = obs["action_mask"]
    print("\n--- MASK ---")
    print("shape:", mask.shape)
    print("sum:", np.sum(mask))
    print("row sum (first 5):", np.sum(mask, axis=1)[:5])

    # =========================
    # P_NET GRAPH (RAW)
    # =========================
    print("\n--- p_net (raw graph) ---")
    print("num_nodes:", p_net.number_of_nodes())
    print("num_edges:", p_net.number_of_edges())

    # node sample
    nodes = list(p_net.nodes(data=True))[:3]
    print("node sample:", nodes)

    # edge sample
    edges = list(p_net.edges(data=True))[:3]
    print("edge sample:", edges)

    # =========================
    # V_NET GRAPH (RAW)
    # =========================
    print("\n--- v_net (raw graph) ---")
    print("num_nodes:", v_net.number_of_nodes())
    print("num_edges:", v_net.number_of_edges())

    nodes = list(v_net.nodes(data=True))[:3]
    print("node sample:", nodes)

    edges = list(v_net.edges(data=True))[:3]
    print("edge sample:", edges)

    # =========================
    # P_NET OBS (FEATURE)
    # =========================
    print("\n--- p_net (obs) ---")
    print("x.shape:", obs["p_net_x"].shape)
    print("edge_index.shape:", obs["p_net_edge_index"].shape)
    print("edge_attr.shape:", obs["p_net_edge_attr"].shape)

    print("x sample:", obs["p_net_x"][:2])
    print("edge_index sample:", obs["p_net_edge_index"][:, :5])
    print("edge_attr sample:", obs["p_net_edge_attr"][:5])

    # =========================
    # V_NET OBS (FEATURE)
    # =========================
    print("\n--- v_net (obs) ---")
    print("x.shape:", obs["v_net_x"].shape)
    print("edge_index.shape:", obs["v_net_edge_index"].shape)
    print("edge_attr.shape:", obs["v_net_edge_attr"].shape)

    print("x sample:", obs["v_net_x"][:2])
    print("edge_index sample:", obs["v_net_edge_index"][:, :5])
    print("edge_attr sample:", obs["v_net_edge_attr"][:5])

    print("=" * 60 + "\n")