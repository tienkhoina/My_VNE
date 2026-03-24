import torch
import numpy as np
from torch_geometric.data import Data, Batch


def obs_as_tensor_for_flag(obs, device):

    # ------------------------
    # SINGLE → LIST
    # ------------------------

    if isinstance(obs, dict):
        obs = [obs]

    p_list = []
    v_list = []

    curr_list = []
    mask_list = []
    size_list = []

    for o in obs:

        # ---------- p_net ----------
        p = Data(
            x=torch.as_tensor(
                o["p_net_x"],
                dtype=torch.float32,
            ),
            edge_index=torch.as_tensor(
                o["p_net_edge_index"],
                dtype=torch.long,
            ),
            edge_attr=torch.as_tensor(
                o["p_net_edge_attr"],
                dtype=torch.float32,
            ),
        )

        # ---------- v_net ----------
        v = Data(
            x=torch.as_tensor(
                o["v_net_x"],
                dtype=torch.float32,
            ),
            edge_index=torch.as_tensor(
                o["v_net_edge_index"],
                dtype=torch.long,
            ),
            edge_attr=torch.as_tensor(
                o["v_net_edge_attr"],
                dtype=torch.float32,
            ),
        )

        p_list.append(p)
        v_list.append(v)

        curr_list.append(o["curr_v_node_id"])
        mask_list.append(o["action_mask"])   # giữ nguyên
        size_list.append(o["v_net_size"])

    # ------------------------
    # Batch graph
    # ------------------------

    p_batch = Batch.from_data_list(p_list).to(device)
    v_batch = Batch.from_data_list(v_list).to(device)

    # ------------------------
    # Scalars
    # ------------------------

    curr_v = torch.tensor(
        curr_list,
        dtype=torch.long,
        device=device,
    )

    v_size = torch.tensor(
        size_list,
        dtype=torch.long,
        device=device,
    )

    # ------------------------
    # MASK (không pad, không reshape)
    # ------------------------

    mask = torch.tensor(
        np.stack(mask_list, axis=0),
        dtype=torch.float32,
        device=device,
    )

    return {
        "p_net": p_batch,
        "v_net": v_batch,
        "curr_v_node_id": curr_v,
        "action_mask": mask,
        "v_net_size": v_size,
    }