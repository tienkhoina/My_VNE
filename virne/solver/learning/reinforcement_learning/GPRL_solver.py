# ==============================================================================
# GPRL Solver
# ==============================================================================

import torch
from omegaconf import DictConfig, open_dict

from virne.network import PhysicalNetwork, VirtualNetwork
from virne.core import Controller, Recorder, Counter, Logger

from virne.solver import SolverRegistry

from virne.solver.learning.rl_core import (
    NodePairStepInstanceRLEnv,
)

from virne.solver.learning.rl_core.rl_solver import (
    PGSolver,
    A2CSolver,
    PPOSolver,
    A3CSolver,
)

from virne.solver.learning.rl_core.instance_agent import (
    InstanceAgent,
)

from virne.solver.learning.rl_core.tensor_convertor import (
    TensorConvertor,
)

from virne.solver.learning.rl_core.policy_builder import (
    PolicyBuilder, OptimizerBuilder
)

from virne.solver.learning.reinforcement_learning.solver_maker import (
    make_solver_class,
)

from virne.solver.learning.rl_policy.GPRL_policy import (
    ActorCritic,
)

# =============================================================================
# OBS AS TENSOR
# =============================================================================
#
# GPRL action mask:
#
# [Np * Nv]
#
# flatten order:
#
# [p0v0, p0v1, ..., p1v0, ...]
#
# =============================================================================
def obs_as_tensor_for_gprl(obs, device):

    import torch

    from torch_geometric.data import (
        Data,
        Batch
    )

    # ============================================================
    # PPO UPDATE
    # ============================================================

    if isinstance(obs, list):

        p_list = []
        v_list = []

        v_size_list = []

        action_mask_list = []

        for o in obs:

            # ----------------------------------------------------
            # PHYSICAL GRAPH
            # ----------------------------------------------------

            p_data = Data(

                x=torch.FloatTensor(
                    o["p_net_x"]
                ),

                edge_index=torch.LongTensor(
                    o["p_net_edge_index"]
                ),

                edge_attr=torch.FloatTensor(
                    o["p_net_edge_attr"]
                )

            )

            # ----------------------------------------------------
            # VIRTUAL GRAPH
            # ----------------------------------------------------

            v_data = Data(

                x=torch.FloatTensor(
                    o["v_net_x"]
                ),

                edge_index=torch.LongTensor(
                    o["v_net_edge_index"]
                ),

                edge_attr=torch.FloatTensor(
                    o["v_net_edge_attr"]
                )

            )

            p_list.append(p_data)

            v_list.append(v_data)

            # ----------------------------------------------------
            # VIRTUAL NETWORK SIZE
            # ----------------------------------------------------

            v_size = o["v_net_size"]

            if isinstance(v_size, tuple):
                v_size = v_size[0]

            v_size = int(v_size)

            v_size_list.append(v_size)

            # ----------------------------------------------------
            # ACTION MASK
            # ----------------------------------------------------
            #
            # env mask shape:
            # [Nv, Np]
            #
            # keep semantic unchanged
            #
            # ----------------------------------------------------

            mask = o["action_mask"]

            if not torch.is_tensor(mask):

                mask = torch.FloatTensor(mask)

            action_mask_list.append(mask)

        # ========================================================
        # PYG BATCH
        # ========================================================

        p_batch = Batch.from_data_list(
            p_list
        ).to(device)

        v_batch = Batch.from_data_list(
            v_list
        ).to(device)

        # ========================================================
        # VIRTUAL NETWORK SIZE
        # ========================================================

        v_net_size = torch.LongTensor(
            v_size_list
        ).to(device)

        # ========================================================
        # PAD ACTION MASK
        # ========================================================
        #
        # final shape:
        # [B, MAX_V, Np]
        #
        # ========================================================

        MAX_V_NODES = 10

        max_v = MAX_V_NODES

        padded_masks = []

        for mask in action_mask_list:

            curr_v = mask.size(0)

            if curr_v < max_v:

                pad = torch.zeros(

                    max_v - curr_v,

                    mask.size(1),

                    dtype=mask.dtype

                )

                mask = torch.cat(
                    [mask, pad],
                    dim=0
                )

            padded_masks.append(mask)

        action_mask = torch.stack(
            padded_masks
        ).to(device)

        return {

            "p_net": p_batch,

            "v_net": v_batch,

            "v_net_size": v_net_size,

            "action_mask": action_mask,

        }

    # ============================================================
    # SINGLE OBSERVATION
    # ============================================================

    p_data = Data(

        x=torch.FloatTensor(
            obs["p_net_x"]
        ),

        edge_index=torch.LongTensor(
            obs["p_net_edge_index"]
        ),

        edge_attr=torch.FloatTensor(
            obs["p_net_edge_attr"]
        )

    )

    v_data = Data(

        x=torch.FloatTensor(
            obs["v_net_x"]
        ),

        edge_index=torch.LongTensor(
            obs["v_net_edge_index"]
        ),

        edge_attr=torch.FloatTensor(
            obs["v_net_edge_attr"]
        )

    )

    # ============================================================
    # VIRTUAL NETWORK SIZE
    # ============================================================

    v_size = obs["v_net_size"]

    if isinstance(v_size, tuple):
        v_size = v_size[0]

    v_size = int(v_size)

    v_net_size = torch.LongTensor(
        [v_size]
    ).to(device)

    # ============================================================
    # ACTION MASK
    # ============================================================

    # ============================================================
    # ACTION MASK
    # ============================================================

    mask = obs["action_mask"]

    if not torch.is_tensor(mask):

        mask = torch.FloatTensor(mask)

    # ------------------------------------------------------------
    # PAD TO FIXED MAX_V_NODES
    # ------------------------------------------------------------

    MAX_V_NODES = 10

    curr_v = mask.size(0)

    if curr_v < MAX_V_NODES:

        pad = torch.zeros(

            MAX_V_NODES - curr_v,

            mask.size(1),

            dtype=mask.dtype

        )

        mask = torch.cat(
            [mask, pad],
            dim=0
        )

    # ------------------------------------------------------------
    # FINAL:
    # [1, MAX_V_NODES, Np]
    # ------------------------------------------------------------

    action_mask = mask.unsqueeze(0).to(device)

    # ============================================================
    # BATCH GRAPH
    # ============================================================

    p_batch = Batch.from_data_list(
        [p_data]
    ).to(device)

    v_batch = Batch.from_data_list(
        [v_data]
    ).to(device)

    return {

        "p_net": p_batch,

        "v_net": v_batch,

        "v_net_size": v_net_size,

        "action_mask": action_mask,

    }
obs_as_tensor = obs_as_tensor_for_gprl


# =============================================================================
# GPRL ENV
# =============================================================================
#
# Uses NodePairStepInstanceRLEnv directly.
#
# Current implementation already matches:
#
# p_node_id = action // Nv
# v_node_id = action % Nv
#
# and:
#
# mask = mask.T.flatten()
#
# =============================================================================

class GPRLInstanceEnv(NodePairStepInstanceRLEnv):

    MAX_V_NODES = 10

    def __init__(
        self,
        p_net: PhysicalNetwork,
        v_net: VirtualNetwork,
        controller: Controller,
        recorder: Recorder,
        counter: Counter,
        logger: Logger,
        config: DictConfig,
        **kwargs
    ):

        with open_dict(config):

            config.rl.feature_constructor.name = \
                'p_net_v_net'

        super().__init__(
            p_net,
            v_net,
            controller,
            recorder,
            counter,
            logger,
            config,
            **kwargs
        )

    # =========================================================================
    # OVERRIDE ACTION MASK
    # =========================================================================
    #
    # Original NodePair mask:
    #
    # [Nv, Np]
    #
    # PPO requires:
    #
    # [MAX_V, Np]
    #
    # flatten:
    #
    # [Np * MAX_V]
    #
    # flatten order:
    #
    # [p0v0,p0v1,...,p1v0,...]
    #
    # =========================================================================

    def generate_action_mask(self):

        # ---------------------------------------------------------------------
        # ORIGINAL MASK
        #
        # shape:
        # [Nv, Np]
        # ---------------------------------------------------------------------

        mask = super().generate_action_mask()

        import numpy as np

        curr_v = mask.shape[0]

        # ---------------------------------------------------------------------
        # PAD VIRTUAL NODES
        # ---------------------------------------------------------------------

        if curr_v < self.MAX_V_NODES:

            pad = np.zeros(

                (
                    self.MAX_V_NODES - curr_v,
                    mask.shape[1]
                ),

                dtype=mask.dtype

            )

            mask = np.concatenate(
                [mask, pad],
                axis=0
            )

        # ---------------------------------------------------------------------
        # TRANSPOSE
        #
        # [MAX_V, Np]
        # ->
        # [Np, MAX_V]
        #
        # flatten order:
        #
        # [p0v0,p0v1,...,p1v0,...]
        #
        # EXACTLY matches:
        #
        # p_node = action // Nv
        # v_node = action % Nv
        # ---------------------------------------------------------------------

        mask = mask.T

        # ---------------------------------------------------------------------
        # FINAL:
        #
        # [Np * MAX_V]
        #
        # example:
        #
        # 100 * 10 = 1000
        # ---------------------------------------------------------------------

        mask = mask.flatten()

        return mask

    # =========================================================================
    # OVERRIDE STEP
    # =========================================================================
    #
    # IMPORTANT:
    #
    # Original NodePairStepInstanceRLEnv uses:
    #
    # p_node_id = action // self.v_net.num_nodes
    # v_node_id = action % self.v_net.num_nodes
    #
    # But GPRL PPO uses FIXED action space:
    #
    # [Np * MAX_V_NODES]
    #
    # Therefore decode MUST use:
    #
    # MAX_V_NODES
    #
    # =========================================================================

    def step(self, action):

        # =========================================================
        # FIXED ACTION SPACE DECODE
        # =========================================================
        #
        # PPO fixed action space:
        #
        # [Np * MAX_V_NODES]
        #
        # flatten order:
        #
        # [p0v0,p0v1,...,p1v0,...]
        #
        # =========================================================

        p_node_id = action // self.MAX_V_NODES

        v_node_id = action % self.MAX_V_NODES

        # =========================================================
        # PADDED VIRTUAL NODE
        # =========================================================

        if v_node_id >= self.v_net.num_nodes:

            solution = self.solution

            solution['result'] = False

            solution['place_result'] = False

            solution['route_result'] = False

            solution['description'] = \
                'select padded vnode'

            self.done = True

            return (
                self.get_observation(),
                0.,
                True,
                self.get_info(solution)
            )

        # =========================================================
        # RE-ENCODE TO ORIGINAL NODEPAIR ACTION
        # =========================================================
        #
        # original NodePair semantic:
        #
        # action =
        # p_node_id * Nv_real + v_node_id
        #
        # =========================================================

        real_action = (

            p_node_id * self.v_net.num_nodes

            + v_node_id

        )

        # =========================================================
        # CALL ORIGINAL NODEPAIR STEP
        # =========================================================

        return super().step(real_action)

# =============================================================================
# POLICY BUILDER
# =============================================================================

def build_gprl_policy(agent):

    feature_dim_config = \
        PolicyBuilder.get_feature_dim_config(
            agent.config
        )

    general_config = \
        PolicyBuilder.get_general_nn_config(
            agent.config
        )

    policy = ActorCritic(

        p_net_num_nodes=
            feature_dim_config['p_net_num_nodes'],

        p_net_x_dim=
            feature_dim_config['p_net_x_dim'],

        p_net_edge_dim=
            feature_dim_config['p_net_edge_dim'],

        v_net_x_dim=
            feature_dim_config['v_net_x_dim'],

        v_net_edge_dim=
            feature_dim_config['v_net_edge_dim'],

        **general_config

    ).to(agent.device)

    optimizer = torch.optim.Adam(

        policy.parameters(),

        lr=agent.config.rl.learning_rate.actor,

    )

    return policy, optimizer


# =============================================================================
# MANUAL REGISTRATION
# =============================================================================

class PpoGPRLSolver(InstanceAgent, PPOSolver):

    def __init__(
        self,
        controller,
        recorder,
        counter,
        logger,
        config,
        **kwargs
    ):

        InstanceAgent.__init__(
            self,
            GPRLInstanceEnv
        )

        PPOSolver.__init__(
            self,
            controller,
            recorder,
            counter,
            logger,
            config,
            build_gprl_policy,
            obs_as_tensor,
            **kwargs
        )


# =============================================================================
# AUTO SOLVER GENERATION
# =============================================================================

POLICY_BUILDERS = {
    'gprl': build_gprl_policy,
}

gprl_solvers = [

    {
        'solver_name': 'pg_gprl',
        'policy_key': 'gprl',
        'solver_cls_name': 'PgGPRLSolver',
        'rl_solver_cls': PGSolver
    },

    {
        'solver_name': 'a2c_gprl',
        'policy_key': 'gprl',
        'solver_cls_name': 'A2CGPRLSolver',
        'rl_solver_cls': A2CSolver
    },

    {
        'solver_name': 'a3c_gprl',
        'policy_key': 'gprl',
        'solver_cls_name': 'A3CGPRLSolver',
        'rl_solver_cls': A3CSolver
    },

    {
        'solver_name': 'ppo_gprl',
        'policy_key': 'gprl',
        'solver_cls_name': 'PpoGPRLSolver',
        'rl_solver_cls': PPOSolver
    },
]


# =============================================================================
# REGISTER ALL SOLVERS
# =============================================================================

for solver_info in gprl_solvers:

    solver_name = solver_info['solver_name']

    policy_key = solver_info['policy_key']

    policy_builder = POLICY_BUILDERS[
        policy_key
    ]

    instance_env_cls = GPRLInstanceEnv

    base_solver_cls = solver_info[
        'rl_solver_cls'
    ]

    make_solver_class(
        solver_name,
        instance_env_cls,
        base_solver_cls,
        policy_builder,
        obs_as_tensor
    )