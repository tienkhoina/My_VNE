# =============================================================================
# FLAG ActorCritic REAL — MATCH PAPER
# =============================================================================

import torch
import torch.nn as nn

from torch_geometric.utils import to_dense_batch

from virne.solver.learning.neural_network import (
    GCNConvNet,
    GraphPooling,
)

from virne.solver.learning.rl_policy.base_policy import (
    BaseActorCritic,
    ActorCriticRegistry,
)


# ============================================================
# Encoder (paper version)
# ============================================================


class NetEncoder(nn.Module):

    def __init__(
        self,
        feat_dim,
        edge_dim,
        embedding_dim=128,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        # init embedding (paper)
        self.init_lin = nn.Linear(
            feat_dim,
            embedding_dim,
        )

        # GNN
        self.gnn = GCNConvNet(
            embedding_dim,
            embedding_dim,
            num_layers=2,
            embedding_dim=embedding_dim,
            edge_dim=edge_dim,
        )

        # pooling
        self.pool = GraphPooling("mean")

    # --------------------------------------------------------

    def forward(self, net):

        x0 = self.init_lin(net.x)

        net = net.clone()
        net.x = x0

        node = self.gnn(net)

        g = self.pool(node, net.batch)

        dense, _ = to_dense_batch(node, net.batch)

        init_dense, _ = to_dense_batch(x0, net.batch)

        # paper combine
        dense = dense + g.unsqueeze(1) + init_dense

        return node, g, dense


# ============================================================
# Base model
# ============================================================


class BaseModel(nn.Module):

    def __init__(
        self,
        p_net_x_dim,
        p_net_edge_dim,
        v_net_x_dim,
        v_net_edge_dim,
        embedding_dim=128,
    ):
        super().__init__()

        self.p_enc = NetEncoder(
            p_net_x_dim,
            p_net_edge_dim,
            embedding_dim,
        )

        self.v_enc = NetEncoder(
            v_net_x_dim,
            v_net_edge_dim,
            embedding_dim,
        )

    # --------------------------------------------------------

    def forward(self, obs):

        v_node, v_g, v_dense = self.v_enc(
            obs["v_net"]
        )

        p_node, p_g, p_dense = self.p_enc(
            obs["p_net"]
        )

        return v_g, v_dense, p_g, p_dense


# ============================================================
# Actor
# ============================================================


class Actor(nn.Module):

    def __init__(self, embedding_dim=128):
        super().__init__()

        self.high = nn.Linear(
            embedding_dim,
            1,
        )

        self.low = nn.Linear(
            embedding_dim,
            1,
        )

    # ----------------------------

    def high_policy(self, x):

        logits = self.high(x).squeeze(-1)

        return logits

    # ----------------------------

    def low_policy(self, x):

        logits = self.low(x).squeeze(-1)

        return logits


# ============================================================
# Critic
# ============================================================


class Critic(nn.Module):

    def __init__(self, embedding_dim=128):
        super().__init__()

        self.v = nn.Linear(
            embedding_dim,
            1,
        )

    def forward(self, x):

        value = self.v(x).mean(1)

        return value


# ============================================================
# ActorCritic
# ============================================================


@ActorCriticRegistry.register("flag_real")
class FlagActorCriticReal(BaseActorCritic):

    def __init__(
        self,
        p_net_num_nodes,
        p_net_x_dim,
        p_net_edge_dim,
        v_net_x_dim,
        v_net_edge_dim,
        embedding_dim=128,
        **kwargs,
    ):
        super().__init__()

        self.encoder = BaseModel(
            p_net_x_dim,
            p_net_edge_dim,
            v_net_x_dim,
            v_net_edge_dim,
            embedding_dim,
        )

        self.actor = Actor(
            embedding_dim,
        )

        self.critic = Critic(
            embedding_dim,
        )

    # =====================================================

    def forward(
        self,
        obs,
        actor_high=False,
        actor_low=False,
        critic=False,
        high_level_action=None,
    ):

        v_g, v_dense, p_g, p_dense = self.encoder(
            obs
        )

        # ---------------- HIGH ----------------

        if actor_high:

            state = v_dense + p_g.unsqueeze(1)

            return self.actor.high_policy(
                state
            )

        # ---------------- LOW ----------------

        if actor_low:

            curr = high_level_action.unsqueeze(
                1
            ).unsqueeze(1)

            curr_v = v_dense.gather(
                1,
                curr.expand(
                    v_dense.size(0),
                    -1,
                    v_dense.size(-1),
                ),
            ).squeeze(1)

            v_g = v_g + curr_v

            state = p_dense + v_g.unsqueeze(1)

            return self.actor.low_policy(
                state
            )

        # ---------------- CRITIC ----------------

        if critic:

            state = p_dense + v_g.unsqueeze(1)

            return self.critic(state)

    # =====================================================
    def act(self, obs):

        logits = self.forward(
            obs,
            actor_high=True,
        )

        return logits


    def evaluate(self, obs):

        v_g, v_dense, p_g, p_dense = self.encoder(
            obs
        )

        state = p_dense + v_g.unsqueeze(1)

        value = self.critic(state)

        return value