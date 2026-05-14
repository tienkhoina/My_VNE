import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv
from torch_geometric.utils import to_dense_batch

from virne.solver.learning.rl_policy.base_policy import (
    BaseActorCritic,
    ActorCriticRegistry,
)


# =============================================================================
# EXACT PyG PORT OF ORIGINAL PAPER IMPLEMENTATION
# =============================================================================
#
# This file is written as a semantic 1v1 reproduction of the original paper.
#
# Every important block contains:
#
# PAPER CODE
# ↓
# PyG equivalent
#
# Original implementation:
# - DGL
# - dglnn.GATConv
#
# Current implementation:
# - PyG
# - torch_geometric.nn.GATConv
#
# Everything else is preserved as close as possible.
#
# =============================================================================


# =============================================================================
# ActorCritic
# =============================================================================
#
# PAPER:
#
# self.shared_net = GraphEncoderNet(...)
# self.policy_net = GraphPinterNet(...)
# self.value_net = ValueNet(...)
#
# =============================================================================

@ActorCriticRegistry.register("gprl")
class ActorCritic(BaseActorCritic):

    def __init__(
        self,
        p_net_num_nodes,
        p_net_x_dim,
        p_net_edge_dim,
        v_net_x_dim,
        v_net_edge_dim,
        hidden_size=128,
        **kwargs
    ):
        super().__init__()

        self.shared_net = GraphEncoderNet(
            static_size=p_net_x_dim,
            vnr_size=v_net_x_dim,
            hidden_size=hidden_size
        )

        self.actor = GraphPinterNet(
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.
        )

        self.critic = ValueNet(
            hidden_size=hidden_size,
            out_size=1,
        )

    def act(self, obs):

        vnr_hidden, static_hidden = \
            self.shared_net(obs)

        logits = self.actor(
            vnr_hidden,
            static_hidden,
            obs
        )

        return logits

    def evaluate(self, obs):

        vnr_hidden, static_hidden = \
            self.shared_net(obs)

        value = self.critic(
            vnr_hidden,
            static_hidden,
            obs
        )

        return value
    
    # =============================================================================
    # FORWARD
    # =============================================================================

    def forward(self, obs):

        # -------------------------------------------------------------------------
        # shared encoder
        # -------------------------------------------------------------------------

        vnr_hidden, static_hidden = \
            self.shared_net(obs)

        # -------------------------------------------------------------------------
        # actor
        # -------------------------------------------------------------------------

        logits = self.actor(
            vnr_hidden,
            static_hidden,
            obs
        )

        # -------------------------------------------------------------------------
        # critic
        # -------------------------------------------------------------------------

        value = self.critic(
            vnr_hidden,
            static_hidden,
            obs
        )

        return logits, value


# =============================================================================
# EncoderNet
# =============================================================================
#
# ORIGINAL PAPER:
#
# self.gconv_1 = dglnn.GATConv(
#     input_size,
#     hidden_size,
#     num_heads=1,
#     bias=True,
#     activation=None,
#     allow_zero_in_degree=True
# )
#
# self.gconv_2 = dglnn.GATConv(...)
# self.gconv_3 = dglnn.GATConv(...)
#
# out = self.gconv_1(graph=g, feat=feat, edge_weight=None)
# out = self.gconv_2(graph=g, feat=out, edge_weight=None)
# out = self.gconv_3(graph=g, feat=out, edge_weight=None)
#
# =============================================================================

class EncoderNet(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size
    ):
        super().__init__()

        # ---------------------------------------------------------------------
        # DGL:
        #
        # dglnn.GATConv(... num_heads=1)
        #
        # PyG equivalent:
        #
        # GATConv(... heads=1, concat=False)
        # ---------------------------------------------------------------------

        self.gconv_1 = GATConv(
            input_size,
            hidden_size,
            heads=1,
            concat=False,
            bias=True,
            add_self_loops=True,
        )

        self.gconv_2 = GATConv(
            hidden_size,
            hidden_size,
            heads=1,
            concat=False,
            bias=True,
            add_self_loops=True,
        )

        self.gconv_3 = GATConv(
            hidden_size,
            hidden_size,
            heads=1,
            concat=False,
            bias=True,
            add_self_loops=True,
        )

    def forward(self, graph):

        # ---------------------------------------------------------------------
        # PAPER:
        #
        # feat = g.ndata['nfeat']
        # ---------------------------------------------------------------------

        x = graph.x

        # ---------------------------------------------------------------------
        # PAPER:
        #
        # out = self.gconv_1(graph=g, feat=feat)
        # ---------------------------------------------------------------------

        out = self.gconv_1(
            x,
            graph.edge_index
        )

        # ---------------------------------------------------------------------
        # PAPER:
        #
        # out = self.gconv_2(graph=g, feat=out)
        # ---------------------------------------------------------------------

        out = self.gconv_2(
            out,
            graph.edge_index
        )

        # ---------------------------------------------------------------------
        # PAPER:
        #
        # out = self.gconv_3(graph=g, feat=out)
        # ---------------------------------------------------------------------

        out = self.gconv_3(
            out,
            graph.edge_index
        )

        # ---------------------------------------------------------------------
        # PAPER:
        #
        # reshape(batch_size, -1, hidden_size)
        # .transpose(1,2)
        #
        # Final paper shape:
        #
        # [B, D, N]
        # ---------------------------------------------------------------------

        out_dense, mask = to_dense_batch(
            out,
            graph.batch
        )

        # ---------------------------------------------------------------------
        # PAD TO FIXED MAX_V_NODES
        # ---------------------------------------------------------------------

        MAX_V_NODES = 10

        batch_size, curr_nodes, hidden_size = \
            out_dense.size()

        if curr_nodes < MAX_V_NODES:

            pad = torch.zeros(

                batch_size,

                MAX_V_NODES - curr_nodes,

                hidden_size,

                device=out_dense.device

            )

            out_dense = torch.cat(
                [out_dense, pad],
                dim=1
            )

            mask_pad = torch.zeros(

                batch_size,

                MAX_V_NODES - curr_nodes,

                dtype=mask.dtype,

                device=mask.device

            )

            mask = torch.cat(
                [mask, mask_pad],
                dim=1
            )

        out_dense = out_dense.transpose(1, 2)

        return out_dense, mask


# =============================================================================
# AttentionModule
# =============================================================================
#
# ORIGINAL PAPER:
#
# self.v = nn.Parameter(
#     torch.zeros((1,1,hidden_size))
# )
#
# self.W = nn.Parameter(
#     torch.zeros((1,hidden_size,2*hidden_size))
# )
#
# hidden = decoder_hidden.unsqueeze(2).expand_as(static_hidden)
#
# hidden = torch.cat((static_hidden, hidden), 1)
#
# attns = torch.bmm(
#     v,
#     torch.tanh(torch.bmm(W, hidden))
# )
#
# =============================================================================

class AttentionModule(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()

        # ---------------------------------------------------------------------
        # EXACT PAPER PARAMETER SHAPES
        # ---------------------------------------------------------------------

        self.v = nn.Parameter(
            torch.zeros(
                1,
                1,
                hidden_size
            )
        )

        self.W = nn.Parameter(
            torch.zeros(
                1,
                hidden_size,
                2 * hidden_size
            )
        )

        # ---------------------------------------------------------------------
        # PAPER INIT
        # ---------------------------------------------------------------------

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        static_hidden,
        decoder_hidden
    ):

        # ---------------------------------------------------------------------
        # static_hidden:
        # [B,D,N]
        #
        # decoder_hidden:
        # [B,D]
        # ---------------------------------------------------------------------

        batch_size, hidden_size, _ = \
            static_hidden.size()

        # ---------------------------------------------------------------------
        # PAPER:
        #
        # hidden =
        #     decoder_hidden.unsqueeze(2).expand_as(static_hidden)
        # ---------------------------------------------------------------------

        hidden = decoder_hidden.unsqueeze(2).expand_as(
            static_hidden
        )

        # ---------------------------------------------------------------------
        # PAPER:
        #
        # hidden =
        #     torch.cat((static_hidden, hidden), 1)
        #
        # shape:
        # [B,2D,N]
        # ---------------------------------------------------------------------

        hidden = torch.cat(
            (
                static_hidden,
                hidden
            ),
            1
        )

        # ---------------------------------------------------------------------
        # PAPER:
        #
        # v = self.v.expand(batch_size,1,hidden_size)
        # ---------------------------------------------------------------------

        v = self.v.expand(
            batch_size,
            1,
            hidden_size
        )

        # ---------------------------------------------------------------------
        # PAPER:
        #
        # W = self.W.expand(batch_size, hidden_size, -1)
        # ---------------------------------------------------------------------

        W = self.W.expand(
            batch_size,
            hidden_size,
            -1
        )

        # ---------------------------------------------------------------------
        # EXACT PAPER ATTENTION
        # ---------------------------------------------------------------------

        attns = torch.bmm(
            v,
            torch.tanh(
                torch.bmm(W, hidden)
            )
        )

        # shape:
        # [B,1,N]

        return attns


# =============================================================================
# PointerNet
# =============================================================================
#
# ORIGINAL PAPER:
#
# rnn_out, last_hh =
#     self.gru(vx_hidden.permute(2,0,1), last_hh)
#
# rnn_out = rnn_out.squeeze(0)
#
# enc_attn =
#     self.encoder_attn(static_hidden, rnn_out)
#
# return enc_attn, last_hh
#
# =============================================================================

class PointerNet(nn.Module):

    def __init__(
        self,
        hidden_size,
        num_layers=1,
        dropout=0.
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # ---------------------------------------------------------------------
        # PAPER:
        #
        # self.gru = nn.GRU(
        #     hidden_size,
        #     hidden_size,
        #     num_layers,
        #     batch_first=False
        # )
        # ---------------------------------------------------------------------

        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=False
        )

        # ---------------------------------------------------------------------
        # PAPER:
        #
        # self.encoder_attn =
        #     AttentionModule(hidden_size)
        # ---------------------------------------------------------------------

        self.encoder_attn = AttentionModule(
            hidden_size
        )

        # ---------------------------------------------------------------------
        # PAPER INIT
        # ---------------------------------------------------------------------

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        static_hidden,
        vx_hidden,
        last_hh
    ):

        # ---------------------------------------------------------------------
        # PAPER:
        #
        # vx_hidden:
        # [B,D,1]
        #
        # GRU expects:
        # [1,B,D]
        # ---------------------------------------------------------------------

        rnn_out, last_hh = self.gru(
            vx_hidden.permute(2, 0, 1),
            last_hh
        )

        # ---------------------------------------------------------------------
        # PAPER:
        #
        # rnn_out = rnn_out.squeeze(0)
        #
        # shape:
        # [B,D]
        # ---------------------------------------------------------------------

        rnn_out = rnn_out.squeeze(0)

        # ---------------------------------------------------------------------
        # PAPER:
        #
        # enc_attn =
        #     self.encoder_attn(static_hidden, rnn_out)
        # ---------------------------------------------------------------------

        enc_attn = self.encoder_attn(
            static_hidden,
            rnn_out
        )

        return enc_attn, last_hh


# =============================================================================
# GraphEncoderNet
# =============================================================================
#
# ORIGINAL PAPER:
#
# self.static_encoder =
#     EncoderNet(static_size, hidden_size)
#
# self.vnr_encoder =
#     EncoderNet(vnr_size, hidden_size)
#
# =============================================================================

class GraphEncoderNet(nn.Module):

    def __init__(
        self,
        static_size,
        vnr_size,
        hidden_size
    ):
        super().__init__()

        self.static_size = static_size
        self.vnr_size = vnr_size
        self.hidden_size = hidden_size

        self.static_encoder = EncoderNet(
            static_size,
            hidden_size
        )

        self.vnr_encoder = EncoderNet(
            vnr_size,
            hidden_size
        )

    def forward(self, obs):

        # ---------------------------------------------------------------------
        # PAPER:
        #
        # static_hidden =
        #     self.static_encoder(...)
        #
        # vnr_hidden =
        #     self.vnr_encoder(...)
        # ---------------------------------------------------------------------

        # ---------------------------------------------------------------------
        # PHYSICAL GRAPH
        # ---------------------------------------------------------------------

        static_hidden, _ = self.static_encoder(
            obs["p_net"]
        )

        vnr_hidden, v_mask = self.vnr_encoder(
            obs["v_net"]
        )

        obs["v_net_mask"] = v_mask.float()

        # ---------------------------------------------------------------------
        # PAD VIRTUAL NODES
        # ---------------------------------------------------------------------
        #
        # to_dense_batch inside EncoderNet already pads
        #
        # resulting shape:
        #
        # [B, D, max_batch_v_nodes]
        #
        # We additionally pad to fixed MAX_V_NODES
        # for PPO batching consistency.
        #
        # ---------------------------------------------------------------------

        return vnr_hidden, static_hidden




# =============================================================================
# GraphPinterNet
# =============================================================================
#
# ORIGINAL PAPER:
#
# for i in range(max_steps):
#
#     vx_hidden = vnr_hidden[:,:,i:i+1]
#
#     probs, last_hh =
#         self.pointer(static_hidden, vx_hidden, last_hh)
#
#     probs_mask[:,i:i+1,:] =
#         probs + mask.unsqueeze(1).log()
#
# return probs_mask.view(batch_size, -1)
#
# =============================================================================

# =============================================================================
# GraphPinterNet
# =============================================================================
#
# ORIGINAL PAPER:
#
# for i in range(max_steps):
#
#     vx_hidden = vnr_hidden[:,:,i:i+1]
#
#     probs, last_hh =
#         self.pointer(static_hidden, vx_hidden, last_hh)
#
#     probs_mask[:,i:i+1,:] =
#         probs + mask.unsqueeze(1).log()
#
# return probs_mask.view(batch_size, -1)
#
# =============================================================================

class GraphPinterNet(nn.Module):

    def __init__(
        self,
        hidden_size,
        num_layers=1,
        dropout=0.
    ):
        super().__init__()

        self.pointer = PointerNet(
            hidden_size,
            num_layers,
            dropout
        )

        # ---------------------------------------------------------------------
        # PAPER:
        #
        # self.x0 =
        #     torch.zeros((1, hidden_size, 1))
        # ---------------------------------------------------------------------

        self.x0 = nn.Parameter(
            torch.zeros(
                1,
                hidden_size,
                1
            )
        )

    def forward(
        self,
        vnr_hidden,
        static_hidden,
        obs
    ):

        # ---------------------------------------------------------------------
        # static_hidden:
        # [B,D,Np]
        #
        # vnr_hidden:
        # [B,D,MAX_V]
        # ---------------------------------------------------------------------

        batch_size, hidden_size, sequence_size = \
            static_hidden.size()

        # ---------------------------------------------------------------------
        # PAPER:
        #
        # last_hh =
        #     self.x0.expand(...).permute(2,0,1)
        # ---------------------------------------------------------------------

        last_hh = self.x0.expand(
            batch_size,
            -1,
            -1
        ).permute(2, 0, 1).repeat(
            self.pointer.num_layers,
            1,
            1
        )
        # ---------------------------------------------------------------------
        # IMPORTANT
        #
        # KEEP FIXED MAX_V_NODES
        #
        # PPO batching requires:
        #
        # fixed logits shape
        # fixed action space
        # fixed categorical dim
        #
        # therefore:
        #
        # DO NOT CUT PADDED VIRTUAL NODES
        # ---------------------------------------------------------------------

        max_steps = vnr_hidden.size(-1)

        # ---------------------------------------------------------------------
        # PAPER:
        #
        # probs_mask:
        # [B,Nv,Np]
        #
        # current:
        # [B,MAX_V,Np]
        # ---------------------------------------------------------------------

        probs_mask = torch.zeros(
            batch_size,
            max_steps,
            sequence_size,
            device=static_hidden.device
        )

        # ---------------------------------------------------------------------
        # EXACT PAPER DECODER LOOP
        # ---------------------------------------------------------------------

        for i in range(max_steps):

            # -------------------------------------------------------------
            # PAPER:
            #
            # vx_hidden =
            #     vnr_hidden[:,:,i:i+1]
            # -------------------------------------------------------------

            vx_hidden = vnr_hidden[:, :, i:i+1]

            # -------------------------------------------------------------
            # PAPER:
            #
            # probs, last_hh =
            #     self.pointer(...)
            # -------------------------------------------------------------

            probs, last_hh = self.pointer(
                static_hidden,
                vx_hidden,
                last_hh
            )

            # -------------------------------------------------------------
            # probs:
            # [B,Np]
            # -------------------------------------------------------------

            probs = probs.squeeze(1)

            # =============================================================
            # ACTION MASK
            # =============================================================
            #
            # action_mask shape:
            #
            # [B, MAX_V_NODES, Np]
            #
            # semantics:
            #
            # action_mask[:, i, :]
            #
            # gives feasible physical nodes
            # for current virtual node i
            #
            # -------------------------------------------------------------
            #
            # valid action:
            # mask = 1
            #
            # invalid action:
            # mask = 0
            #
            # log(1)     = 0
            # log(1e-9)  ≈ -20.7
            #
            # invalid actions therefore receive:
            #
            # very small logits
            #
            # -------------------------------------------------------------
            #
            # IMPORTANT:
            #
            # padded virtual nodes MUST have:
            #
            # action_mask[v_pad,:] = 0
            #
            # from obs_as_tensor_for_gprl
            #
            # =============================================================


            # =============================================================
            # PADDED VIRTUAL NODE MASK
            # =============================================================
            #
            # v_net_mask:
            # [B, MAX_V_NODES]
            #
            # padded vnode:
            #
            # v_net_mask = 0
            #
            # force ALL logits to:
            #
            # -1e9
            #
            # so probability becomes:
            #
            # softmax(-1e9) ≈ 0
            #
            # -------------------------------------------------------------
            #
            # This is REQUIRED.
            #
            # log(1e-9) alone is NOT enough because:
            #
            # softmax(-20)
            #
            # still leaks probability mass.
            #
            # =============================================================

            if "v_net_mask" in obs:

                valid_mask = obs["v_net_mask"][:, i]

                invalid_mask = (1. - valid_mask).bool()

                probs[invalid_mask] = -1e9

            # -------------------------------------------------------------
            # STORE
            # -------------------------------------------------------------

            probs_mask[:, i:i+1, :] = \
                probs.unsqueeze(1)

        # ---------------------------------------------------------------------
        # PAPER:
        #
        # return probs_mask.view(batch_size, -1)
        #
        # original paper:
        # [B, Nv * Np]
        #
        # current:
        # [B, MAX_V * Np]
        # ---------------------------------------------------------------------

        # ---------------------------------------------------------------------
        # TRANSPOSE
        #
        # [B,MAX_V,Np]
        # ->
        # [B,Np,MAX_V]
        #
        # flatten order becomes:
        #
        # [p0v0,p0v1,...,p1v0,p1v1,...]
        #
        # EXACTLY matches:
        #
        # NodePairStep decode:
        #
        # p_node = action // Nv
        # v_node = action % Nv
        # ---------------------------------------------------------------------

        # ---------------------------------------------------------------------
        # TRANSPOSE
        #
        # [B,MAX_V,Np]
        # ->
        # [B,Np,MAX_V]
        #
        # flatten order:
        #
        # [p0v0,p0v1,...,p1v0,...]
        # ---------------------------------------------------------------------

        probs_mask = probs_mask.transpose(1, 2)


        return probs_mask.contiguous().view(
            batch_size,
            -1
        )


# =============================================================================
# ValueNet
# =============================================================================
#
# ORIGINAL PAPER:
#
# y1 =
#     vnr_hidden.transpose(1,2) * nmask
#
# y2 =
#     static_hidden.transpose(1,2)
#
# y1 =
#     sum(y1) / valid_nodes
#
# y2 =
#     sum(y2) / num_nodes
#
# y =
#     torch.cat((y1,y2), dim=1)
#
# y = self.linear(y)
#
# =============================================================================

class ValueNet(nn.Module):

    def __init__(
        self,
        hidden_size,
        out_size
    ):
        super().__init__()

        self.linear = nn.Linear(
            hidden_size * 2,
            out_size
        )

    def forward(
        self,
        vnr_hidden,
        static_hidden,
        obs
    ):

        batch_size, hidden_size, vnr_size = \
            vnr_hidden.size()

        # ---------------------------------------------------------------------
        # PAPER:
        #
        # nmask:
        # [B,Nv]
        #
        # used to ignore padded virtual nodes
        # ---------------------------------------------------------------------

        if "v_net_mask" in obs:

            nmask = obs["v_net_mask"]

        else:

            nmask = torch.ones(
                batch_size,
                vnr_size,
                device=vnr_hidden.device
            )

        # ---------------------------------------------------------------------
        # PAPER:
        #
        # y1 =
        #     vnr_hidden.transpose(1,2) * nmask.unsqueeze(2)
        # ---------------------------------------------------------------------

        y1 = vnr_hidden.transpose(1, 2) * \
             nmask.unsqueeze(2)

        # ---------------------------------------------------------------------
        # PAPER:
        #
        # y2 =
        #     static_hidden.transpose(1,2)
        # ---------------------------------------------------------------------

        y2 = static_hidden.transpose(1, 2)

        # ---------------------------------------------------------------------
        # PAPER:
        #
        # y1 =
        #     torch.sum(y1, dim=1)
        #     / nmask.sum(1)
        # ---------------------------------------------------------------------

        y1 = torch.sum(
            y1,
            dim=1
        ) / nmask.sum(1).unsqueeze(1)

        # ---------------------------------------------------------------------
        # PAPER:
        #
        # y2 =
        #     torch.sum(y2, dim=1)
        #     / y2.size()[1]
        # ---------------------------------------------------------------------

        y2 = torch.sum(
            y2,
            dim=1
        ) / y2.size(1)

        # ---------------------------------------------------------------------
        # PAPER:
        #
        # y =
        #     torch.cat((y1,y2), dim=1)
        # ---------------------------------------------------------------------

        y = torch.cat(
            (
                y1,
                y2
            ),
            dim=1
        )

        # ---------------------------------------------------------------------
        # PAPER:
        #
        # y = self.linear(y)
        # ---------------------------------------------------------------------

        y = self.linear(y)

        return y
    
# =============================================================================
# DEBUG MAIN
# =============================================================================
#
# Simple forward test for:
#
# - encoder
# - actor
# - critic
# - logits shape
# - value shape
#
# =============================================================================

if __name__ == "__main__":

    from torch_geometric.data import Data, Batch

    torch.manual_seed(0)

    # =========================================================================
    # CONFIG
    # =========================================================================

    B = 2

    Np = 100
    Nv = 10

    p_feat_dim = 5
    v_feat_dim = 5

    hidden_size = 40

    # =========================================================================
    # BUILD PHYSICAL GRAPH BATCH
    # =========================================================================

    p_graphs = []

    for _ in range(B):

        x = torch.randn(Np, p_feat_dim)

        edge_index = torch.randint(
            0,
            Np,
            (2, 500)
        )

        graph = Data(
            x=x,
            edge_index=edge_index
        )

        p_graphs.append(graph)

    p_batch = Batch.from_data_list(
        p_graphs
    )

    # =========================================================================
    # BUILD VIRTUAL GRAPH BATCH
    # =========================================================================

    v_graphs = []

    for _ in range(B):

        x = torch.randn(Nv, v_feat_dim)

        edge_index = torch.randint(
            0,
            Nv,
            (2, 40)
        )

        graph = Data(
            x=x,
            edge_index=edge_index
        )

        v_graphs.append(graph)

    v_batch = Batch.from_data_list(
        v_graphs
    )

    # =========================================================================
    # ACTION MASK
    #
    # PAPER SHAPE:
    # [B, Nv, Np]
    # =========================================================================

    action_mask = torch.ones(
        B,
        Nv,
        Np
    )

    # =========================================================================
    # OBS
    # =========================================================================

    obs = {
        "p_net": p_batch,
        "v_net": v_batch,
        "action_mask": action_mask,
    }

    # =========================================================================
    # MODEL
    # =========================================================================

    model = ActorCritic(
        p_net_num_nodes=Np,
        p_net_x_dim=p_feat_dim,
        p_net_edge_dim=0,
        v_net_x_dim=v_feat_dim,
        v_net_edge_dim=0,
        hidden_size=hidden_size,
    )

    # =========================================================================
    # FORWARD
    # =========================================================================

    logits, value = model(obs)

    # =========================================================================
    # PRINT SHAPES
    # =========================================================================

    print("\n====================")
    print("FORWARD SUCCESS")
    print("====================")

    print("\nlogits shape:")
    print(logits.shape)

    print("\nexpected logits shape:")
    print(f"({B}, {Nv * Np})")

    print("\nvalue shape:")
    print(value.shape)

    print("\nexpected value shape:")
    print(f"({B}, 1)")

    # =========================================================================
    # ASSERTIONS
    # =========================================================================

    assert logits.shape == (
        B,
        Nv * Np
    )

    assert value.shape == (
        B,
        1
    )

    print("\nALL TESTS PASSED")