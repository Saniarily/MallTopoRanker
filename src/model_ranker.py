import torch
import torch.nn as nn

from .model_components import QueryFeatureTransformer, GraphSAGEEncoder, MetricTokenizer

class GraphMatchRanker(nn.Module):
    def __init__(self, q_features: int, metric_features: int, node_in_dim: int,
                 d_model: int = 64, gnn_layers: int = 3, cross_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.q_enc = QueryFeatureTransformer(q_features, d_model, nhead=cross_heads, dropout=dropout)
        self.g_enc = GraphSAGEEncoder(node_in_dim, d_model, d_model, layers=gnn_layers, dropout=dropout)
        self.m_tok = MetricTokenizer(metric_features, d_model)

        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=cross_heads,
                                                dropout=dropout, batch_first=True)

        self.scorer = nn.Sequential(
            nn.Linear(d_model * 5, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, 1)
        )

        # Apply explicit initialization for better training stability.
        self.apply(self._init_weights)

    @staticmethod
    def _normalize_node_features(x_node: torch.Tensor) -> torch.Tensor:
        # x_node[:, 0] is Total_L_Neighbors. Log scaling reduces long-tail magnitude.
        x = x_node.clone()
        x[:, 0] = torch.log1p(torch.clamp(x[:, 0], min=0.0))
        return x

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.MultiheadAttention):
            if module.in_proj_weight is not None:
                nn.init.xavier_uniform_(module.in_proj_weight)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
            nn.init.xavier_uniform_(module.out_proj.weight)
            if module.out_proj.bias is not None:
                nn.init.zeros_(module.out_proj.bias)
        elif module.__class__.__name__ == "SAGEConv":
            # Keep PyG's own parameter policy for graph conv layers.
            module.reset_parameters()

    def forward(self, q, m, x_node, edge_index, batch):
        """
        q: [B, Fq]
        m: [B, Fm]
        x_node: [sumN, node_in_dim]
        edge_index: [2, sumE]
        batch: [sumN] -> graph id
        """
        q_tokens, q_global = self.q_enc(q)   # [B,Fq,d], [B,d]

        x_node = self._normalize_node_features(x_node)
        node_emb, g_global = self.g_enc(x_node, edge_index, batch)  # [sumN,d], [B,d]

        # 组装 node token 为 padded batch
        # 为简化：用 PyG 的 to_dense_batch
        from torch_geometric.utils import to_dense_batch
        node_dense, node_mask = to_dense_batch(node_emb, batch)  # [B, Nmax, d], [B,Nmax]

        metric_tokens = self.m_tok(m)  # [B, M, d]
        graph_tokens = torch.cat([metric_tokens, node_dense], dim=1)  # [B, M+Nmax, d]

        # key_padding_mask: True 表示需要mask
        # metric token 全有效；node token 用 node_mask 标识有效
        B, Nmax = node_mask.shape
        M = metric_tokens.shape[1]
        pad_mask = torch.cat([torch.zeros((B, M), dtype=torch.bool, device=q.device),
                              ~node_mask], dim=1)  # [B, M+Nmax]

        # Cross-Attn: QueryTokens attend to GraphTokens
        cross_out, attn = self.cross_attn(
            query=q_tokens, key=graph_tokens, value=graph_tokens,
            key_padding_mask=pad_mask,
            need_weights=True, average_attn_weights=False
        )
        # attn: [B, heads, Fq, (M+Nmax)]
        cross_global = cross_out.mean(dim=1)  # [B, d]

        inter = torch.cat([
            q_global,
            g_global,
            cross_global,
            torch.abs(q_global - g_global),
            q_global * g_global
        ], dim=1)

        score = self.scorer(inter).squeeze(-1)  # [B]
        return score, attn, M  # M用于切分metric/node