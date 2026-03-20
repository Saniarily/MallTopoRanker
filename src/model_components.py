import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_mean_pool

class QueryFeatureTransformer(nn.Module):
    """
    把10个结构化特征当10个token，做轻量Transformer编码。
    """
    def __init__(self, num_features: int, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model

        self.feature_id_emb = nn.Embedding(num_features, d_model)
        self.value_proj = nn.Linear(1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.pool = nn.Linear(d_model, d_model)

        fid = torch.arange(num_features, dtype=torch.long)
        self.register_buffer("feature_ids", fid, persistent=False)

    def forward(self, q: torch.Tensor):
        """
        q: [B, F]
        return:
          q_tokens: [B, F, d]
          q_global: [B, d]
        """
        B, F = q.shape
        assert F == self.num_features

        ids = self.feature_ids.unsqueeze(0).expand(B, F)  # [B,F]
        id_emb = self.feature_id_emb(ids)                 # [B,F,d]
        val_emb = self.value_proj(q.unsqueeze(-1))        # [B,F,d]
        tokens = id_emb + val_emb

        out = self.encoder(tokens)                        # [B,F,d]
        q_global = self.pool(out.mean(dim=1))             # [B,d]
        return out, q_global

class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden))
        for _ in range(layers - 2):
            self.convs.append(SAGEConv(hidden, hidden))
        self.convs.append(SAGEConv(hidden, out_dim))
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i != len(self.convs) - 1:
                h = self.act(h)
                h = self.dropout(h)
        g = global_mean_pool(h, batch)  # [B, out_dim]
        return h, g  # node_emb, graph_emb

class MetricTokenizer(nn.Module):
    """
    4个关键指标做成4个token：id_emb + value_proj
    """
    def __init__(self, num_metrics: int, d_model: int):
        super().__init__()
        self.num_metrics = num_metrics
        self.d_model = d_model
        self.metric_id_emb = nn.Embedding(num_metrics, d_model)
        self.value_proj = nn.Linear(1, d_model)

        mid = torch.arange(num_metrics, dtype=torch.long)
        self.register_buffer("metric_ids", mid, persistent=False)

    def forward(self, m: torch.Tensor):
        """
        m: [B, M]
        return metric_tokens: [B, M, d]
        """
        B, M = m.shape
        assert M == self.num_metrics
        ids = self.metric_ids.unsqueeze(0).expand(B, M)
        return self.metric_id_emb(ids) + self.value_proj(m.unsqueeze(-1))