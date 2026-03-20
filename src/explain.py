import numpy as np

def summarize_condition_metric_attention(attn, M: int, query_feature_names, metric_names):
    """
    attn: torch.Tensor [B, heads, Fq, (M+Nmax)]
    返回：
      cond_importance: [Fq]
      metric_importance: [M]
      cond_to_metric: dict(cond -> list of (metric, weight))
    """
    A = attn.detach().cpu().numpy()
    # 聚合head
    A = A.mean(axis=1)  # [B, Fq, T]
    A_metric = A[:, :, :M]  # [B, Fq, M]

    # batch平均
    A_metric_mean = A_metric.mean(axis=0)  # [Fq, M]

    cond_imp = A_metric_mean.sum(axis=1)  # [Fq]
    metric_imp = A_metric_mean.sum(axis=0)  # [M]

    cond_to_metric = {}
    for i, cname in enumerate(query_feature_names):
        weights = A_metric_mean[i]
        order = np.argsort(-weights)
        cond_to_metric[cname] = [(metric_names[j], float(weights[j])) for j in order]

    return cond_imp, metric_imp, cond_to_metric

def format_explanation_text(query_dict, constraints_text, metric_imp, metric_names, cond_to_metric, topn_metrics=4, topn_drives=2):
    # Top metrics
    order_m = np.argsort(-metric_imp)[:topn_metrics]
    top_metrics = [(metric_names[i], float(metric_imp[i])) for i in order_m]

    # Top conditions
    # 这里用“每个条件对所有指标的注意力总和”当驱动强度
    cond_scores = {c: sum(w for _, w in cond_to_metric[c][:]) for c in cond_to_metric.keys()}
    top_conds = sorted(cond_scores.items(), key=lambda x: -x[1])[:topn_drives]

    lines = []
    lines.append("### 推荐解释")
    lines.append("**策划条件（Query）**：")
    for k, v in query_dict.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("**约束命中**：")
    for t in constraints_text:
        lines.append(f"- {t}")
    lines.append("")
    lines.append("**关键拓扑指标贡献（metric tokens）**：")
    for name, w in top_metrics:
        lines.append(f"- {name}: {w:.4f}")
    lines.append("")
    lines.append("**条件→指标 驱动关系（attention）**：")
    for c, _ in top_conds:
        top_m = cond_to_metric[c][:3]
        lines.append(f"- 因为 **{c}** 的匹配需求更强，模型更关注： " +
                     ", ".join([f"{mn}({mw:.3f})" for mn, mw in top_m]))
    return "\n".join(lines), top_metrics