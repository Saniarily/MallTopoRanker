# src/case_study_runner.py
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np

from .infer_ranker import RankerService


def render_graph_png(service: RankerService, sample_id: str, out_png: str, title: str = ""):
    """
    用节点坐标画拓扑图（论文/汇报可用）。
    依赖：matplotlib + networkx
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        print("Skip graph rendering: please install matplotlib & networkx.")
        return

    g = service.cache_dir / f"{sample_id}.pt"
    data = service._graph_mem_cache.get(sample_id) if hasattr(service, "_graph_mem_cache") and sample_id in service._graph_mem_cache else None
    if data is None:
        import torch
        data = torch.load(g)

    # build networkx
    G = nx.Graph()
    n = data.x.shape[0]
    for i in range(n):
        G.add_node(i)

    edge_index = data.edge_index.cpu().numpy()
    for u, v in edge_index.T:
        if u != v:
            G.add_edge(int(u), int(v))

    # positions from normalized coords in x[:,1:3]
    coords = data.x[:, 1:3].cpu().numpy()
    pos = {i: (float(coords[i, 0]), float(coords[i, 1])) for i in range(n)}

    plt.figure(figsize=(5, 5))
    nx.draw_networkx_edges(G, pos, alpha=0.6, width=1.0)
    nx.draw_networkx_nodes(G, pos, node_size=60, node_color="#4a5568")
    plt.axis("off")
    if title:
        plt.title(title, fontsize=10)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def format_case_md(case_name: str, res: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"# {case_name}")
    lines.append("")
    lines.append("## Query（策划条件）")
    for k, v in res["query"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## 约束与检索范围")
    for c in res.get("constraints", []):
        lines.append(f"- {c}")
    lines.append("")
    lines.append("## Top-K 推荐结果")
    for item in res.get("topk", []):
        lines.append(f"### Rank {item['rank']}: {item['sample_id']}")
        lines.append(f"- 模型匹配分数: {item['score']:.4f}")
        if item.get("total_score_true") is not None:
            lines.append(f"- 样本真实 total_score: {item['total_score_true']}")
        if "metrics" in item:
            lines.append("- 关键指标：")
            for mn, mv in item["metrics"].items():
                lines.append(f"  - {mn}: {mv}")
        expl = item.get("explanations", {})
        if expl:
            lines.append("- 指标级贡献（Top）：")
            for tm in expl.get("top_metrics", []):
                lines.append(f"  - {tm['name']}: {tm['contribution']:.4f}")
            lines.append("- 条件→指标驱动链：")
            for d in expl.get("condition_to_metric", []):
                tops = ", ".join([f"{t['name']}({t['weight']:.3f})" for t in d.get("top_metrics", [])])
                lines.append(f"  - 因为 {d['condition']}，更关注：{tops}")
        lines.append("")
    return "\n".join(lines)


def main(config_path: str = "./config.yaml", checkpoint_path: str = "./outputs/checkpoints/latest.pt", cases_json: str = "./query_cases.json"):
    svc = RankerService(config_path=config_path, checkpoint_path=checkpoint_path)

    cases = json.load(open(cases_json, "r", encoding="utf-8"))
    out_dir = Path("./outputs/cases")
    out_dir.mkdir(parents=True, exist_ok=True)

    for case in cases:
        name = case["name"]
        city_cluster = int(case["city_cluster"])
        q = case["query_features"]
        topk = int(case.get("topk", 10))

        res = svc.recommend(
            query_features=q,
            city_cluster=city_cluster,
            topk=topk,
            hard_filter=True,
            soft_relax=True,     # 案例展示时建议开，避免候选过少
            max_candidates=800,
            return_attention=True
        )

        # save json
        case_id = name.replace(" ", "_")
        with open(out_dir / f"{case_id}.json", "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

        # save md
        md = format_case_md(name, res)
        with open(out_dir / f"{case_id}.md", "w", encoding="utf-8") as f:
            f.write(md)

        # render graph png for Top-3
        for item in res.get("topk", [])[:3]:
            sid = item["sample_id"]
            render_graph_png(svc, sid, str(out_dir / "graphs" / f"{case_id}__{sid}.png"),
                             title=f"{name} | {sid}")

    print("Saved cases to outputs/cases/ (json, md, graphs png)")


if __name__ == "__main__":
    main()
