"""Validate the research graph and render a standalone HTML view."""

import argparse
from collections import defaultdict
from datetime import datetime
from html import escape
import json
from pathlib import Path
import re
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "experiment_tree.json"
PAGE = ROOT / "docs" / "experiment_tree.html"
STATES = {
    "complete": "完成", "partial": "部分完成", "running": "运行中", "waiting": "等待",
    "planned": "计划", "deferred": "暂缓", "failed": "失败", "superseded": "已替代",
    "documented": "已归档", "unknown": "证据不足",
}
RELATIONS = {
    "compares_to": ("比较对照", "作为对照"),
    "derived_from": ("设计继承", "派生试验"),
    "reuses": ("复用", "被复用"),
    "combines": ("组合成分", "被组合"),
    "part_of": ("属于批次", "批次组成"),
    "follow_up_to": ("后续检验的对象", "后续检验"),
    "waits_for": ("等待", "被等待"),
    "waits_for_gpu_lock": ("等待现有 GPU 锁", "快照中持有 GPU 锁"),
    "cites": ("协议引用", "被协议引用"),
}
SOURCE_LABELS = {
    "protocol": "协议", "report": "报告", "archive": "归档", "metrics": "指标",
    "lock": "选择锁", "state": "状态", "log": "日志", "manifest": "清单",
}
PENDING = {"partial", "running", "waiting", "planned", "deferred"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(graph):
    require(graph["schema_version"] == 1, "unsupported graph version")
    for field in ("inventory_as_of", "queue_as_of"):
        require(datetime.fromisoformat(graph[field]).tzinfo, f"{field}: expected a time zone")
    groups = {group["id"] for group in graph["groups"]}
    require(len(groups) == len(graph["groups"]), "duplicate group ID")
    nodes = graph["nodes"]
    ids = {node["id"] for node in nodes}
    require(len(ids) == len(nodes), "duplicate node ID")
    for node in nodes:
        node_id = node["id"]
        require(re.fullmatch(r"[a-z0-9-]+", node_id), f"{node_id}: invalid ID")
        require(node["name"].strip() and node["group"] in groups, f"{node_id}: name/group")
        require(node["kind"] in {"baseline", "experiment", "audit", "archive"}, f"{node_id}: kind")
        require(node["status"] in STATES, f"{node_id}: unsupported status")
        if node["status"] in {"running", "waiting", "partial"}:
            timestamp = datetime.fromisoformat(node["status_as_of"])
            require(timestamp.tzinfo, f"{node_id}: status needs a time zone")
        require(node.get("notes") or node.get("effects"), f"{node_id}: empty details")
        require(node.get("sources"), f"{node_id}: missing evidence references")
        edges = set()
        for edge in node.get("relations", []):
            target = edge["target"]
            identity = edge["type"], target
            require(edge["type"] in RELATIONS, f"{node_id}: unknown relation")
            require(target in ids and target != node_id, f"{node_id}: invalid target {target}")
            require(identity not in edges and edge["scope"].strip(), f"{node_id}: duplicate/empty relation")
            edges.add(identity)
        source_paths = set()
        for source in node["sources"]:
            path = source["path"]
            parts = path.split("/")
            require(source["kind"] in SOURCE_LABELS, f"{node_id}: unknown source kind")
            require(parts[0] in {"docs", "outputs", "data", "configs", "scripts"}
                    and all(parts) and ".." not in parts and not re.search(r"[\\:#?]", path),
                    f"{node_id}: expected repository-relative path, got {path}")
            if parts[0] in {"docs", "configs", "scripts"}:
                require((ROOT / path).is_file(), f"{node_id}: missing repository source {path}")
            source_paths.add(path)
        for effect in node.get("effects", []):
            require(all(key in effect for key in ("metric", "condition", "unit", "source")),
                    f"{node_id}: incomplete effect")
            require(effect["source"] in source_paths, f"{node_id}: effect source is unlisted")
            require(node["status"] not in {"waiting", "planned", "deferred"},
                    f"{node_id}: pending node cannot publish an effect")


def source_link(path, label=None):
    href = quote("../" + path, safe="/")
    return f'<a href="{href}">{escape(label or path)}</a>'


def relation_lists(node, incoming, by_id):
    lists = defaultdict(list)
    for edge in node.get("relations", []):
        lists[RELATIONS[edge["type"]][0]].append((edge["target"], edge["scope"]))
    for source, edge in incoming[node["id"]]:
        lists[RELATIONS[edge["type"]][1]].append((source, edge["scope"]))
    if not lists:
        return ""
    sections = []
    for label, links in lists.items():
        items = "".join(
            f'<li><a href="#{target}">{escape(by_id[target]["name"])}</a>'
            f'<small>{escape(scope)}</small></li>' for target, scope in links
        )
        sections.append(f"<details><summary>{label}</summary><ul>{items}</ul></details>")
    return "<dt>实验关系</dt><dd>" + "".join(sections) + "</dd>"


def effect_table(effects):
    if not effects:
        return ""
    rows = []
    for effect in effects:
        fields = (
            effect["condition"], effect["metric"], effect["unit"], effect.get("control"),
            effect.get("candidate"), effect.get("delta"), effect.get("interval_95"),
        )
        tr = "".join(f"<td>{escape(str(value)) if value is not None else '—'}</td>" for value in fields)
        rows.append(f'<tr>{tr}<td>{source_link(effect["source"], "原表")}</td></tr>')
    return (
        '<dt>效果量</dt><dd class="table"><table><thead><tr><th>比较口径</th><th>指标</th>'
        '<th>单位</th><th>对照</th><th>改动后</th><th>差值</th><th>95%区间</th>'
        '<th>来源</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></dd>"
    )


def render(graph):
    by_id = {node["id"]: node for node in graph["nodes"]}
    incoming = defaultdict(list)
    by_group = defaultdict(list)
    for node in by_id.values():
        by_group[node["group"]].append(node)
        for edge in node.get("relations", []):
            incoming[edge["target"]].append((node["id"], edge))

    branches = []
    pending = "".join(
        f'<li><a href="#{node["id"]}">{escape(node["name"])}</a></li>'
        for node in by_id.values() if node["status"] in PENDING
    )
    if pending:
        branches.append(f"<details class='branch'><summary>进行中与待验证</summary><ul>{pending}</ul></details>")
    for group in graph["groups"]:
        entries = []
        for node in by_group[group["id"]]:
            status = STATES[node["status"]]
            if timestamp := node.get("status_as_of"):
                status += f" · {timestamp}"
            meta = f"<dt>状态</dt><dd>{escape(status)}</dd>"
            if batch := node.get("batch_id"):
                meta += f"<dt>登记 ID</dt><dd><code>{escape(batch)}</code></dd>"
            notes = "".join(
                f"<dt>{escape(label)}</dt><dd>{escape(value)}</dd>" for label, value in node.get("notes", {}).items()
            )
            sources = "".join(
                f'<li>{SOURCE_LABELS[source["kind"]]} · {source_link(source["path"])}</li>'
                for source in node["sources"]
            )
            entries.append(
                f'<details class="node" id="{node["id"]}"><summary>{escape(node["name"])}</summary>'
                f'<dl>{meta}{notes}{effect_table(node.get("effects"))}'
                f'{relation_lists(node, incoming, by_id)}<dt>证据</dt><dd><ul>{sources}</ul></dd></dl></details>'
            )
        branches.append(
            f'<details class="branch"><summary>{escape(group["name"])}</summary>'
            f'<div class="trunk">{"".join(entries)}</div></details>'
        )
    snapshots = (
        f'研究盘点 <time>{escape(graph["inventory_as_of"])}</time>'
        f' · 队列核对 <time>{escape(graph["queue_as_of"])}</time>'
    )
    return TEMPLATE.replace("<!-- SNAPSHOTS -->", snapshots).replace("<!-- TREE -->", "\n".join(branches))


TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>实验改动树</title>
<style>
  :root { color-scheme: light dark; font-family: system-ui, "Microsoft YaHei", sans-serif; }
  body { max-width: 70rem; margin: 2rem auto; padding: 0 1rem; line-height: 1.55; overflow-wrap: anywhere; }
  h1 { margin: 0 0 .35rem; font-size: 1.8rem; }
  nav { margin-bottom: .45rem; }
  nav a { margin-inline-end: .9rem; }
  .snapshots { margin: 0 0 1.25rem; color: GrayText; }
  summary { cursor: pointer; padding: .36rem 0; }
  summary:focus-visible, a:focus-visible { outline: 2px solid Highlight; outline-offset: 2px; }
  .branch > summary, .node[open] > summary { font-weight: 600; }
  .trunk { border-inline-start: 1px solid GrayText; margin-inline-start: .44rem; padding-inline-start: 1.1rem; }
  .node { position: relative; scroll-margin-top: 1rem; }
  .node::before { content: ""; position: absolute; inset: 1.13rem 100% auto auto; width: 1.1rem; border-block-start: 1px solid GrayText; pointer-events: none; }
  dl { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: .45rem .9rem; margin: .4rem 0 1rem; }
  dt { font-weight: 600; }
  dd { margin: 0; }
  dd ul { padding-inline-start: 1.25rem; margin: 0; }
  dd small { display: block; }
  .table { overflow-x: auto; }
  table { border-collapse: collapse; text-align: left; font-variant-numeric: tabular-nums; }
  th, td { padding: .35rem .55rem; border-block-end: 1px solid GrayText; }
  .node:target > summary { text-decoration: underline; }
  @media (max-width: 520px) {
    dl { display: block; }
    dt:not(:first-child) { margin-top: .65rem; }
  }
  @media (pointer: coarse) { summary { box-sizing: border-box; min-block-size: 44px; } }
</style>
</head>
<body>
<main>
<h1>实验改动树</h1>
<nav aria-label="研究资料">
  <a href="INDEX.md">文档导航</a>
  <a href="experiment_tree.json">结构化数据</a>
  <a href="experiment_tree.md">更新说明</a>
  <a href="research_20261119_experiment_comparison_results.md">完整对照报告</a>
</nav>
<p class="snapshots"><!-- SNAPSHOTS --></p>
<!-- TREE -->
</main>
<script>
  function revealNode() {
    const node = document.getElementById(location.hash.slice(1));
    if (!node?.classList.contains("node")) return;
    node.closest(".branch").open = true;
    node.open = true;
    node.querySelector("summary").focus({preventScroll: true});
    node.scrollIntoView({block: "start"});
  }
  addEventListener("hashchange", revealNode);
  revealNode();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate data and check that HTML is current")
    args = parser.parse_args()
    graph = json.loads(DATA.read_text(encoding="utf-8"))
    validate(graph)
    page = render(graph)
    if args.check:
        require(PAGE.is_file() and PAGE.read_text(encoding="utf-8") == page,
                "HTML is stale. Run python scripts/render_experiment_tree.py")
    else:
        PAGE.write_text(page, encoding="utf-8", newline="\n")
    print(f'{len(graph["nodes"])} nodes, {sum(len(node.get("relations", [])) for node in graph["nodes"])} links')


if __name__ == "__main__":
    main()
