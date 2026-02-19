"""
制图师 (Cartographer) 关系建模

使用 NetworkX 在本地构建和维护多重有向图：事实中的 entity 为节点，
根据 attribute 建立有向边并记录 relation_type，insights 作为节点高阶属性。
边带 weight：同会话或同场景标签下共现的实体对，其边权重增加，模拟「越常一起出现，联想越强」。
参考：docs/knowledge_base/3_graph_rag.html
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
from loguru import logger


def _node_id(raw: str) -> str:
    """规范化节点 ID，避免空串与空白。"""
    s = (raw or "").strip()
    return s if s else "_empty"


# 共现边的 relation_type，用于「同会话/同标签下一起出现」的联想强化
CO_OCCURRENCE_RELATION = "co_occurrence"


class Cartographer:
    """
    制图师：用 NetworkX MultiDiGraph 维护实体关系图。
    - 节点：来自 facts 的 entity 与 value（作为实体）
    - 有向边：entity -> value，relation_type = attribute，带 weight（默认 1，共现时递增）
    - 同会话或同场景标签下共现的实体对：其之间的边权重增加（模拟「越常一起出现，联想越强」）
    - 节点属性：insights 列表（高阶洞察）
    - 按 member_id 隔离，节点/边均带 member_id 属性
    """

    def __init__(self):
        # 多重有向图：支持同一对节点间多条不同 relation_type 的边
        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()

    def _add_or_increment_edge(
        self,
        u: str,
        v: str,
        relation_type: str,
        member_id: str,
    ) -> Optional[int]:
        """
        添加边 (u, v) 或使已有同 relation_type 的边权重 +1。
        返回边的 key；若 u 或 v 为 _empty 则不操作并返回 None。
        """
        if u == "_empty" or v == "_empty" or u == v:
            return None
        if not self._graph.has_node(u) or not self._graph.has_node(v):
            return None
        edge_data = self._graph.get_edge_data(u, v) or {}
        for key, data in list(edge_data.items()):
            if data.get("relation_type") == relation_type and data.get("member_id") == member_id:
                data["weight"] = data.get("weight", 1) + 1
                return key
        key = self._graph.add_edge(
            u, v,
            relation_type=relation_type,
            member_id=member_id,
            weight=1,
        )
        return key

    def _strengthen_co_occurrence(
        self,
        pairs: Set[Tuple[str, str]],
        member_id: str,
        updates: List[Dict[str, Any]],
    ) -> None:
        """对实体对集合中每一对 (a, b) 增加共现边权重（双向）。"""
        for a, b in pairs:
            k1 = self._add_or_increment_edge(a, b, CO_OCCURRENCE_RELATION, member_id)
            k2 = self._add_or_increment_edge(b, a, CO_OCCURRENCE_RELATION, member_id)
            if k1 is not None or k2 is not None:
                updates.append({
                    "action": "increment_weight",
                    "relation_type": CO_OCCURRENCE_RELATION,
                    "pair": (a, b),
                    "member_id": member_id,
                })

    def update_graph(
        self,
        facts: Optional[List[Dict[str, Any]]] = None,
        insights: Optional[List[Dict[str, Any]]] = None,
        member_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        将本批 facts 与 insights 织入图谱。
        - facts：entity 与 value 作为节点，attribute 作为有向边 relation_type（同关系重复则 weight+1）
        - 同会话：本批中所有实体两两之间共现边权重 +1；同场景标签：同一 context_tag 下实体两两之间共现边权重 +1
        - insights：挂到本批涉及到的节点上作为高阶属性

        Args:
            facts: 事实列表，每项含 entity, attribute, value（及可选 context_tags）
            insights: 洞察列表，每项含 insight, tag, related_goals
            member_id: 成员 ID，用于隔离与写入属性

        Returns:
            本轮图谱更新摘要，如 [{"action": "add_node", "id": "小明"}, ...]
        """
        facts = facts or []
        insights = insights or []
        member_id = member_id or "default"
        updates: List[Dict[str, Any]] = []

        # 本轮涉及到的节点（用于挂载 insights 与共现对）
        touched_nodes: Set[str] = set()
        # 按场景标签聚合实体：tag -> set(entity, value...)
        tag_to_entities: Dict[str, Set[str]] = {}

        for f in facts:
            if not isinstance(f, dict):
                continue
            entity = _node_id(f.get("entity") or "")
            value = _node_id(f.get("value") or "")
            attribute = (f.get("attribute") or "").strip() or "related_to"
            tags = f.get("context_tags") or []
            if not isinstance(tags, list):
                tags = []
            if entity == "_empty":
                continue

            # 节点：entity、value（value 作为实体端点）
            if not self._graph.has_node(entity):
                self._graph.add_node(entity, member_id=member_id, insights=[])
                updates.append({"action": "add_node", "id": entity, "member_id": member_id})
            touched_nodes.add(entity)
            if value != "_empty" and value != entity:
                if not self._graph.has_node(value):
                    self._graph.add_node(value, member_id=member_id, insights=[])
                    updates.append({"action": "add_node", "id": value, "member_id": member_id})
                touched_nodes.add(value)
                # 有向边：entity -[attribute]-> value，带 weight（同关系重复出现则权重递增）
                self._add_or_increment_edge(entity, value, attribute, member_id)
                updates.append({
                    "action": "add_edge",
                    "source": entity,
                    "target": value,
                    "relation_type": attribute,
                    "member_id": member_id,
                })
            # 同场景标签：该事实中的实体归入对应 tag
            for t in tags:
                if isinstance(t, str) and t.strip():
                    tag = t.strip()
                    tag_to_entities.setdefault(tag, set()).add(entity)
                    if value != "_empty" and value != entity:
                        tag_to_entities[tag].add(value)

        # 同会话共现：本批中所有出现过的实体两两之间边权重 +1（联想强化）
        conversation_pairs: Set[Tuple[str, str]] = set()
        nodes_list = [n for n in touched_nodes if n != "_empty"]
        for i, a in enumerate(nodes_list):
            for b in nodes_list[i + 1 :]:
                if a != b:
                    conversation_pairs.add((a, b))
        self._strengthen_co_occurrence(conversation_pairs, member_id, updates)

        # 同场景标签共现：同一 tag 下出现过的实体两两之间边权重 +1
        for tag, entities in tag_to_entities.items():
            entities = [e for e in entities if e != "_empty"]
            tag_pairs: Set[Tuple[str, str]] = set()
            for i, a in enumerate(entities):
                for b in entities[i + 1 :]:
                    if a != b:
                        tag_pairs.add((a, b))
            self._strengthen_co_occurrence(tag_pairs, member_id, updates)

        # 将 insights 挂到本轮涉及到的节点上（作为节点的高阶属性）
        if insights and touched_nodes:
            insight_records = [
                {
                    "insight": i.get("insight") or "",
                    "tag": i.get("tag"),
                    "related_goals": i.get("related_goals") or [],
                }
                for i in insights
                if isinstance(i, dict) and (i.get("insight") or i.get("text"))
            ]
            if not insight_records and insights:
                for i in insights:
                    if isinstance(i, dict):
                        text = i.get("insight") or i.get("text") or ""
                        if text:
                            insight_records.append({
                                "insight": text,
                                "tag": i.get("tag"),
                                "related_goals": i.get("related_goals") or [],
                            })
            for n in touched_nodes:
                if self._graph.has_node(n):
                    existing = self._graph.nodes[n].get("insights") or []
                    if not isinstance(existing, list):
                        existing = []
                    self._graph.nodes[n]["insights"] = existing + insight_records
            updates.append({
                "action": "attach_insights",
                "member_id": member_id,
                "nodes": list(touched_nodes),
                "count": len(insight_records),
            })

        if updates:
            logger.debug(
                "Cartographer update_graph member_id={} facts={} nodes_touched={}",
                member_id,
                len(facts),
                len(touched_nodes),
            )
        return updates

    def save_graph(self, path: str | Path, *, format: str = "json") -> None:
        """
        将图谱持久化到文件。
        - format="json"：使用 NetworkX node-link 格式，便于读写的 JSON
        - format="graphml"：GraphML 格式，便于其他图工具导入
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if format == "graphml":
            # GraphML 对自定义属性支持好，但 list/dict 需序列化
            nx.write_graphml(self._graph, path, encoding="utf-8")
            return
        # json (node-link)
        data = nx.node_link_data(self._graph)
        # 确保 insights 等 list 可序列化
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_graph(self, path: str | Path, *, format: str = "json") -> None:
        """
        从文件加载图谱，覆盖当前内存中的图。
        """
        path = Path(path)
        if not path.exists():
            logger.warning("Cartographer load_graph 文件不存在: {}", path)
            return
        if format == "graphml":
            self._graph = nx.read_graphml(path)
            # read_graphml 返回 DiGraph，转为 MultiDiGraph 以保持接口一致
            if not isinstance(self._graph, nx.MultiDiGraph):
                G = nx.MultiDiGraph(self._graph)
                self._graph = G
            return
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self._graph = nx.node_link_graph(data)

    @property
    def graph(self) -> nx.MultiDiGraph:
        """暴露底层图，供查询或可视化。"""
        return self._graph

    def node_count(self, member_id: Optional[str] = None) -> int:
        """节点数，可选按 member_id 过滤。"""
        if member_id is None:
            return self._graph.number_of_nodes()
        return sum(
            1
            for n, d in self._graph.nodes(data=True)
            if d.get("member_id") == member_id
        )

    def edge_count(self, member_id: Optional[str] = None) -> int:
        """边数，可选按 member_id 过滤。"""
        if member_id is None:
            return self._graph.number_of_edges()
        return sum(
            1
            for _u, _v, d in self._graph.edges(data=True)
            if d.get("member_id") == member_id
        )
