"""
TaskGraph - 内存中的 Insight-Mission 图数据结构（不落盘）。

图的核心元素：
- Insight: 已确认的发现（含固定的 "ROOT" 节点）
- Mission: 探索任务，从一个或多个 Insight 出发，产出新的 Insight

图始终按 "insight --mission--> 新insight" 的方式向前推进。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum


ROOT_INSIGHT_ID = "ROOT"


class MissionStatus(str, Enum):
    """Mission 的状态"""
    PENDING = "pending"      # 待探索
    EXPLORING = "exploring"  # 探索中
    CONCLUDED = "concluded"  # 已探索（已有结论 insight）


@dataclass
class Insight:
    """已有发现节点"""
    id: str  # "ROOT" / "I-001" / "I-002" ...
    description: str
    created_at_step: int = 0  # 由哪个 reason/explore 周期产出


@dataclass
class Mission:
    """探索方向（一条边）"""
    id: str  # "M-001" / "M-002" ...
    from_insights: list[str]  # 来源 insight id 列表
    description: str
    priority: int = 5  # 优先级 1-10，10 为最高
    status: MissionStatus = MissionStatus.PENDING  # 状态：pending / exploring / concluded
    to_insight: str | None = None  # 结论 insight id（None 表示尚未完成）
    created_at_step: int = 0


class TaskGraph:
    """
    内存中的 insight-mission 图。

    线程安全说明：
    - 所有读写操作均通过 threading.Lock 保护，支持多线程并行访问。
    """

    def __init__(self, origin_desc: str):
        # 初始只有 ROOT 节点
        self.insights: dict[str, Insight] = {
            ROOT_INSIGHT_ID: Insight(id=ROOT_INSIGHT_ID, description=origin_desc, created_at_step=0),
        }
        self.missions: dict[str, Mission] = {}
        self._insight_counter = 0
        self._mission_counter = 0
        self._step_counter = 0

        # 线程锁，保护所有读写操作
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # 基础 mutator（带锁）
    # ------------------------------------------------------------------

    def next_step(self) -> int:
        """进入下一个 reason/explore 周期，返回新的 step 编号"""
        with self._lock:
            self._step_counter += 1
            return self._step_counter

    def _next_insight_id(self) -> str:
        # 内部方法，调用方需已持有锁
        self._insight_counter += 1
        return f"I-{self._insight_counter:03d}"

    def _next_mission_id(self) -> str:
        # 内部方法，调用方需已持有锁
        self._mission_counter += 1
        return f"M-{self._mission_counter:03d}"

    def add_insight(self, description: str, step: int | None = None) -> Insight:
        """直接添加一个新 insight（不通过 mission）"""
        with self._lock:
            insight_id = self._next_insight_id()
            insight = Insight(
                id=insight_id,
                description=description,
                created_at_step=step if step is not None else self._step_counter,
            )
            self.insights[insight_id] = insight
            return insight

    def add_flag_insight(self, description: str, step: int | None = None) -> Insight:
        """添加最终 flag insight；如果已经存在同内容 flag insight，则复用已有节点。"""
        with self._lock:
            for insight in self.insights.values():
                if insight.description == description:
                    return insight

            insight_id = self._next_insight_id()
            insight = Insight(
                id=insight_id,
                description=description,
                created_at_step=step if step is not None else self._step_counter,
            )
            self.insights[insight_id] = insight
            return insight

    def create_mission(
        self,
        from_insights: list[str],
        description: str,
        priority: int = 5,
        step: int | None = None,
    ) -> Mission:
        """
        创建一条新的探索方向（pending 状态）。

        Args:
            from_insights: 来源 insight id 列表
            description: 探索方向描述
            priority: 优先级 1-10，10 为最高
            step: 所属周期编号
        """
        with self._lock:
            # 过滤掉不存在的 insight
            valid_sources = [
                fid for fid in from_insights
                if fid in self.insights
            ]
            if not valid_sources:
                # 至少以 ROOT 作为兜底来源，避免空 mission
                valid_sources = [ROOT_INSIGHT_ID]

            # 钳制优先级到 [1, 10]
            priority = max(1, min(10, priority))

            mission_id = self._next_mission_id()
            mission = Mission(
                id=mission_id,
                from_insights=valid_sources,
                description=description,
                priority=priority,
                status=MissionStatus.PENDING,
                to_insight=None,
                created_at_step=step if step is not None else self._step_counter,
            )
            self.missions[mission_id] = mission
            return mission

    def conclude_mission(self, mission_id: str, insight_description: str) -> Insight:
        """
        将一个 mission 收尾：产出新 insight，标记为 concluded。
        """
        with self._lock:
            if mission_id not in self.missions:
                raise KeyError(f"Mission {mission_id} not found")
            mission = self.missions[mission_id]
            if mission.to_insight is not None:
                raise ValueError(f"Mission {mission_id} already concluded")

            new_insight = self._add_insight_internal(insight_description)
            mission.to_insight = new_insight.id
            mission.status = MissionStatus.CONCLUDED
            return new_insight

    def set_mission_exploring(self, mission_id: str) -> None:
        """
        将 mission 标记为探索中状态。
        """
        with self._lock:
            if mission_id not in self.missions:
                raise KeyError(f"Mission {mission_id} not found")
            mission = self.missions[mission_id]
            if mission.status != MissionStatus.PENDING:
                raise ValueError(f"Mission {mission_id} is not pending (status: {mission.status})")
            mission.status = MissionStatus.EXPLORING

    def set_mission_pending(self, mission_id: str) -> None:
        """
        将 mission 恢复为待探索状态（探索失败时使用）。
        """
        with self._lock:
            if mission_id not in self.missions:
                raise KeyError(f"Mission {mission_id} not found")
            mission = self.missions[mission_id]
            if mission.status != MissionStatus.EXPLORING:
                raise ValueError(f"Mission {mission_id} is not exploring (status: {mission.status})")
            mission.status = MissionStatus.PENDING

    def _add_insight_internal(self, description: str, step: int | None = None) -> Insight:
        """内部添加 insight（调用方需已持有锁）"""
        insight_id = self._next_insight_id()
        insight = Insight(
            id=insight_id,
            description=description,
            created_at_step=step if step is not None else self._step_counter,
        )
        self.insights[insight_id] = insight
        return insight

    # ------------------------------------------------------------------
    # Mission 更新操作（reason 阶段使用）
    # ------------------------------------------------------------------

    def replace_pending_missions(
        self,
        new_missions_data: list[dict],
        step: int | None = None,
    ) -> list[Mission]:
        """
        替换所有待探索（pending）的 mission，保留 exploring 和 concluded 的。

        Args:
            new_missions_data: 新的 mission 列表，每个元素为 dict:
                {"from": ["I-001"], "description": "...", "priority": 7}
            step: 所属周期编号

        Returns:
            新创建的 mission 列表
        """
        with self._lock:
            # 1. 删除所有 pending 的 mission（保留 exploring 和 concluded）
            to_remove = [
                iid for iid, mission in self.missions.items()
                if mission.status == MissionStatus.PENDING
            ]
            for iid in to_remove:
                del self.missions[iid]

            # 2. 创建新的 mission
            current_step = step if step is not None else self._step_counter
            created_missions = []

            for item in new_missions_data:
                if not isinstance(item, dict):
                    continue
                desc = item.get("description") or ""
                from_insights = item.get("from") or [ROOT_INSIGHT_ID]
                priority = item.get("priority", 5)

                if not isinstance(from_insights, list):
                    from_insights = [from_insights]

                if not desc.strip():
                    continue

                # 钳制优先级到 [1, 10]
                priority = max(1, min(10, priority))

                # 过滤掉不存在的 insight
                valid_sources = [fid for fid in from_insights if fid in self.insights]
                if not valid_sources:
                    valid_sources = [ROOT_INSIGHT_ID]

                mission_id = self._next_mission_id()
                mission = Mission(
                    id=mission_id,
                    from_insights=valid_sources,
                    description=desc.strip(),
                    priority=priority,
                    status=MissionStatus.PENDING,
                    to_insight=None,
                    created_at_step=current_step,
                )
                self.missions[mission_id] = mission
                created_missions.append(mission)

            return created_missions

    # ------------------------------------------------------------------
    # 查询（带锁）
    # ------------------------------------------------------------------

    def pending_missions(self) -> list[Mission]:
        """返回所有待探索的 mission（status == PENDING）"""
        with self._lock:
            return [i for i in self.missions.values() if i.status == MissionStatus.PENDING]

    def exploring_missions(self) -> list[Mission]:
        """返回所有正在探索的 mission（status == EXPLORING）"""
        with self._lock:
            return [i for i in self.missions.values() if i.status == MissionStatus.EXPLORING]

    def concluded_missions(self) -> list[Mission]:
        """返回所有已结论的 mission（status == CONCLUDED）"""
        with self._lock:
            return [i for i in self.missions.values() if i.status == MissionStatus.CONCLUDED]

    def open_missions(self) -> list[Mission]:
        """返回所有未完成的 mission（pending + exploring），向后兼容"""
        with self._lock:
            return [
                i for i in self.missions.values()
                if i.status in (MissionStatus.PENDING, MissionStatus.EXPLORING)
            ]

    def highest_priority_pending_mission(self) -> Mission | None:
        """返回优先级最高的待探索 mission（无则 None）。
        优先级相同时，选择创建时间最新的。"""
        with self._lock:
            pending = [i for i in self.missions.values() if i.status == MissionStatus.PENDING]
            if not pending:
                return None
            # 按 priority 降序，created_at_step 降序排列
            pending.sort(key=lambda i: (i.priority, i.created_at_step), reverse=True)
            return pending[0]

    def acquire_highest_priority_pending(self) -> Mission | None:
        """原子地获取最高优先级 pending mission 并标记为 exploring。

        用于多线程调度：多个 explorer 线程同时调用此方法时，
        不会重复获取同一个 mission。

        Returns:
            获取到的 mission（已标记为 exploring），无则返回 None
        """
        with self._lock:
            pending = [i for i in self.missions.values() if i.status == MissionStatus.PENDING]
            if not pending:
                return None
            # 按 priority 降序，created_at_step 降序排列
            pending.sort(key=lambda i: (i.priority, i.created_at_step), reverse=True)
            mission = pending[0]
            mission.status = MissionStatus.EXPLORING
            return mission

    def newest_open_mission(self) -> Mission | None:
        """返回最新创建的未完成 mission（无则 None）- 保留向后兼容"""
        with self._lock:
            opens = [
                i for i in self.missions.values()
                if i.status in (MissionStatus.PENDING, MissionStatus.EXPLORING)
            ]
            if not opens:
                return None
            return max(opens, key=lambda i: i.created_at_step)

    def insight_ids(self) -> list[str]:
        """可被新 mission 引用的所有 insight id"""
        with self._lock:
            return list(self.insights.keys())

    def get_insight(self, insight_id: str) -> Insight | None:
        with self._lock:
            return self.insights.get(insight_id)

    def get_mission(self, mission_id: str) -> Mission | None:
        with self._lock:
            return self.missions.get(mission_id)

    # ------------------------------------------------------------------
    # 序列化（给 LLM 看，带锁）
    # ------------------------------------------------------------------

    def to_yaml_insights_only(self) -> str:
        """只序列化 insights 部分（给 explorer 用，减少信息过载）"""
        with self._lock:
            lines: list[str] = []

            lines.append("insights:")
            # ROOT 放最前，其余按创建顺序
            ordered_insight_ids = [ROOT_INSIGHT_ID]
            ordered_insight_ids += [
                fid for fid in self.insights.keys() if fid != ROOT_INSIGHT_ID
            ]
            for fid in ordered_insight_ids:
                insight = self.insights.get(fid)
                if not insight:
                    continue
                desc = _indent_block(insight.description, "    ")
                lines.append(f"  {fid}: |")
                lines.append(desc.rstrip())

            return "\n".join(lines)

    def to_yaml(self) -> str:
        """把图序列化为人类/LLM 友好的 YAML 风格文本（包含 insights 和 missions）"""
        with self._lock:
            lines: list[str] = []

            lines.append("insights:")
            # ROOT 放最前，其余按创建顺序
            ordered_insight_ids = [ROOT_INSIGHT_ID]
            ordered_insight_ids += [
                fid for fid in self.insights.keys() if fid != ROOT_INSIGHT_ID
            ]
            for fid in ordered_insight_ids:
                insight = self.insights.get(fid)
                if not insight:
                    continue
                desc = _indent_block(insight.description, "    ")
                lines.append(f"  {fid}: |")
                lines.append(desc.rstrip())

            lines.append("")
            lines.append("missions:")
            if not self.missions:
                lines.append("  []")
            else:
                # 按状态分组，再按优先级排序展示
                sorted_missions = sorted(
                    self.missions.values(),
                    key=lambda i: (
                        0 if i.status == MissionStatus.EXPLORING else 1 if i.status == MissionStatus.PENDING else 2,
                        -i.priority,
                        -i.created_at_step,
                    ),
                )
                for mission in sorted_missions:
                    from_str = ", ".join(mission.from_insights) if mission.from_insights else ROOT_INSIGHT_ID
                    to_str = mission.to_insight if mission.to_insight else "(open)"
                    desc = _indent_block(mission.description, "    ")
                    lines.append(f"  {mission.id}:")
                    lines.append(f"    status: {mission.status.value}")
                    lines.append(f"    priority: {mission.priority}")
                    lines.append(f"    from: [{from_str}]")
                    lines.append(f"    to: {to_str}")
                    lines.append(f"    description: |")
                    lines.append(desc.rstrip())

            return "\n".join(lines)

    def to_monitor_dict(self) -> dict:
        """序列化为 Web Monitor 使用的结构化图数据。"""
        with self._lock:
            insights = [
                {
                    "id": insight.id,
                    "description": insight.description,
                    "created_at_step": insight.created_at_step,
                }
                for insight in self.insights.values()
            ]

            missions = [
                {
                    "id": mission.id,
                    "description": mission.description,
                    "priority": mission.priority,
                    "status": mission.status.value,
                    "from_insights": list(mission.from_insights),
                    "to_insight": mission.to_insight,
                    "created_at_step": mission.created_at_step,
                }
                for mission in self.missions.values()
            ]

            edges = []
            for mission in self.missions.values():
                for from_insight in mission.from_insights or [ROOT_INSIGHT_ID]:
                    edges.append({
                        "from": from_insight,
                        "to": mission.to_insight,
                        "mission_id": mission.id,
                        "status": mission.status.value,
                        "priority": mission.priority,
                    })

            return {
                "insights": insights,
                "missions": missions,
                "edges": edges,
                "step": self._step_counter,
            }

    def summary(self) -> str:
        """生成整张图的最终文字总结（供最终输出用）"""
        with self._lock:
            parts: list[str] = []
            parts.append("=== Insight-Mission 图探索最终总结 ===")
            parts.append(f"insights 数量: {len(self.insights)}（含 ROOT）")
            parts.append(f"missions 数量: {len(self.missions)}")

            pending = [i for i in self.missions.values() if i.status == MissionStatus.PENDING]
            exploring = [i for i in self.missions.values() if i.status == MissionStatus.EXPLORING]
            concluded = [i for i in self.missions.values() if i.status == MissionStatus.CONCLUDED]
            parts.append(f"待探索: {len(pending)}, 探索中: {len(exploring)}, 已完成: {len(concluded)}")
            parts.append("")

            confirmed = [
                f for fid, f in self.insights.items()
                if fid != ROOT_INSIGHT_ID
            ]
            if confirmed:
                parts.append("已确认发现:")
                for i, insight in enumerate(confirmed, 1):
                    parts.append(f"  {i}. [{insight.id}] {insight.description}")
                parts.append("")

            if exploring:
                parts.append("正在探索:")
                for mission in exploring:
                    parts.append(f"  - [{mission.id}] (优先级:{mission.priority}) {mission.description}")
                parts.append("")

            if pending:
                parts.append("待探索方向（按优先级排序）:")
                pending.sort(key=lambda i: (i.priority, i.created_at_step), reverse=True)
                for mission in pending:
                    parts.append(f"  - [{mission.id}] (优先级:{mission.priority}) {mission.description}")
                parts.append("")

            return "\n".join(parts)


def _indent_block(text: str, prefix: str) -> str:
    """给多行文本的每一行加前缀"""
    if not text:
        return prefix.strip()
    return "\n".join(
        (prefix + line if line else prefix.rstrip())
        for line in text.splitlines()
    )
