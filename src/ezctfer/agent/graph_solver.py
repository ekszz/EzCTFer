"""
GraphSolver - 图模式（Insight-Mission 图探索）多线程调度器。

调度策略：
  - 最开始先运行一次 Reason
  - 每次 Reason 结束后，按当前空闲名额调度最多 2 个 Explorer
  - 每个 Explorer 结束并提交 insight 后，立刻串行触发一次 Reason
  - Reason 和两个 Explorer 各自复用固定 CTFSolver 对象

核心组件：
  - _busy_explorer_slots: 当前正在运行的 explorer slot
  - _explorer_slots_lock: 保护 explorer slot 与调度的互斥锁
  - _reason_lock: 保证任意时刻只有一个 Reason 在运行
  - TaskGraph: missions 存储在图中，每次从图中获取最新的 pending mission
"""

from __future__ import annotations

import threading
import time

from ..config.prompts import (
    EXPLORE_CONCLUDE_HINT,
    EXPLORE_SYSTEM_PROMPT,
    REASON_CONCLUDE_HINT,
    REASON_SYSTEM_PROMPT,
)
from ..config.log import Colors, log_color, log_error, log_info, log_success, log_warning
from ..tools.tools import (
    clear_graph_result,
    get_flag_finder_thread_id,
    get_found_flag,
    get_graph_result,
    get_graph_tools_for_explore,
    get_graph_tools_for_reason,
    init_graph_mode,
    is_flag_found,
    is_result_submitted,
    is_stop_requested,
)
from ..web.monitor import get_monitor
from .ctf_solver import CTFSolver
from .task_graph import TaskGraph

# 每轮 reason 最多提出的新 mission 数量
DEFAULT_MAX_MISSIONS = 3
# 最大并发 explorer 数量
DEFAULT_MAX_EXPLORERS = 2
# Reason 两次执行之间的最小间隔（秒）：若距上次 Reason 不足此时间且仍有 pending missions，则跳过
REASON_MIN_INTERVAL_SECONDS = 10.0


class GraphSolver:
    """
    图模式多线程调度器。
    """

    def __init__(
        self,
        task_description: str,
        max_iterations: int = 100,
        max_rounds: int = 10,
        max_missions_per_round: int = DEFAULT_MAX_MISSIONS,
        max_explorers: int = DEFAULT_MAX_EXPLORERS,
    ):
        self.task_description = task_description
        self.max_iterations = max_iterations
        self.max_rounds = max_rounds
        self.max_missions_per_round = max_missions_per_round
        self.max_explorers = min(max_explorers, DEFAULT_MAX_EXPLORERS)

        init_graph_mode()

        # 监控器用的递增轮次编号
        self._monitor_round_counter = 0
        self._monitor_round_lock = threading.Lock()

        # 图
        self.graph = TaskGraph(origin_desc=task_description)
        self._push_graph_state_to_monitor()

        # 固定角色 solver：Reason 使用 SINGLE_THREAD_LLM，Explorer 使用 DUAL_THREAD_0/1_LLM
        self._reason_solver = CTFSolver(
            self.task_description,
            max_iterations=self.max_iterations,
            thread_id=0,
            thread_name="Reason",
        )
        self._explorer_solvers = {
            1: CTFSolver(
                self.task_description,
                max_iterations=self.max_iterations,
                thread_id=1,
                thread_name="Explorer-1",
            ),
            2: CTFSolver(
                self.task_description,
                max_iterations=self.max_iterations,
                thread_id=2,
                thread_name="Explorer-2",
            ),
        }

        # 并发控制
        self._busy_explorer_slots: set[int] = set()
        self._explorer_slots_lock = threading.Lock()
        self._reason_lock = threading.Lock()
        self._state_event = threading.Event()

        # 标记是否找到 flag
        self._flag_found = False
        self._flag_lock = threading.Lock()

        # Reason 轮次计数
        self._reason_round = 0
        self._reason_round_lock = threading.Lock()

        # Reason 最小间隔控制：记录最近一次 Reason 开始时间
        self._last_reason_start_time: float = 0.0
        self._last_reason_start_time_lock = threading.Lock()

    # ------------------------------------------------------------------
    # 入口
    # ------------------------------------------------------------------

    def solve(self) -> tuple[bool, str]:
        """
        执行图探索。

        Returns:
            (是否找到 flag, flag 内容或图总结)
        """
        log_color("🗺️  启动图模式（Insight-Mission 图探索）- 多线程模式", Colors.CYAN, bold=True)
        log_info(f"📌 题目描述: {self.task_description}")
        log_info(f"🔄 最大 reason 轮数: {self.max_rounds}，最大并发 explorer: {self.max_explorers}")

        # 最开始先运行一次 Reason
        self._run_reason_cycle(blocking=True)
        self._try_schedule_explorers()

        while True:
            if is_stop_requested():
                log_warning("收到停止信号，终止图探索")
                break

            with self._flag_lock:
                if self._flag_found:
                    return True, get_found_flag()

            running_explorers = self._get_running_explorer_count()
            pending_missions = self.graph.pending_missions()
            with self._reason_round_lock:
                reason_exhausted = self._reason_round >= self.max_rounds

            if reason_exhausted and running_explorers == 0:
                log_warning(f"⚠️ 达到最大 reason 轮数 {self.max_rounds}，结束图探索")
                break

            # 图空闲时，再补一次 Reason，保证不会卡在“没有 explorer 也没有 pending mission”的状态
            if running_explorers == 0 and not pending_missions and not reason_exhausted:
                self._run_reason_cycle(blocking=False)

            self._state_event.wait(timeout=0.5)
            self._state_event.clear()

        log_warning("⚠️ 图探索结束")
        return False, self.graph.summary()

    # ------------------------------------------------------------------
    # 监控器辅助
    # ------------------------------------------------------------------

    def _next_monitor_round(self) -> int:
        with self._monitor_round_lock:
            self._monitor_round_counter += 1
            return self._monitor_round_counter

    def _get_running_explorer_count(self) -> int:
        with self._explorer_slots_lock:
            return len(self._busy_explorer_slots)

    def _push_graph_to_monitor(self, phase: str, round_num: int, thread_id: int = 0) -> None:
        try:
            monitor = get_monitor()
            graph_yaml = self.graph.to_yaml()
            title = f"🗺️ 图状态 [{phase}] 第{round_num}步"
            monitor.add_major_finding(
                title=title,
                content=graph_yaml,
                round_num=round_num,
                thread_id=thread_id,
            )
        except Exception:
            pass

    def _push_graph_state_to_monitor(self) -> None:
        try:
            monitor = get_monitor()
            monitor.update_graph_snapshot(self.graph.to_monitor_dict())
        except Exception:
            pass

    # ------------------------------------------------------------------
    # REASON 阶段
    # ------------------------------------------------------------------

    def _should_skip_reason(self) -> bool:
        """
        判断是否应跳过本次 Reason：
        若距上次 Reason 开始时间不足 REASON_MIN_INTERVAL_SECONDS 且仍有 pending missions，则返回 True。
        """
        with self._last_reason_start_time_lock:
            last_start = self._last_reason_start_time
        if last_start == 0.0:
            return False
        elapsed = time.time() - last_start
        has_pending = bool(self.graph.pending_missions())
        if elapsed < REASON_MIN_INTERVAL_SECONDS and has_pending:
            log_info(
                f"⏭️ 距上次 Reason 仅 {elapsed:.1f}s（阈值 {REASON_MIN_INTERVAL_SECONDS}s），"
                f"且仍有 pending missions，跳过本次 Reason"
            )
            return True
        return False

    def _run_reason_cycle(self, blocking: bool = True) -> bool:
        """
        执行一次 Reason 调用。

        Returns:
            是否实际执行了 reason
        """
        with self._flag_lock:
            if self._flag_found:
                return False

        if is_stop_requested():
            return False

        acquired = self._reason_lock.acquire(blocking=blocking)
        if not acquired:
            return False

        try:
            # 记录 Reason 开始时间（用于最小间隔判断）
            with self._last_reason_start_time_lock:
                self._last_reason_start_time = time.time()

            with self._reason_round_lock:
                if self._reason_round >= self.max_rounds:
                    return False
                self._reason_round += 1
                current_round = self._reason_round

            step = self.graph.next_step()
            log_color(f"\n=== Reason 第 {current_round} 轮 (step={step}) ===", Colors.CYAN, bold=True)

            clear_graph_result(self._reason_solver.thread_id)

            system_prompt = self._format_reason_prompt()
            extra_tools = get_graph_tools_for_reason()
            llm_index = self._reason_solver._get_llm_index()
            initial_message = "请分析当前任务图，更新所有待探索方向（含优先级）。"

            monitor_round = self._next_monitor_round()
            try:
                found, _ = self._reason_solver.run_single_llm(
                    llm_index,
                    initial_message=initial_message,
                    round_num=monitor_round,
                    system_prompt_override=system_prompt,
                    extra_tools=extra_tools,
                    conclude_hint=REASON_CONCLUDE_HINT,
                    use_base_tools=False,
                )
            except Exception as exc:
                log_error(f"Reason 阶段执行失败: {exc}")
                self._state_event.set()
                return True

            if found or is_flag_found():
                result = get_graph_result(self._reason_solver.thread_id) or {}
                if result.get("type") == "insight":
                    insight_desc = result.get("description") or get_found_flag()
                else:
                    insight_desc = get_found_flag()
                new_insight = self.graph.add_flag_insight(insight_desc)
                log_success(f"✅ Reason 产出 flag insight: {new_insight.id} - {insight_desc[:80]}")
                with self._flag_lock:
                    self._flag_found = True
                self._state_event.set()
                self._push_graph_state_to_monitor()
                return True

            if not is_result_submitted(self._reason_solver.thread_id):
                log_warning("⚠️ Reason 未提交结构化结果，跳过本轮 mission 更新")
                self._state_event.set()
                return True

            result = get_graph_result(self._reason_solver.thread_id) or {}
            result_type = result.get("type")

            if result_type == "missions":
                missions_data = result.get("missions", [])
                created_missions = self.graph.replace_pending_missions(
                    missions_data[: self.max_missions_per_round]
                )
                log_success(f"✅ Reason 更新了 {len(created_missions)} 个待探索 mission")
                for mission in created_missions:
                    log_info(f"  ➕ {mission.id} (优先级:{mission.priority}): {mission.description[:60]}")
                self._push_graph_state_to_monitor()
                self._push_graph_to_monitor("Reason后", monitor_round)

                if not created_missions:
                    log_warning("⚠️ Reason 未生成任何 mission")
            else:
                log_warning(f"⚠️ Reason 提交了未知类型的结果: {result_type}")

            # Reason 结束后，立刻尝试补充 explorer
            self._try_schedule_explorers()
            self._state_event.set()
            return True
        finally:
            self._reason_lock.release()

    # ------------------------------------------------------------------
    # EXPLORE 调度
    # ------------------------------------------------------------------

    def _try_schedule_explorers(self) -> None:
        """
        尝试调度 Explore 线程。
        """
        while True:
            with self._flag_lock:
                if self._flag_found:
                    return

            with self._explorer_slots_lock:
                if len(self._busy_explorer_slots) >= self.max_explorers:
                    return

                available_slots = [
                    slot for slot in sorted(self._explorer_solvers)
                    if slot <= self.max_explorers and slot not in self._busy_explorer_slots
                ]
                if not available_slots:
                    return

                mission = self.graph.acquire_highest_priority_pending()
                if mission is None:
                    return

                explorer_id = available_slots[0]
                self._busy_explorer_slots.add(explorer_id)

            log_color(
                f"🔍 [Explore-{explorer_id}] 调度探索方向 {mission.id} (优先级:{mission.priority})",
                Colors.BLUE,
            )
            self._push_graph_state_to_monitor()

            explorer_thread = threading.Thread(
                target=self._run_explore_thread,
                args=(mission, explorer_id),
                name=f"Explorer-{explorer_id}-{mission.id}",
                daemon=True,
            )
            try:
                explorer_thread.start()
            except Exception:
                with self._explorer_slots_lock:
                    self._busy_explorer_slots.discard(explorer_id)
                try:
                    self.graph.set_mission_pending(mission.id)
                except (KeyError, ValueError):
                    pass
                self._push_graph_state_to_monitor()
                raise

        self._state_event.set()

    # ------------------------------------------------------------------
    # EXPLORE 阶段（在线程中执行）
    # ------------------------------------------------------------------

    def _run_explore_thread(self, mission, explorer_id: int) -> None:
        explorer_solver = self._explorer_solvers[explorer_id]

        try:
            self._run_explore_single(mission, explorer_id, explorer_solver)
        finally:
            with self._explorer_slots_lock:
                self._busy_explorer_slots.discard(explorer_id)

            self._state_event.set()

            if not is_stop_requested() and not is_flag_found():
                if not self._should_skip_reason():
                    self._run_reason_cycle(blocking=True)

    def _run_explore_single(self, mission, explorer_id: int, explorer_solver: CTFSolver) -> None:
        with self._flag_lock:
            if self._flag_found:
                return

        if is_stop_requested():
            try:
                self.graph.set_mission_pending(mission.id)
                self._push_graph_state_to_monitor()
            except (KeyError, ValueError):
                pass
            return

        clear_graph_result(explorer_solver.thread_id)

        system_prompt = self._format_explore_prompt(mission.description)
        extra_tools = get_graph_tools_for_explore()
        llm_index = explorer_solver._get_llm_index()
        initial_message = (
            f"请围绕以下探索任务进行实际操作，确认一条关键结论或重大发现。\n\n"
            f"探索方向 [{mission.id}] (优先级:{mission.priority}): {mission.description}"
        )

        monitor_round = self._next_monitor_round()
        try:
            get_monitor().set_round_metadata(
                monitor_round,
                explorer_solver.thread_id,
                {"graph_mission_id": mission.id},
            )
        except Exception:
            pass
        log_color(
            f"\n🔎 Explore-{explorer_id} {mission.id} (优先级:{mission.priority}): {mission.description[:80]}",
            Colors.MAGENTA,
        )
        try:
            found, summary = explorer_solver.run_single_llm(
                llm_index,
                initial_message=initial_message,
                round_num=monitor_round,
                system_prompt_override=system_prompt,
                extra_tools=extra_tools,
                conclude_hint=EXPLORE_CONCLUDE_HINT,
            )
        except Exception as exc:
            log_error(f"Explore-{explorer_id} {mission.id} 失败: {exc}")
            try:
                self.graph.set_mission_pending(mission.id)
            except (KeyError, ValueError):
                pass
            self._push_graph_state_to_monitor()
            return

        if found or is_flag_found():
            finder_thread_id = get_flag_finder_thread_id()
            if finder_thread_id is not None and finder_thread_id != explorer_solver.thread_id:
                log_info(
                    f"Explore-{explorer_id} 检测到线程 {finder_thread_id} 已找到 flag，"
                    "跳过 flag insight 收尾并等待 finder 线程完成 writeup",
                    explorer_id,
                )
                self._push_graph_state_to_monitor()
                return

            result = get_graph_result(explorer_solver.thread_id) or {}
            if result.get("type") == "insight":
                insight_desc = result.get("description") or get_found_flag()
            else:
                insight_desc = get_found_flag()

            try:
                new_insight = self.graph.conclude_mission(mission.id, insight_desc)
                log_success(
                    f"✅ Explore-{explorer_id} Mission {mission.id} 产出 flag insight: "
                    f"{new_insight.id} - {insight_desc[:80]}"
                )
                self._push_graph_to_monitor(f"Explore-{explorer_id}后(flag:{mission.id})", monitor_round, explorer_id)
            except (KeyError, ValueError) as exc:
                log_warning(f"⚠️ Explore-{explorer_id} 无法将 flag 挂到 mission {mission.id}: {exc}")

            with self._flag_lock:
                self._flag_found = True
            self._push_graph_state_to_monitor()
            return

        if is_result_submitted(explorer_solver.thread_id):
            result = get_graph_result(explorer_solver.thread_id) or {}
            if result.get("type") == "insight":
                insight_desc = result.get("description") or summary
            else:
                insight_desc = summary
        else:
            insight_desc = (summary or "")[:500] or "（探索未产出明确发现）"

        try:
            new_insight = self.graph.conclude_mission(mission.id, insight_desc)
            log_success(f"✅ Explore-{explorer_id} Mission {mission.id} 结论: {new_insight.id} - {insight_desc[:80]}")
        except (KeyError, ValueError) as exc:
            log_warning(f"⚠️ Explore-{explorer_id} 无法收尾 mission {mission.id}: {exc}")
            try:
                self.graph.set_mission_pending(mission.id)
            except (KeyError, ValueError):
                pass
            self._push_graph_state_to_monitor()

        self._push_graph_to_monitor(f"Explore-{explorer_id}后({mission.id})", monitor_round, explorer_id)
        self._push_graph_state_to_monitor()
        self._state_event.set()

    # ------------------------------------------------------------------
    # 提示词格式化
    # ------------------------------------------------------------------

    def _format_reason_prompt(self) -> str:
        pending_missions = self.graph.pending_missions()
        pending_missions.sort(key=lambda i: (i.priority, i.created_at_step), reverse=True)
        pending_missions_text = "\n".join(
            f"  - {i.id} (优先级:{i.priority}): {i.description}" for i in pending_missions
        ) if pending_missions else "(无)"

        return REASON_SYSTEM_PROMPT.format(
            graph_yaml=self.graph.to_yaml(),
            insight_ids=", ".join(self.graph.insight_ids()) or "ROOT",
            open_missions=pending_missions_text,
            max_missions=self.max_missions_per_round,
        )

    def _format_explore_prompt(self, mission_description: str) -> str:
        return EXPLORE_SYSTEM_PROMPT.format(
            graph_yaml=self.graph.to_yaml_insights_only(),
            mission_description=mission_description,
        )


def solve_ctf_graph(
    task_description: str,
    max_iterations: int = 100,
    max_rounds: int = 10,
    max_explorers: int = DEFAULT_MAX_EXPLORERS,
) -> tuple[bool, str]:
    solver = GraphSolver(
        task_description,
        max_iterations=max_iterations,
        max_rounds=max_rounds,
        max_explorers=max_explorers,
    )
    return solver.solve()
