"""
LLM调用监控器
记录所有LLM轮次的对话消息，供Web界面展示
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime
import threading
import uuid


@dataclass
class ChatMessage:
    """单条消息"""
    role: str  # 'user', 'assistant', 'tool', 'system'
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))
    tool_name: str | None = None  # 工具名称（如果是工具调用）
    tool_args: dict | None = None  # 工具参数
    thinking: str | None = None  # 思考内容
    thread_id: int = 0  # 线程ID（用于区分双线程模式下的不同线程）


@dataclass
class MajorFinding:
    """重大发现"""
    id: str
    title: str
    content: str
    round_num: int  # 来源轮次
    thread_id: int = 0  # 来源线程ID
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


@dataclass
class LLMRound:
    """一轮LLM调用"""
    round_num: int
    llm_name: str
    llm_model: str
    messages: list[ChatMessage] = field(default_factory=list)
    start_time: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))
    end_time: str | None = None
    status: str = "running"  # 'running', 'completed', 'error'
    summary: str | None = None
    thread_id: int = 0  # 线程ID（用于区分双线程模式下的不同线程）
    
    def add_message(self, message: ChatMessage) -> None:
        """添加一条消息"""
        self.messages.append(message)
    
    def complete(self, summary: str | None = None) -> None:
        """标记轮次完成"""
        self.end_time = datetime.now().strftime("%H:%M:%S")
        self.status = "completed"
        self.summary = summary
    
    def error(self, error_msg: str) -> None:
        """标记轮次出错"""
        self.end_time = datetime.now().strftime("%H:%M:%S")
        self.status = "error"
        self.summary = error_msg


class LLMMonitor:
    """
    LLM调用监控器
    单例模式，全局唯一
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._rounds: list[LLMRound] = []
                    cls._instance._current_rounds: dict = {}  # 按线程ID存储当前轮次，支持双线程模式
                    cls._instance._major_findings: list[MajorFinding] = []
                    cls._instance._data_lock = threading.Lock()
        return cls._instance
    
    def start_round(self, round_num: int, llm_name: str, llm_model: str, thread_id: int = 0) -> LLMRound:
        """
        开始一个新的LLM轮次
        
        Args:
            round_num: 轮次号
            llm_name: LLM名称
            llm_model: LLM模型名称
            thread_id: 线程ID（双线程模式下使用）
            
        Returns:
            新创建的轮次对象
        """
        with self._data_lock:
            # 检查该线程是否有正在进行的轮次，如果有则先标记完成
            current_round = self._current_rounds.get(thread_id)
            if current_round and current_round.status == "running":
                current_round.complete("被新轮次中断")
            
            # 创建新轮次
            new_round = LLMRound(
                round_num=round_num,
                llm_name=llm_name,
                llm_model=llm_model,
                thread_id=thread_id
            )
            self._rounds.append(new_round)
            self._current_rounds[thread_id] = new_round
            
            return new_round
    
    def add_message(self, message: ChatMessage) -> None:
        """向当前轮次添加消息"""
        with self._data_lock:
            thread_id = message.thread_id
            current_round = self._current_rounds.get(thread_id)
            if current_round:
                current_round.add_message(message)
    
    def complete_current_round(self, summary: str | None = None, thread_id: int = 0) -> None:
        """完成当前轮次"""
        with self._data_lock:
            current_round = self._current_rounds.get(thread_id)
            if current_round:
                current_round.complete(summary)
                if thread_id in self._current_rounds:
                    del self._current_rounds[thread_id]
    
    def error_current_round(self, error_msg: str, thread_id: int = 0) -> None:
        """标记当前轮次出错"""
        with self._data_lock:
            current_round = self._current_rounds.get(thread_id)
            if current_round:
                current_round.error(error_msg)
                if thread_id in self._current_rounds:
                    del self._current_rounds[thread_id]
    
    def get_rounds(self) -> list[dict]:
        """获取所有轮次的数据（用于API返回）"""
        with self._data_lock:
            return [self._round_to_dict(r) for r in self._rounds]
    
    def get_round(self, round_num: int, thread_id: int = 0) -> dict | None:
        """获取指定轮次的数据"""
        with self._data_lock:
            for r in self._rounds:
                if r.round_num == round_num and r.thread_id == thread_id:
                    return self._round_to_dict(r)
            return None
    
    def get_current_round(self, thread_id: int = 0) -> LLMRound | None:
        """获取当前轮次对象"""
        with self._data_lock:
            return self._current_rounds.get(thread_id)
    
    def _round_to_dict(self, r: LLMRound) -> dict:
        """将轮次对象转换为字典"""
        return {
            "round_num": r.round_num,
            "llm_name": r.llm_name,
            "llm_model": r.llm_model,
            "start_time": r.start_time,
            "end_time": r.end_time,
            "status": r.status,
            "summary": r.summary,
            "thread_id": r.thread_id,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp,
                    "tool_name": m.tool_name,
                    "tool_args": m.tool_args,
                    "thinking": m.thinking,
                    "thread_id": m.thread_id
                }
                for m in r.messages
            ]
        }
    
    def add_major_finding(self, title: str, content: str, round_num: int | None = None, thread_id: int = 0) -> MajorFinding:
        """
        添加一个重大发现
        
        Args:
            title: 发现标题
            content: 发现内容
            round_num: 来源轮次（可选）
            thread_id: 线程ID（可选）
            
        Returns:
            新创建的重大发现对象
        """
        with self._data_lock:
            current_round = self._current_rounds.get(thread_id)
            finding = MajorFinding(
                id=str(uuid.uuid4()),
                title=title,
                content=content,
                round_num=round_num or (current_round.round_num if current_round else 0),
                thread_id=thread_id
            )
            self._major_findings.append(finding)
            return finding
    
    def get_major_findings(self) -> list[dict]:
        """获取所有重大发现"""
        with self._data_lock:
            return [
                {
                    "id": f.id,
                    "title": f.title,
                    "content": f.content,
                    "round_num": f.round_num,
                    "thread_id": f.thread_id,
                    "timestamp": f.timestamp
                }
                for f in self._major_findings
            ]
    
    def delete_major_finding(self, finding_id: str) -> bool:
        """删除指定ID的重大发现"""
        with self._data_lock:
            for i, f in enumerate(self._major_findings):
                if f.id == finding_id:
                    self._major_findings.pop(i)
                    return True
            return False
    
    def clear(self) -> None:
        """清除所有数据"""
        with self._data_lock:
            self._rounds.clear()
            self._current_rounds.clear()
            self._major_findings.clear()


# 全局监控器实例
_monitor: LLMMonitor | None = None

# 全局待处理的用户消息（按线程ID存储）
# key: thread_id, value: 消息内容
_pending_human_messages: dict[int, str] = {}
_pending_message_lock = threading.Lock()


def get_monitor() -> LLMMonitor:
    """获取全局监控器实例"""
    global _monitor
    if _monitor is None:
        _monitor = LLMMonitor()
    return _monitor


def clear_monitor() -> None:
    """清除监控器数据"""
    monitor = get_monitor()
    monitor.clear()


def set_pending_human_message(message: str, thread_id: int = 0) -> None:
    """
    设置待处理的用户消息
    该消息将在下一次LLM调用前追加到消息列表中
    
    Args:
        message: 用户输入的消息内容
        thread_id: 目标线程ID（0表示单线程模式或所有线程）
    """
    global _pending_human_messages
    with _pending_message_lock:
        _pending_human_messages[thread_id] = message


def get_pending_human_message(thread_id: int = 0) -> Optional[str]:
    """
    获取并清除指定线程的待处理用户消息
    
    Args:
        thread_id: 线程ID
        
    Returns:
        待处理的用户消息，如果没有则返回None
    """
    global _pending_human_messages
    with _pending_message_lock:
        msg = _pending_human_messages.pop(thread_id, None)
        return msg


def has_pending_human_message(thread_id: int = 0) -> bool:
    """
    检查指定线程是否有待处理的用户消息
    
    Args:
        thread_id: 线程ID
        
    Returns:
        是否有待处理的消息
    """
    with _pending_message_lock:
        return thread_id in _pending_human_messages


def has_any_pending_human_message() -> bool:
    """
    检查是否有任意线程的待处理用户消息
    
    Returns:
        是否有待处理的消息
    """
    with _pending_message_lock:
        return len(_pending_human_messages) > 0
