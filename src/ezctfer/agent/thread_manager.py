"""
双线程管理模块
用于管理两个并行解题线程的启动、停止和线程间通信
"""

import threading
from typing import Callable, Optional, Tuple

from ..config.log import log_info, log_warning, log_error, log_success, log_separator
from ..tools.tools import (
    init_dual_thread_mode,
    is_dual_thread_mode,
    is_stop_requested,
    get_stop_signal,
    clear_dual_thread_state,
    clear_state,
    build_double_ctrl_c_handler,
)


class ThreadManager:
    """
    双线程管理器
    管理两个并行解题线程的启动、停止和线程间通信
    """
    
    def __init__(self):
        self.thread1: threading.Thread | None = None
        self.thread2: threading.Thread | None = None
        self.result1: Tuple[bool, str] = (False, "")
        self.result2: Tuple[bool, str] = (False, "")
        self._lock = threading.Lock()
    
    def init_dual_mode(self) -> None:
        """初始化双线程模式"""
        init_dual_thread_mode()
        clear_state()
        clear_dual_thread_state()
    
    def _run_solver(self, solver_func: Callable, thread_id: int, thread_name: str) -> None:
        """
        在线程中运行解题器
        
        Args:
            solver_func: 解题函数
            thread_id: 线程ID
            thread_name: 线程名称
        """
        try:
            result = solver_func()
            with self._lock:
                if thread_id == 1:
                    self.result1 = result
                else:
                    self.result2 = result
        except Exception as e:
            log_error(f"线程 {thread_name} 执行出错: {e}")
            import traceback
            traceback.print_exc()
            with self._lock:
                if thread_id == 1:
                    self.result1 = (False, f"线程执行出错: {str(e)}")
                else:
                    self.result2 = (False, f"线程执行出错: {str(e)}")
    
    def start_threads(self, solver_func1: Callable, solver_func2: Callable) -> None:
        """
        启动两个解题线程
        
        Args:
            solver_func1: 线程1的解题函数
            solver_func2: 线程2的解题函数
        """
        log_separator()
        log_info("🚀 启动双线程解题模式")
        log_separator()
        
        # 创建线程
        self.thread1 = threading.Thread(
            target=self._run_solver,
            args=(solver_func1, 1, "线程1"),
            name="CTF-Solver-Thread-1"
        )
        self.thread2 = threading.Thread(
            target=self._run_solver,
            args=(solver_func2, 2, "线程2"),
            name="CTF-Solver-Thread-2"
        )
        
        # 设置为守护线程（主线程退出时自动结束）
        self.thread1.daemon = True
        self.thread2.daemon = True
        
        # 启动线程
        self.thread1.start()
        self.thread2.start()
        
        log_success("双线程已启动")
    
    def wait_for_completion(self, timeout: float | None = None) -> Tuple[Tuple[bool, str], Tuple[bool, str]]:
        """
        等待线程完成
        
        Args:
            timeout: 超时时间（秒），None表示无限等待
            
        Returns:
            (线程1结果, 线程2结果)，每个结果为 (是否成功, flag或总结)
        """
        import signal

        def request_stop():
            # 设置停止信号
            stop_signal = get_stop_signal()
            if stop_signal:
                stop_signal.set()

        signal_handler = build_double_ctrl_c_handler(
            on_first_interrupt=request_stop,
            first_message="检测到 Ctrl+C，正在停止所有线程... 再按一次将强制结束程序。",
            second_message="再次检测到 Ctrl+C，正在强制结束程序...",
        )

        old_handler = signal.signal(signal.SIGINT, signal_handler)
        
        try:
            # 使用循环 join 方式，每隔 0.5 秒检查一次
            while True:
                # 检查线程是否都结束
                t1_alive = self.thread1 and self.thread1.is_alive()
                t2_alive = self.thread2 and self.thread2.is_alive()
                
                if not t1_alive and not t2_alive:
                    break
                
                # 等待一小段时间，让信号有机会被处理
                if t1_alive:
                    self.thread1.join(timeout=0.5)
                if t2_alive:
                    self.thread2.join(timeout=0.5)
        finally:
            # 恢复原来的信号处理器
            signal.signal(signal.SIGINT, old_handler)
        
        return self.result1, self.result2
    
    def is_any_thread_found_flag(self) -> bool:
        """检查是否有线程找到了flag"""
        return self.result1[0] or self.result2[0]
    
    def get_winning_result(self) -> Tuple[bool, str]:
        """
        获取获胜结果（找到flag的那个线程的结果）
        
        Returns:
            (是否成功, flag或总结)
        """
        if self.result1[0]:
            return self.result1
        elif self.result2[0]:
            return self.result2
        else:
            # 都没找到flag，返回线程1的结果（或合并）
            return self.result1


# 全局线程管理器实例
_thread_manager: ThreadManager | None = None


def get_thread_manager() -> ThreadManager:
    """获取全局线程管理器实例"""
    global _thread_manager
    if _thread_manager is None:
        _thread_manager = ThreadManager()
    return _thread_manager


def run_dual_thread_solve(
    task_description: str,
    max_iterations: int = 120,
    max_rounds: int = 60
) -> Tuple[bool, str]:
    """
    运行双线程解题
    
    Args:
        task_description: 题目描述
        max_iterations: 每个LLM的最大迭代次数
        max_rounds: 最大轮换轮数
        
    Returns:
        (是否成功找到flag, flag内容或总结)
    """
    from .ctf_solver import CTFSolver
    
    manager = get_thread_manager()
    manager.init_dual_mode()
    
    # 创建两个求解器实例
    solver1 = CTFSolver(
        task_description=task_description,
        max_iterations=max_iterations,
        thread_id=1,
        thread_name="线程1"
    )
    solver2 = CTFSolver(
        task_description=task_description,
        max_iterations=max_iterations,
        thread_id=2,
        thread_name="线程2"
    )
    
    # 定义解题函数
    def solve1():
        return solver1.solve(max_rounds=max_rounds)
    
    def solve2():
        return solver2.solve(max_rounds=max_rounds)
    
    # 启动线程
    manager.start_threads(solve1, solve2)
    
    # 等待完成
    result1, result2 = manager.wait_for_completion()
    
    # 返回结果
    # 只有当 flag 非空时才输出成功信息
    if result1[0] and result1[1]:
        log_separator()
        log_info("🎉 线程1成功找到FLAG!")
        log_info(f"🚩 Flag: {result1[1]}")
        log_separator()
        return result1
    elif result2[0] and result2[1]:
        log_separator()
        log_info("🎉 线程2成功找到FLAG!")
        log_info(f"🚩 Flag: {result2[1]}")
        log_separator()
        return result2
    else:
        log_separator()
        log_warning("两个线程都未能找到flag")
        log_separator()
        # 返回线程1的结果（或可以合并两个结果）
        return result1
