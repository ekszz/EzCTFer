"""
Tool definitions for CTF Solver
Provides various tools for executing commands, reading/writing files, making HTTP requests, etc.
"""

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import contextvars
from pathlib import Path
from typing import Callable, Optional, Union

import requests
from langchain_core.tools import tool

from ..config.prompts import (
    TOOL_RECORD_FINDING_DESCRIPTION,
    TOOL_SUBMIT_FLAG_DESCRIPTION,
    TOOL_RETRIEVE_KNOWLEDGE_DESCRIPTION,
)
from ..config.log import (
    log_info,
    log_warning,
    log_error,
    log_success,
    log_color,
    log_debug,
    console_print,
    prompt_input,
    Colors,
)
from ..rag import query_knowledge


# 全局状态：存储重大发现（消息内容, 创建者线程ID）
important_info: list[tuple[str, int]] = []

# 全局状态：是否找到flag
flag_found: bool = False

# 全局状态：找到的flag内容
found_flag: str = ""

flag_finder_thread_id: int | None = None

# 全局状态：是否启用 RAG 知识检索
_rag_enabled: bool = False

# 全局状态：是否启用安静模式（找到flag时自动确认）
_quiet_mode: bool = False

# 全局状态：是否禁用 writeup 生成
_no_writeup: bool = False

_SANDBOX_VENV_DIR = Path(__file__).resolve().parents[3] / "sandbox"
_PROMPT_STYLE = f"{Colors.WHITE_BG}{Colors.BLACK}"


def _decode_process_output(output: Optional[bytes]) -> str:
    if not output:
        return ""

    # Windows console output is commonly GBK; Linux/macOS is typically UTF-8.
    encodings = ("gbk", "utf-8") if os.name == "nt" else ("utf-8", "gbk")
    for encoding in encodings:
        try:
            return output.decode(encoding)
        except UnicodeDecodeError:
            continue
    return output.decode(encodings[0], errors="replace")


def _format_process_result(result: subprocess.CompletedProcess) -> str:
    stdout_text = _decode_process_output(result.stdout)
    stderr_text = _decode_process_output(result.stderr)

    output_parts = []
    if stdout_text:
        output_parts.append(f"STDOUT:\n{stdout_text}")
    if stderr_text:
        output_parts.append(f"STDERR:\n{stderr_text}")
    if result.returncode != 0:
        output_parts.append(f"Return code: {result.returncode}")

    return "\n".join(output_parts) if output_parts else "Command executed successfully with no output"


def _get_sandbox_python_path() -> Path:
    if os.name == "nt":
        return _SANDBOX_VENV_DIR / "Scripts" / "python.exe"
    return _SANDBOX_VENV_DIR / "bin" / "python"


def _ensure_sandbox_venv() -> tuple[Optional[str], Optional[str]]:
    sandbox_python = _get_sandbox_python_path()
    if sandbox_python.exists():
        return str(sandbox_python), None

    try:
        _SANDBOX_VENV_DIR.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(["uv", "venv", str(_SANDBOX_VENV_DIR)], capture_output=True, timeout=120)
        if result.returncode != 0:
            return None, f"Error creating sandbox venv.\n{_format_process_result(result)}"
    except subprocess.TimeoutExpired:
        return None, "Error: Creating sandbox venv timed out after 120 seconds"
    except FileNotFoundError:
        return None, "Error: `uv` not found in PATH"
    except Exception as e:
        return None, f"Error creating sandbox venv: {str(e)}"

    if not sandbox_python.exists():
        return None, f"Error: sandbox python not found at {sandbox_python}"

    return str(sandbox_python), None


def set_rag_enabled(enabled: bool) -> None:
    """设置 RAG 知识检索是否启用"""
    global _rag_enabled
    _rag_enabled = enabled


def is_rag_enabled() -> bool:
    """检查 RAG 知识检索是否启用"""
    return _rag_enabled


def set_quiet_mode(quiet: bool) -> None:
    """设置安静模式（找到flag时自动确认，不提示用户）"""
    global _quiet_mode
    _quiet_mode = quiet


def is_quiet_mode() -> bool:
    """检查是否启用安静模式"""
    return _quiet_mode


def set_no_writeup(no_writeup: bool) -> None:
    """设置是否禁用 writeup 生成"""
    global _no_writeup
    _no_writeup = no_writeup


def is_no_writeup() -> bool:
    """检查是否禁用 writeup 生成"""
    return _no_writeup


# ============ 全局停止信号 ============
# 独立于双线程模式，用于 Ctrl+C 优雅退出

# 全局停止信号（用于 Ctrl+C 优雅退出）
_global_stop_signal: Optional[threading.Event] = None
_sigint_press_count: int = 0
_sigint_lock = threading.Lock()
_emergency_exit_callbacks: list[Callable[[], None]] = []
_emergency_exit_lock = threading.Lock()
_shutdown_loggers_silenced = False


def init_global_stop_signal() -> None:
    """初始化全局停止信号"""
    global _global_stop_signal
    _global_stop_signal = threading.Event()


def request_global_stop() -> None:
    """设置全局停止信号（用于 Ctrl+C）"""
    global _global_stop_signal
    if _global_stop_signal:
        _global_stop_signal.set()


def reset_sigint_press_count() -> None:
    """重置 Ctrl+C 次数"""
    global _sigint_press_count
    with _sigint_lock:
        _sigint_press_count = 0


def register_emergency_exit_callback(callback: Callable[[], None]) -> None:
    """注册强制退出前执行的清理函数"""
    with _emergency_exit_lock:
        if callback not in _emergency_exit_callbacks:
            _emergency_exit_callbacks.append(callback)


def run_emergency_exit_callbacks() -> None:
    """执行强制退出前的清理函数"""
    with _emergency_exit_lock:
        callbacks = list(_emergency_exit_callbacks)

    for callback in callbacks:
        try:
            callback()
        except Exception as e:
            log_warning(f"强制退出前清理失败: {e}")


def silence_expected_shutdown_errors() -> None:
    """静音强制退出阶段的预期断连日志"""
    global _shutdown_loggers_silenced
    if _shutdown_loggers_silenced:
        return

    for logger_name in ("mcp.client.sse", "httpx", "httpcore", "httpx_sse"):
        logger = logging.getLogger(logger_name)
        logger.disabled = True
        logger.propagate = False

    _shutdown_loggers_silenced = True


def force_exit_immediately(signum: int = signal.SIGINT) -> None:
    """立即强制退出程序，跳过其它清理逻辑"""
    try:
        silence_expected_shutdown_errors()
        run_emergency_exit_callbacks()
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(128 + signum)


def build_double_ctrl_c_handler(
    on_first_interrupt: Callable[[], None],
    first_message: str,
    second_message: str,
):
    """构建第一次优雅退出、第二次强制退出的 SIGINT 处理器"""
    reset_sigint_press_count()

    def signal_handler(signum, frame):
        global _sigint_press_count

        with _sigint_lock:
            _sigint_press_count += 1
            press_count = _sigint_press_count

        if press_count == 1:
            if first_message:
                log_warning(first_message)
            on_first_interrupt()
            return

        if second_message:
            log_error(second_message)
        force_exit_immediately(signum)

    return signal_handler


# ============ 双线程模式同步机制 ============
# 以下变量仅在双线程模式下启用

# 是否启用双线程模式
_is_dual_thread_mode: bool = False

# 停止信号（当某线程找到flag时通知另一线程停止）
_stop_signal: Optional[threading.Event] = None

# 全局状态锁（保护 important_info、flag_found、found_flag 的并发访问）
_state_lock: Optional[threading.Lock] = None

# 每线程的已读位置（用于追踪 important_info 的读取进度）
# key: thread_id, value: 已读位置（不包含该位置）
_thread_read_pos: dict[int, int] = {}

# 线程ID到线程名称的映射
_thread_names: dict = {}

# 使用 ContextVar 存储当前线程ID（可以在异步/线程池中正确传递）
_current_thread_id_var: contextvars.ContextVar[int] = contextvars.ContextVar('current_thread_id', default=0)


def init_dual_thread_mode() -> None:
    """
    初始化双线程模式
    必须在启动双线程之前调用
    """
    global _is_dual_thread_mode, _stop_signal, _state_lock, _thread_read_pos
    _is_dual_thread_mode = True
    _stop_signal = threading.Event()
    _state_lock = threading.Lock()
    _thread_read_pos = {}


def is_dual_thread_mode() -> bool:
    """检查是否启用了双线程模式"""
    return _is_dual_thread_mode


def register_thread(thread_id: int, thread_name: str = "") -> None:
    """
    注册当前线程（在双线程模式下）
    
    Args:
        thread_id: 线程ID（建议使用 1 或 2）
        thread_name: 线程名称（可选）
    """
    if not _is_dual_thread_mode:
        return
    
    # 设置 ContextVar（可以在异步/线程池中正确传递）
    _current_thread_id_var.set(thread_id)
    
    # 初始化该线程的已读位置为当前 important_info 的长度
    # 这样该线程只会收到注册之后其他线程添加的发现
    if thread_id not in _thread_read_pos:
        _thread_read_pos[thread_id] = len(important_info)
    if thread_name:
        _thread_names[thread_id] = thread_name


def get_unread_findings(thread_id: int) -> list[str]:
    """
    获取指定线程未读的发现消息
    
    Args:
        thread_id: 线程ID
        
    Returns:
        未读的发现消息列表（排除当前线程自己创建的消息），并更新该线程的已读位置
    """
    if not _is_dual_thread_mode:
        return []
    
    if thread_id not in _thread_read_pos:
        return []
    
    read_pos = _thread_read_pos[thread_id]
    current_len = len(important_info)
    
    if read_pos >= current_len:
        return []
    
    # 获取未读的消息，但排除当前线程自己创建的消息
    # important_info 中存储的是 (消息内容, 创建者线程ID) 元组
    unread = [
        msg for msg, creator_thread_id in important_info[read_pos:current_len]
        if creator_thread_id != thread_id
    ]
    
    # 更新已读位置
    _thread_read_pos[thread_id] = current_len
    
    return unread


def get_stop_signal() -> Optional[threading.Event]:
    """获取停止信号"""
    return _stop_signal


def is_stop_requested() -> bool:
    """检查是否收到停止信号（包括全局停止信号和双线程模式停止信号）"""
    # 检查全局停止信号（用于 Ctrl+C）
    if _global_stop_signal is not None and _global_stop_signal.is_set():
        return True
    # 检查双线程模式停止信号
    if _is_dual_thread_mode and _stop_signal is not None and _stop_signal.is_set():
        return True
    return False


def clear_dual_thread_state() -> None:
    """清除双线程模式状态"""
    global _thread_read_pos, _thread_names
    if _stop_signal:
        _stop_signal.clear()
    _thread_read_pos = {}
    _thread_names.clear()


class FlagFoundException(Exception):
    """当找到flag时抛出的异常，用于中断agent执行"""
    def __init__(self, flag: str):
        self.flag = flag
        super().__init__(f"Flag found: {flag}")


def get_important_info() -> list[str]:
    """获取当前的重大发现列表（只返回消息内容，不含线程ID）"""
    return [msg for msg, _ in important_info]


def is_flag_found() -> bool:
    """检查是否已找到flag"""
    return flag_found


def get_found_flag() -> str:
    """获取找到的flag内容"""
    return found_flag


def get_flag_finder_thread_id() -> int | None:
    """获取找到 flag 的线程 ID"""
    return flag_finder_thread_id


def clear_state() -> None:
    """清除全局状态"""
    global important_info, flag_found, found_flag, flag_finder_thread_id
    important_info = []
    flag_found = False
    found_flag = ""
    flag_finder_thread_id = None


@tool
def execute_command(command: str) -> str:
    """Execute a shell command and return the output. Use this for running system commands, scripts, or tools like curl, wget, python, etc."""
    try:
        log_debug(f"🔧 Executing: {command}")
        # Use binary mode to avoid encoding errors, then decode with error handling
        result = subprocess.run(command, shell=True, capture_output=True, timeout=30)
        
        # Decode stdout and stderr with error handling for non-UTF-8 characters
        stdout_text = result.stdout.decode('utf-8', errors='replace') if result.stdout else ""
        stderr_text = result.stderr.decode('utf-8', errors='replace') if result.stderr else ""
        
        output = ""
        if stdout_text:
            output += f"STDOUT:\n{stdout_text}"
        if stderr_text:
            output += f"STDERR:\n{stderr_text}"
        if result.returncode != 0:
            output += f"Return code: {result.returncode}"
        
        return output if output else "Command executed successfully with no output"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds"
    except Exception as e:
        return f"Error executing command: {str(e)}"


@tool
def read_file(path: str) -> str:
    """Read the contents of a file at the given path. Use this to examine source code, configuration files, logs, etc."""
    try:
        log_info(f"📖 Reading file: {path}")
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return content
    except FileNotFoundError:
        return f"Error: File not found at {path}"
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file at the given path. Use this to create scripts, modify configuration files, save outputs, etc. 
    
    IMPORTANT: To avoid file conflicts, always append a random UUID suffix to the filename before the extension. 
    For example: instead of 'script.py', write to 'script_<uuid>.py' (e.g., 'script_a1b2c3d4.py').
    Generate a random 8-character alphanumeric string as the UUID suffix.
    """
    try:
        log_info(f"✏️  Writing to file: {path}")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"


@tool
def list_directory(path: str = ".") -> str:
    """List files and directories at the given path. Use this to explore the file system structure."""
    try:
        log_info(f"📂 Listing directory: {path}")
        entries = os.listdir(path)
        result = "\n".join(entries)
        return result
    except Exception as e:
        return f"Error listing directory: {str(e)}"


@tool
def http_request(url: str, method: str = "GET", headers: str = "", body: str = "", timeout: int = 30) -> str:
    """Make an HTTP request to the given URL. Use this for testing web endpoints, checking responses, etc."""
    try:
        log_debug(f"🌐 HTTP {method} request to: {url}")
        
        # 解析 headers 字符串为字典
        headers_dict = {}
        if headers:
            for header in headers.split("\n"):
                header = header.strip()
                if header and ":" in header:
                    key, value = header.split(":", 1)
                    headers_dict[key.strip()] = value.strip()
        
        # 发送请求
        response = requests.request(
            method=method.upper(),
            url=url,
            headers=headers_dict if headers_dict else None,
            data=body if body else None,
            timeout=timeout,
            allow_redirects=False
        )
        
        # 构建类似 curl -i 的输出格式（包含响应头和响应体）
        output_lines = []
        
        # 状态行
        output_lines.append(f"HTTP/1.1 {response.status_code} {response.reason}")
        
        # 响应头
        for key, value in response.headers.items():
            output_lines.append(f"{key}: {value}")
        
        # 空行分隔
        output_lines.append("")
        
        # 响应体
        if response.content:
            output_lines.append(response.text)
        
        return "\n".join(output_lines)
    
    except requests.exceptions.Timeout:
        return "Error: HTTP request timed out"
    except requests.exceptions.ConnectionError as e:
        return f"Error: Connection failed - {str(e)}"
    except requests.exceptions.RequestException as e:
        return f"Error making HTTP request: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def python_exec(script: str, timeout: int = 30) -> str:
    """Execute Python code in the sandbox uv virtual environment with a timeout (seconds)."""
    if timeout <= 0:
        return "Error: timeout must be greater than 0 seconds"

    sandbox_python, error = _ensure_sandbox_venv()
    if error:
        return error

    try:
        # log_info("Executing Python in sandbox venv")
        command = [sandbox_python, "-c", script]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            output_parts = [f"Error: Python execution timed out after {timeout} seconds and the process was terminated"]
            if stdout:
                output_parts.append(f"Partial STDOUT:\n{_decode_process_output(stdout)}")
            if stderr:
                output_parts.append(f"Partial STDERR:\n{_decode_process_output(stderr)}")
            return "\n".join(output_parts)

        result = subprocess.CompletedProcess(args=command, returncode=process.returncode, stdout=stdout, stderr=stderr)
        return _format_process_result(result)
    except Exception as e:
        return f"Error executing Python script: {str(e)}"


@tool
def python_pip(package: str = "") -> str:
    """Install a package in sandbox uv venv via `uv pip install`.

    Examples:
    - `pwntools`
    - `pwntools==1.0.0`
    """
    if not package.strip():
        return "Error: pip install arguments are required"
    package_args = package.strip()

    sandbox_python, error = _ensure_sandbox_venv()
    if error:
        return error
    command = [
        "uv",
        "pip",
        "install",
        "--python",
        sandbox_python,
        "-i",
        "https://pypi.tuna.tsinghua.edu.cn/simple",
        package_args,
    ]

    try:
        log_info(f"Installing in sandbox venv: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, timeout=300)
        return _format_process_result(result)
    except subprocess.TimeoutExpired:
        return "Error: uv pip install timed out after 300 seconds"
    except FileNotFoundError:
        return "Error: `uv` not found in PATH"
    except Exception as e:
        return f"Error executing uv pip install: {str(e)}"


def _get_thread_id_from_context() -> int:
    """
    获取当前线程ID（优先使用 ContextVar）
    
    优先级：
    1. ContextVar 中存储的值（最可靠，可以跨线程池传递）
    2. 从线程名称推断
    """
    # 首先尝试从 ContextVar 获取
    thread_id = _current_thread_id_var.get()
    if thread_id > 0:
        return thread_id
    
    # 回退：从线程名称推断
    thread_name = threading.current_thread().name
    if "Thread-1" in thread_name or "线程1" in thread_name or thread_name == "线程1":
        return 1
    elif "Thread-2" in thread_name or "线程2" in thread_name or thread_name == "线程2":
        return 2
    
    return 0


@tool
def record_finding(finding: str) -> str:
    """记录解题过程中的重大发现。"""
    global important_info
    
    # 获取当前线程ID（优先使用 ContextVar）
    thread_id = _get_thread_id_from_context()
    
    # 双线程模式下使用锁保护
    if _is_dual_thread_mode and _state_lock:
        with _state_lock:
            # 存储为 (消息内容, 创建者线程ID) 元组
            important_info.append((finding, thread_id))
    else:
        # 非双线程模式也存储为元组格式，保持一致性
        important_info.append((finding, thread_id))
    
    log_color(f"🔍 [重大发现] {finding}", Colors.YELLOW, thread_id, bold=True)
    
    # 同步到Web监控系统
    try:
        from ..web.monitor import get_monitor
        monitor = get_monitor()
        current_round = monitor.get_current_round(thread_id)
        round_num = current_round.round_num if current_round else None
        monitor.add_major_finding(title=finding[:50] + "..." if len(finding) > 50 else finding, 
                                  content=finding, 
                                  round_num=round_num,
                                  thread_id=thread_id)
    except Exception as e:
        # 不影响主流程，静默失败
        pass
    
    # 新方案：不再需要显式通知其他线程
    # 其他线程通过 get_unread_findings() 检查 important_info 的 read_pos 来获取未读消息
    
    return f"Major finding recorded! Please continue your exploration."


# 设置工具描述（从提示词模块导入）
record_finding.description = TOOL_RECORD_FINDING_DESCRIPTION


@tool
def submit_flag(flag: str) -> str:
    """Submit a candidate CTF flag."""
    global flag_found, found_flag, flag_finder_thread_id

    thread_id = _get_thread_id_from_context()

    if _quiet_mode:
        log_color(f"🎉 [FLAG FOUND] {flag}", Colors.GREEN, thread_id, bold=True)

        if _is_dual_thread_mode and _state_lock:
            with _state_lock:
                flag_found = True
                found_flag = flag
                flag_finder_thread_id = thread_id
        else:
            flag_found = True
            found_flag = flag
            flag_finder_thread_id = thread_id

        try:
            from ..web.monitor import get_monitor

            monitor = get_monitor()
            current_round = monitor.get_current_round(thread_id)
            round_num = current_round.round_num if current_round else None
            monitor.add_major_finding(
                title=f"🎉 FLAG FOUND: {flag}",
                content=flag,
                round_num=round_num,
                thread_id=thread_id,
            )
        except Exception:
            pass

        if _is_dual_thread_mode and _stop_signal:
            _stop_signal.set()
            log_info("📣 Notified other threads to stop", thread_id)

        raise FlagFoundException(flag)

    header_lines: list[str] = []

    while True:
        try:
            user_input = prompt_input(
                header_lines,
                f"🚩 Thread {thread_id} found FLAG: {flag} | Enter y (confirm) / n (reject): ",
                style=_PROMPT_STYLE,
            ).strip().lower()

            if user_input == "y":
                console_print("✅ Flag confirmed.")

                if _is_dual_thread_mode and _state_lock:
                    with _state_lock:
                        flag_found = True
                        found_flag = flag
                        flag_finder_thread_id = thread_id
                else:
                    flag_found = True
                    found_flag = flag
                    flag_finder_thread_id = thread_id

                log_color(f"🎉 [FLAG FOUND] {flag}", Colors.GREEN, thread_id, bold=True)

                try:
                    from ..web.monitor import get_monitor

                    monitor = get_monitor()
                    current_round = monitor.get_current_round(thread_id)
                    round_num = current_round.round_num if current_round else None
                    monitor.add_major_finding(
                        title=f"🎉 FLAG FOUND: {flag}",
                        content=flag,
                        round_num=round_num,
                        thread_id=thread_id,
                    )
                except Exception:
                    pass

                if _is_dual_thread_mode and _stop_signal:
                    _stop_signal.set()
                    log_info("📣 Notified other threads to stop", thread_id)

                raise FlagFoundException(flag)

            if user_input == "n":
                console_print("❌ Flag rejected. Continue exploring.")
                log_color(f"⚠️ [FLAG REJECTED] {flag}", Colors.YELLOW, thread_id, bold=True)

                try:
                    from ..web.monitor import get_monitor

                    monitor = get_monitor()
                    current_round = monitor.get_current_round(thread_id)
                    round_num = current_round.round_num if current_round else None
                    monitor.add_major_finding(
                        title=f"❌ FLAG REJECTED: {flag}",
                        content=f"Flag: {flag}\nUser rejected this flag.",
                        round_num=round_num,
                        thread_id=thread_id,
                    )
                except Exception:
                    pass

                return "Flag rejected. Please continue exploring."

            console_print("❓ Invalid input. Please enter y or n.")
        except (EOFError, KeyboardInterrupt):
            console_print("\n❌ Flag rejected due to user interrupt. Continue exploring.")
            return "Flag rejected. Please continue exploring."

submit_flag.description = TOOL_SUBMIT_FLAG_DESCRIPTION


@tool
def retrieve_knowledge(query: str, top_k: int = 3) -> str:
    """从本地 LightRAG 索引中检索与问题相关的知识片段。"""
    return json.dumps(query_knowledge(query, top_k=top_k), ensure_ascii=False, indent=2)


# 设置工具描述（从提示词模块导入）
retrieve_knowledge.description = TOOL_RETRIEVE_KNOWLEDGE_DESCRIPTION


# 基础工具列表（始终可用）
_BASE_TOOLS = [
    execute_command,
    read_file,
    write_file,
    list_directory,
    http_request,
    python_exec,
    python_pip,
    record_finding,
    submit_flag,
]

# Skill 工具
_skill_tool = None


def init_skill_tools_module() -> None:
    """初始化 skill 工具模块"""
    global _skill_tool
    if _skill_tool is None:
        from ..skills.skill_adapter import init_skill_tools, get_skill_tool
        init_skill_tools()
        _skill_tool = get_skill_tool()


def get_tools() -> list:
    """
    获取当前可用的工具列表
    
    根据配置动态返回工具列表：
    - 始终包含基础工具
    - 如果启用了 RAG 则包含 retrieve_knowledge 工具
    - 始终包含 get_skill 工具（用于获取 skill 内容）
    """
    tools = _BASE_TOOLS.copy()
    
    if _rag_enabled:
        tools.append(retrieve_knowledge)
    
    # 确保初始化 skill 工具
    if _skill_tool is None:
        init_skill_tools_module()
    
    if _skill_tool:
        tools.append(_skill_tool)
    
    return tools


# 为了向后兼容，保留 TOOLS 变量（但建议使用 get_tools()）
TOOLS = _BASE_TOOLS
