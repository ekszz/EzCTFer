"""
LTeam 命令行入口点
"""

import argparse
import asyncio
import io
import json
import os
import random
import re
import shutil
import sys
import textwrap
import time
import tomllib
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from .config import init_config
from .config.log import (
    console_print,
    log_info,
    log_debug,
    log_warning,
    log_error,
    log_separator,
    prompt_wait_key,
    Colors,
    set_debug_enabled,
)
from .llm import init_llms, get_llm_manager
from .agent import solve_ctf, CTFSolver
from .agent.thread_manager import run_dual_thread_solve
from .web import start_web_server, clear_monitor
from .mcp_integration.mcp_client import (
    is_ida_pro_mcp_enabled,
    is_idalib_mcp_enabled,
    is_jadx_mcp_enabled,
    load_mcp_tools,
    set_enable_ida_pro_mcp,
    set_enable_jadx_mcp,
    set_enable_idalib_mcp,
)
from .mcp_integration.ida_util import ensure_ida_service, register_cleanup as register_ida_cleanup, stop_ida_service
from .mcp_integration.jadx_util import launch_jadx, register_cleanup as register_jadx_cleanup, stop_jadx
from .rag import close_rag, ensure_rag_runtime_ready, initialize_knowledge_base
from .skills.skill_loader import list_skill_names
from .tools.tools import (
    set_rag_enabled,
    is_rag_enabled,
    init_global_stop_signal,
    request_global_stop,
    set_quiet_mode,
    set_no_writeup,
    init_skill_tools_module,
    build_double_ctrl_c_handler,
    register_emergency_exit_callback,
)


_PROMPT_STYLE = f"{Colors.WHITE_BG}{Colors.BLACK}"
_APP_NAME = "EzCTFer"
_MONITOR_HOST = "localhost"
_MONITOR_PORT = 8000
_PANEL_MIN_WIDTH = 78
_PANEL_MAX_WIDTH = 96
_IDA_ENABLE_SENTINEL = object()
_JADX_ENABLE_SENTINEL = object()
_STARTUP_QUOTES = [
    "WHERE MANY MINDS CONVERGE, THE FLAG REVEALS ITS LIGHT",
    "IN THE CHORUS OF LOGIC, HIDDEN SECRETS LEARN TO SING",
    "WHEN THOUGHTS ALIGN IN SILENCE, EVEN CIPHERS LEAVE A TRACE",
    "FROM FRAGMENTS OF CLUES, A CLEARER CONSTELLATION IS DRAWN",
    "BENEATH THE STATIC OF THE VOID, THE TRUE SIGNAL STARTS TO GLOW",
    "EACH PROMPT IS A LANTERN HELD AGAINST THE DARK OF THE UNKNOWN",
    "WHERE PATIENCE MEETS INGENUITY, LOCKED DOORS REMEMBER THEIR KEYS",
    "THROUGH LAYERS OF SHADOW, THE PATH TO THE FLAG TURNS GOLDEN",
    "IN THE ENGINE OF MANY MINDS, EVEN MYSTERY BEGINS TO YIELD",
    "LET EVERY CLUE BECOME A SPARK, AND EVERY SPARK A WAY FORWARD",
]


def _run_quietly(func, *args, **kwargs):
    """Run a startup step while suppressing transient console noise."""
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        return func(*args, **kwargs)


def _get_panel_width() -> int:
    terminal_width = shutil.get_terminal_size(fallback=(110, 24)).columns
    return max(_PANEL_MIN_WIDTH, min(_PANEL_MAX_WIDTH, terminal_width))


def _format_panel_field(label: str, value: str, width: int) -> list[tuple[str | None, str]]:
    label_width = 10 if label else 0
    available = max(8, width - label_width - (2 if label else 0))
    wrapped = textwrap.wrap(value, width=available) or [""]
    lines = [(label or None, wrapped[0])]
    lines.extend((None, item) for item in wrapped[1:])
    return lines


def _get_mcp_config_path() -> Path:
    config_path = os.getenv("MCP_CONFIG_FILEPATH")
    if config_path and config_path.strip():
        return Path(config_path.strip())
    return Path.cwd() / "mcp.json"


def _get_version() -> str:
    """从 pyproject.toml 读取版本号"""
    try:
        # 尝试从项目根目录的 pyproject.toml 读取
        pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
        if pyproject_path.exists():
            with pyproject_path.open("rb") as f:
                data = tomllib.load(f)
                return data.get("project", {}).get("version", "0.0.0")
    except Exception:
        pass
    
    # 如果读取失败，回退到 __version__
    try:
        from . import __version__
        return __version__
    except Exception:
        return "0.0.0"


__version__ = _get_version()


def _resolve_ida_cli_mode(ida_option: object) -> tuple[bool, str | None]:
    """Resolve CLI flags into either IDA Pro mode or idalib mode."""
    if ida_option is _IDA_ENABLE_SENTINEL:
        return True, None

    if isinstance(ida_option, str):
        ida_option = ida_option.strip()
        if ida_option:
            return False, ida_option
        return True, None

    return False, None


def _resolve_jadx_cli_mode(jadx_option: object) -> tuple[bool, str | None]:
    """Resolve CLI flags into either plain JADX mode or JADX-launch mode."""
    if jadx_option is _JADX_ENABLE_SENTINEL:
        return True, None

    if isinstance(jadx_option, str):
        jadx_option = jadx_option.strip()
        if jadx_option:
            return True, jadx_option
        return True, None

    return False, None


def _collect_mcp_status() -> tuple[int, dict[str, bool]]:
    statuses = {
        "ida_pro_mcp": is_ida_pro_mcp_enabled(),
        "idalib_mcp": is_idalib_mcp_enabled(),
        "jadx_mcp": is_jadx_mcp_enabled(),
    }

    try:
        with _get_mcp_config_path().open("r", encoding="utf-8") as file:
            config = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return 0, statuses

    servers = config.get("mcpServers", {})
    enabled_count = 0
    for server_name in servers:
        if server_name == "ida_pro_mcp":
            enabled_count += int(statuses["ida_pro_mcp"])
        elif server_name == "idalib_mcp":
            enabled_count += int(statuses["idalib_mcp"])
        elif server_name == "jadx_mcp":
            enabled_count += int(statuses["jadx_mcp"])
        else:
            enabled_count += 1

    return enabled_count, statuses


def _format_selected_llm(manager, original_index: int | None) -> str:
    if original_index is None:
        return "RANDOM"

    llm = manager.get_llm_by_original_index(original_index)
    if llm is None:
        return "RANDOM"

    return f"{llm.name} [{llm.config.model}]"


def _build_selected_llm_summary(config, manager, dual_thread: bool) -> str:
    if dual_thread:
        thread_1 = _format_selected_llm(manager, config.dual_thread_0_llm)
        thread_2 = _format_selected_llm(manager, config.dual_thread_1_llm)
        return f"T1: {thread_1} | T2: {thread_2}"

    return _format_selected_llm(manager, config.single_thread_llm)


def _style_panel_value(label: str | None, value: str, static_color: str, dynamic_color: str) -> str:
    if not value:
        return ""

    if label == "MODE":
        return f"{dynamic_color}{value}{Colors.RESET}"

    if label == "LLMS":
        parts = value.split(" ", 1)
        if len(parts) == 2:
            return f"{dynamic_color}{parts[0]}{Colors.RESET} {static_color}{parts[1]}{Colors.RESET}"
        return f"{dynamic_color}{value}{Colors.RESET}"

    if label == "LLM LIST":
        return f"{dynamic_color}{value}{Colors.RESET}"

    if label == "ROUTING":
        styled_parts = []
        for chunk in value.split(" | "):
            if ": " in chunk:
                prefix, current_value = chunk.split(": ", 1)
                styled_parts.append(
                    f"{static_color}{prefix}: {Colors.RESET}{dynamic_color}{current_value}{Colors.RESET}"
                )
            else:
                styled_parts.append(f"{dynamic_color}{chunk}{Colors.RESET}")
        return f"{static_color} | {Colors.RESET}".join(styled_parts)

    if label == "SKILLS":
        parts = value.split(" ", 1)
        if len(parts) == 2 and parts[0].isdigit():
            return f"{dynamic_color}{parts[0]}{Colors.RESET} {static_color}{parts[1]}{Colors.RESET}"
        return f"{dynamic_color}{value}{Colors.RESET}"

    if label == "MCP":
        styled_parts = []
        for chunk in re.split(r"(\d+)", value):
            if not chunk:
                continue
            color = dynamic_color if chunk.isdigit() else static_color
            styled_parts.append(f"{color}{chunk}{Colors.RESET}")
        return "".join(styled_parts)

    if label is None and (": ON" in value or ": OFF" in value):
        styled_parts = []
        for chunk in value.split(" | "):
            if ": " in chunk:
                prefix, current_value = chunk.split(": ", 1)
                styled_parts.append(
                    f"{static_color}{prefix}: {Colors.RESET}{dynamic_color}{current_value}{Colors.RESET}"
                )
            else:
                styled_parts.append(f"{dynamic_color}{chunk}{Colors.RESET}")
        return f"{static_color} | {Colors.RESET}".join(styled_parts)

    if label == "HTTP MON":
        return f"{dynamic_color}{value}{Colors.RESET}"

    if label == "RAG":
        return f"{static_color}{value}{Colors.RESET}"

    return f"{dynamic_color}{value}{Colors.RESET}"


def _print_startup_panel(
    *,
    dual_thread: bool,
    configured_llm_count: int,
    llm_labels: list[str],
    selected_llm_summary: str,
    skills_summary: str,
    mcp_tool_count: int,
    monitor_summary: str,
) -> None:
    panel_width = _get_panel_width()
    inner_width = panel_width - 4
    left_width = 20
    right_width = inner_width - left_width - 3

    mcp_count, mcp_status = _collect_mcp_status()
    # 合并 ida_pro_mcp 和 idalib_mcp 为 ida-mcp，任意一个为 ON 则显示 ON
    ida_mcp_on = mcp_status["ida_pro_mcp"] or mcp_status["idalib_mcp"]
    jadx_mcp_on = mcp_status["jadx_mcp"]
    mcp_flags_text = " | ".join([
        f"ida-mcp: {'ON' if ida_mcp_on else 'OFF'}",
        f"jadx-mcp: {'ON' if jadx_mcp_on else 'OFF'}",
    ])

    right_lines: list[tuple[str | None, str]] = []
    right_lines.append(("__TITLE__", random.choice(_STARTUP_QUOTES)))
    right_lines.append((None, ""))
    right_lines.extend(
        _format_panel_field("MODE", "DUAL-THREAD" if dual_thread else "SINGLE-THREAD", right_width)
    )
    right_lines.extend(_format_panel_field("LLMS", f"{configured_llm_count} configured", right_width))
    right_lines.extend(
        _format_panel_field("LLM LIST", ", ".join(llm_labels) if llm_labels else "none", right_width)
    )
    right_lines.extend(_format_panel_field("ROUTING", selected_llm_summary, right_width))
    right_lines.extend(_format_panel_field("SKILLS", skills_summary, right_width))
    right_lines.extend(_format_panel_field("MCP", f"{mcp_count} enabled, {mcp_tool_count} tools", right_width))
    right_lines.extend(_format_panel_field("", mcp_flags_text, right_width))
    right_lines.extend(_format_panel_field("HTTP MON", monitor_summary, right_width))
    if is_rag_enabled():
        right_lines.extend(_format_panel_field("RAG", "enabled", right_width))

    left_lines = [
        " ███████████",
        "░█░░░░░░███ ",
        "░     ███░  ",
        "     ███    ",
        "    ███     ",
        "  ████     █",
        " ███████████",
        "░░░░░░░░░░░ ",
        "",
        f"{_APP_NAME}",
        f"v{__version__}",
    ]

    content_height = max(len(left_lines), len(right_lines))
    left_lines.extend([""] * (content_height - len(left_lines)))
    right_lines.extend([(None, "")] * (content_height - len(right_lines)))

    border_color = f"{Colors.YELLOW}"
    logo_color = f"{Colors.YELLOW}{Colors.BOLD}"
    name_color = f"{Colors.RED}{Colors.BOLD}"
    version_color = f"{Colors.DIM}"
    label_color = f"{Colors.MAGENTA}{Colors.BOLD}"
    value_color = f"{Colors.RED}{Colors.BOLD}"
    accent_color = f"{Colors.RED}{Colors.BOLD}"
    dynamic_value_color = f"{Colors.YELLOW}{Colors.BOLD}"

    top_border = f"{border_color}╭{'─' * (panel_width - 2)}╮{Colors.RESET}"
    bottom_border = f"{border_color}╰{'─' * (panel_width - 2)}╯{Colors.RESET}"

    console_print()
    console_print(top_border)

    for left_text, right_line in zip(left_lines, right_lines):
        label, value = right_line
        if label == "__TITLE__":
            right_plain = value.center(right_width)
            right_styled = f"{accent_color}{right_plain}{Colors.RESET}"
        elif label:
            display_label = f"• {label}"
            right_plain = f"{display_label:<10}  {value}"
            right_styled = (
                f"{label_color}{display_label:<10}{Colors.RESET}  "
                f"{_style_panel_value(label, value, value_color, dynamic_value_color)}"
            )
        else:
            right_plain = f"{'':<10}  {value}"
            right_styled = (
                f"{' ' * 12}"
                f"{_style_panel_value(None, value, value_color, dynamic_value_color)}"
            )

        if left_text.strip().startswith("█"):
            left_styled = f"{logo_color}{left_text.center(left_width)}{Colors.RESET}"
        elif "▄" in left_text or "▀" in left_text:
            left_styled = f"{logo_color}{left_text.center(left_width)}{Colors.RESET}"
        elif left_text.strip().startswith("v"):
            left_styled = f"{version_color}{left_text.center(left_width)}{Colors.RESET}"
        elif left_text.strip():
            left_styled = f"{name_color}{left_text.center(left_width)}{Colors.RESET}"
        else:
            left_styled = " " * left_width

        console_print(
            f"{border_color}│{Colors.RESET} "
            f"{left_styled} {border_color}│{Colors.RESET} "
            f"{right_styled}{' ' * max(0, right_width - len(right_plain))} "
            f"{border_color}│{Colors.RESET}"
        )

    console_print(bottom_border)


def main(dual_thread: bool = False, quiet: bool = False, no_writeup: bool = False, config_path: str | None = None, prompt: str | None = None):
    # 设置安静模式
    set_quiet_mode(quiet)
    # 设置是否禁用 writeup
    set_no_writeup(no_writeup)

    configured_llm_count = 0

    # 1. 初始化配置（从指定的或默认的 .env 文件读取）
    try:
        config = _run_quietly(init_config, config_path)
        configured_llm_count = len(config.llm_configs)
    except Exception as e:
        log_error(f"配置加载失败: {e}")
        return
    
    # 2. 初始化所有 LLM
    try:
        _run_quietly(init_llms)
        manager = get_llm_manager()
        if len(manager) == 0:
            log_error("没有可用的 LLM，请检查配置")
            return
    except Exception as e:
        log_error(f"LLM 初始化失败: {e}")
        return

    llm_labels = [f"{llm.name} [{llm.config.model}]" for llm in manager]
    selected_llm_summary = _build_selected_llm_summary(config, manager, dual_thread)

    # 2.5. 初始化 skill 工具
    skills_summary = "0 loaded"
    try:
        _run_quietly(init_skill_tools_module)
        skills_count = len(list_skill_names())
        skills_summary = f"{skills_count} loaded"
    except Exception as e:
        skills_summary = f"init failed ({e})"

    # 3. 启动WEB监控服务
    monitor_summary = f"http://{_MONITOR_HOST}:{_MONITOR_PORT}"
    try:
        _run_quietly(clear_monitor)  # 清除之前的监控数据
        _run_quietly(start_web_server, host="0.0.0.0", port=_MONITOR_PORT)
    except Exception as e:
        monitor_summary = f"unavailable ({e})"

    mcp_tool_count = 0
    try:
        mcp_tools = _run_quietly(load_mcp_tools)
        mcp_tool_count = len(mcp_tools)
    except Exception as e:
        log_warning(f"预加载 MCP 工具失败，将在运行时按需加载: {e}")

    _print_startup_panel(
        dual_thread=dual_thread,
        configured_llm_count=configured_llm_count,
        llm_labels=llm_labels,
        selected_llm_summary=selected_llm_summary,
        skills_summary=skills_summary,
        mcp_tool_count=mcp_tool_count,
        monitor_summary=monitor_summary,
    )

    # 4. 获取题目描述
    if prompt:
        log_info("📋 Using the challenge description provided via command line:")
        print(prompt)
        task_description = prompt.strip()
    else:
        log_info("Please enter the CTF challenge description (submit an empty line to finish):")
        
        lines = []
        try:
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
        except EOFError:
            pass
        
        task_description = "\n".join(lines).strip()
    
    if not task_description:
        log_error("题目描述不能为空")
        return
    
    # 5. 开始解题
    if dual_thread:
        log_debug("开始解题（双线程模式）...")
    else:
        log_debug("开始解题...")
    
    try:
        if dual_thread:
            # 双线程模式
            success, result = run_dual_thread_solve(
                task_description=task_description,
                max_iterations=config.max_iterations,
                max_rounds=config.max_rounds
            )
        else:
            # 单线程模式：初始化全局停止信号并设置 Ctrl+C 信号处理器
            import signal
            init_global_stop_signal()

            signal_handler = build_double_ctrl_c_handler(
                on_first_interrupt=request_global_stop,
                first_message="检测到 Ctrl+C，正在优雅退出... 再按一次将强制结束程序。",
                second_message="再次检测到 Ctrl+C，正在强制结束程序...",
            )

            old_handler = signal.signal(signal.SIGINT, signal_handler)
            
            try:
                success, result = solve_ctf(
                    task_description=task_description,
                    max_iterations=config.max_iterations,
                    max_rounds=config.max_rounds
                )
            finally:
                # 恢复原来的信号处理器
                signal.signal(signal.SIGINT, old_handler)
        
        log_separator()
        # 只有当 success 为 True 且 result（flag内容）非空时才输出成功信息
        if success and result:
            log_info("🎉 解题成功!")
            log_info(f"🚩 Flag: {result}")
        else:
            log_warning("未能找到flag")
            if result:
                log_info("\n最终总结:")
                print(result)
        log_separator()
        
    except Exception as e:
        log_error(f"解题过程出错: {e}")
        import traceback
        traceback.print_exc()


def run_demo(config_path: str | None = None):
    """
    演示模式 - 展示如何使用CTFSolver
    """
    log_separator()
    log_info("解对解题YES - 多LLM的CTF解题框架 (演示模式)")
    log_separator()
    
    # 初始化
    config = init_config(config_path)
    log_info(f"已加载 {len(config.llm_configs)} 个 LLM 配置")
    
    init_llms()
    manager = get_llm_manager()
    log_info(f"LLM 管理器已初始化，共 {len(manager)} 个 LLM")
    
    if len(manager) == 0:
        log_error("没有可用的 LLM，请检查配置")
        return
    
    # 示例题目
    demo_task = """
这是一道Web安全题目。
目标URL: http://example.com
提示: 可能存在SQL注入漏洞
"""
    
    log_info(f"\n示例题目: {demo_task}")
    
    # 创建solver实例
    solver = CTFSolver(
        task_description=demo_task,
        max_iterations=config.max_iterations
    )
    
    # 开始解题
    success, result = solver.solve(max_rounds=config.max_rounds)
    
    if success:
        log_info(f"\n找到Flag: {result}")
    else:
        log_info(f"\n解题结果: {result}")


def cli_entry():
    """CLI 入口函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="LTeam - 多 LLM CTF 解题框架")
    parser.add_argument("--config", type=str, help="指定配置文件路径 (.env 文件)")
    parser.add_argument("--demo", action="store_true", help="运行演示模式")
    parser.add_argument(
        "--ida",
        nargs="?",
        const=_IDA_ENABLE_SENTINEL,
        metavar="ARGS",
        help="启用 IDA Pro MCP 工具；如果提供 ARGS，则改为启动 idalib MCP 工具，并将 ARGS 附加到 idalib-mcp 命令后",
    )
    parser.add_argument(
        "--jadx",
        nargs="?",
        const=_JADX_ENABLE_SENTINEL,
        metavar="TARGET",
        help="启用 JADX MCP 工具；如果提供 TARGET，则先执行 `jadx-gui TARGET` 启动 JADX",
    )
    parser.add_argument("--rag", action="store_true", help="启用 LightRAG 知识检索工具")
    parser.add_argument("--init-rag", action="store_true", help="初始化 LightRAG 知识库并退出")
    parser.add_argument("--dual-thread", action="store_true", help="启用双线程解题模式")
    parser.add_argument("--debug", action="store_true", help="显示 debug 级别日志")
    parser.add_argument("--prompt", type=str, help="直接提供题目描述，无需手动输入")
    parser.add_argument("--quiet", action="store_true", help="安静模式，找到flag时自动确认，不提示用户")
    parser.add_argument("--no-writeup", action="store_true", help="禁用 writeup 生成")
    args = parser.parse_args()
    set_debug_enabled(args.debug)

    if args.init_rag:
        try:
            initialize_knowledge_base()
            return
        except Exception as e:
            log_error(f"初始化 RAG 知识库时出错: {e}")
            sys.exit(1)

    enable_ida_pro_mcp, idalib_cli_args = _resolve_ida_cli_mode(args.ida)
    enable_jadx_mcp, jadx_cli_target = _resolve_jadx_cli_mode(args.jadx)
    
    # 根据参数设置 IDA Pro MCP 是否启用
    set_enable_ida_pro_mcp(enable_ida_pro_mcp)
    
    # 根据参数设置 JADX MCP 是否启用
    set_enable_jadx_mcp(enable_jadx_mcp)
    
    # 如果启用 idalib，启动 IDA 后台服务并启用 idalib_mcp
    if idalib_cli_args:
        try:
            # 注册退出清理函数，确保程序退出时关闭后台服务
            _run_quietly(register_ida_cleanup)
            register_emergency_exit_callback(stop_ida_service)
            # ensure_ida_service 是异步函数，需要用 asyncio.run 调用
            # 将参数分割为列表，附加到命令后
            ida_extra_args = idalib_cli_args.split()
            success = _run_quietly(
                lambda: asyncio.run(ensure_ida_service(extra_args=ida_extra_args))
            )
            if not success:
                log_error("IDA 后台服务启动失败，程序退出。")
                sys.exit(1)
            # 启用 idalib_mcp
            set_enable_idalib_mcp(True)
        except Exception as e:
            log_error(f"启动 IDA 后台服务时出错: {e}")
            sys.exit(1)

    # 如果为 --jadx 传入目标，则先启动 JADX GUI
    if jadx_cli_target:
        try:
            _run_quietly(register_jadx_cleanup)
            register_emergency_exit_callback(stop_jadx)
            success = _run_quietly(launch_jadx, extra_args=[jadx_cli_target])
            if not success:
                log_error("JADX 启动失败，程序退出。")
                sys.exit(1)
        except Exception as e:
            log_error(f"启动 JADX 时出错: {e}")
            sys.exit(1)
    
    # 如果启用 RAG，则仅校验并启用本地检索工具
    if args.rag:
        try:
            _run_quietly(lambda: ensure_rag_runtime_ready(require_index=True))
            register_emergency_exit_callback(close_rag)
            set_rag_enabled(True)
        except Exception as e:
            log_error(f"启用 RAG 检索工具时出错: {e}")
            sys.exit(1)
    
    # 运行对应模式
    if args.demo:
        run_demo(config_path=args.config)
    else:
        main(dual_thread=args.dual_thread, quiet=args.quiet, no_writeup=args.no_writeup, config_path=args.config, prompt=args.prompt)
    
    # 退出逻辑：如果设置了 --quiet 参数，则等待10秒后自动退出；否则等待用户按键
    if args.quiet:
        print("\n程序将在10秒后自动退出...注意：退出后HTTP服务将无法访问。\n")
        time.sleep(10)
    else:
        prompt_wait_key([], "😉 Press any key to exit (HTTP monitor will stop): ", style=_PROMPT_STYLE)


if __name__ == "__main__":
    cli_entry()
