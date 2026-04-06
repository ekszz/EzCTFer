"""JADX 启动辅助工具。"""

import atexit
import os
import signal
import socket
import shutil
import subprocess
import sys
import time

import psutil

from ..config.log import log_debug, log_error, log_info, log_success, log_warning

JADX_SERVICE_HOST = "127.0.0.1"
JADX_SERVICE_PORT = 8650
JADX_GUI_CMD = ["jadx-gui"]
_JADX_PROCESS_PID: int | None = None
_cleanup_registered = False


def is_port_listening(host: str, port: int) -> bool:
    """检查指定端口是否在监听。"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        return result == 0
    except socket.error:
        return False
    finally:
        sock.close()


def check_jadx_service_status() -> bool:
    """检查 JADX 服务端口是否已开始监听。"""
    return is_port_listening(JADX_SERVICE_HOST, JADX_SERVICE_PORT)


def _should_use_xvfb() -> bool:
    """在 Linux 无 X11 环境下使用 xvfb-run。"""
    if not sys.platform.startswith("linux"):
        return False

    return not os.environ.get("DISPLAY", "").strip()


def launch_jadx(extra_args: list[str] | None = None) -> bool:
    """启动 JADX GUI。"""
    global _JADX_PROCESS_PID

    launch_cmd = JADX_GUI_CMD.copy()
    if extra_args:
        launch_cmd.extend(extra_args)
        log_debug(f"JADX 附加参数: {' '.join(extra_args)}")

    if _should_use_xvfb():
        xvfb_run = shutil.which("xvfb-run")
        if not xvfb_run:
            log_error("Linux 环境未检测到 X11，且未找到 `xvfb-run`，无法启动 JADX。")
            return False
        launch_cmd = [xvfb_run, "-a", *launch_cmd]
        log_info("Linux 环境未检测到 X11，使用 xvfb-run 启动 JADX。")

    log_info("启动 JADX...")

    popen_kwargs = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "start_new_session": True,
    }

    if sys.platform == "win32":
        creationflags = (
            getattr(subprocess, "DETACHED_PROCESS", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        )
        if creationflags:
            popen_kwargs["creationflags"] = creationflags

    try:
        process = subprocess.Popen(launch_cmd, **popen_kwargs)
        _JADX_PROCESS_PID = process.pid

        for i in range(30):
            time.sleep(1)
            if process.poll() is not None:
                log_error("JADX 进程启动后立即退出。")
                return False

            log_info(f"等待 JADX 服务监听端口 {JADX_SERVICE_PORT} ({i + 1}/30)...")
            if check_jadx_service_status():
                log_success(f"JADX 已启动 (PID: {process.pid}, 端口 {JADX_SERVICE_PORT})。")
                return True

        log_error(f"JADX 启动超时，端口 {JADX_SERVICE_PORT} 未开始监听。")
        return False
    except FileNotFoundError:
        log_error("未找到 `jadx-gui` 命令，请确认 JADX 已安装并加入 PATH。")
        return False
    except Exception as e:
        log_error(f"JADX 启动失败: {e}")
        return False


def stop_jadx() -> None:
    """停止 JADX 进程（同步函数，用于退出时调用）。"""
    global _JADX_PROCESS_PID

    if _JADX_PROCESS_PID is None:
        return

    try:
        if psutil.pid_exists(_JADX_PROCESS_PID):
            proc = psutil.Process(_JADX_PROCESS_PID)

            try:
                if sys.platform == "win32":
                    proc.terminate()
                    log_warning(f"已发送终止信号到 JADX 进程 (PID: {_JADX_PROCESS_PID})")
                else:
                    try:
                        os.killpg(os.getpgid(_JADX_PROCESS_PID), signal.SIGTERM)
                        log_warning(f"已发送终止信号到 JADX 进程组 (PID: {_JADX_PROCESS_PID})")
                    except (ProcessLookupError, PermissionError, OSError):
                        proc.terminate()
                        log_warning(f"已发送终止信号到 JADX 进程 (PID: {_JADX_PROCESS_PID})")
            except Exception as e:
                log_warning(f"发送 JADX 终止信号时出错: {e}")
                proc.terminate()

            try:
                proc.wait(timeout=5)
                log_success(f"JADX 进程已停止 (PID: {_JADX_PROCESS_PID})")
            except psutil.TimeoutExpired:
                proc.kill()
                log_success(f"已强制停止 JADX 进程 (PID: {_JADX_PROCESS_PID})")
    except Exception as e:
        log_warning(f"停止 JADX 时出错: {e}")
    finally:
        _JADX_PROCESS_PID = None


def _cleanup_handler() -> None:
    """退出清理处理器。"""
    stop_jadx()


def _signal_handler(signum, frame) -> None:
    """信号处理器。"""
    log_info(f"收到信号 {signum}，正在清理 JADX...")
    stop_jadx()
    sys.exit(0)


def register_cleanup() -> None:
    """注册退出清理函数。"""
    global _cleanup_registered

    if _cleanup_registered:
        return

    atexit.register(_cleanup_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    _cleanup_registered = True
    log_success("已注册 JADX 退出清理函数")
