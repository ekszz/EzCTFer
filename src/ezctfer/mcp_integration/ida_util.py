# IDA MCP服务状态管理
import asyncio
import sys
import subprocess
import psutil
import atexit
import signal
import socket
import os
import time

from ..config.log import log_info, log_warning, log_error, log_success, log_debug

# IDA服务配置
IDA_SERVICE_HOST = "127.0.0.1"
IDA_SERVICE_PORT = 8845
IDA_SERVICE_CMD = ["uv", "run", "idalib-mcp", "--port", str(IDA_SERVICE_PORT)]

_IDA_SERVICE_PID = None
_IDA_SERVICE_LOCK = asyncio.Lock()
_cleanup_registered = False


def is_port_listening(host: str, port: int) -> bool:
    """检查指定端口是否在监听"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        return result == 0
    except socket.error:
        return False
    finally:
        sock.close()


def check_ida_service_status() -> bool:
    """检查IDA服务是否正在运行（通过端口检测）"""
    return is_port_listening(IDA_SERVICE_HOST, IDA_SERVICE_PORT)


async def ensure_ida_service(extra_args: list[str] | None = None):
    """
    确保IDA服务运行（全局单例模式）
    
    Args:
        extra_args: 附加到 IDA_SERVICE_CMD 的额外参数列表
    
    Returns:
        bool: 服务是否成功启动
    """
    global _IDA_SERVICE_PID

    async with _IDA_SERVICE_LOCK:
        # 检查服务是否已经运行
        if check_ida_service_status():
            log_success(f"IDA服务已运行 (端口 {IDA_SERVICE_PORT})。")
            return True

        # 检查全局服务PID是否有效
        if _IDA_SERVICE_PID:
            if psutil.pid_exists(_IDA_SERVICE_PID):
                log_warning(f"IDA服务进程 {_IDA_SERVICE_PID} 存在但端口未监听，尝试重启...")
            else:
                _IDA_SERVICE_PID = None

        # 检查端口占用（只清理非当前进程组的进程）
        port_in_use = is_port_listening(IDA_SERVICE_HOST, IDA_SERVICE_PORT)

        if port_in_use:
            log_warning(f"端口 {IDA_SERVICE_PORT} 已被占用，检查是否为IDA服务进程...")

            # 检查端口占用是否为IDA服务进程
            try:
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        for conn in proc.connections(kind='inet'):
                            if conn.laddr.port == IDA_SERVICE_PORT:
                                cmdline = proc.info.get('cmdline', [])
                                if cmdline and any("idalib-mcp" in str(cmd) for cmd in cmdline):
                                    log_success(f"检测到持久化IDA服务正在运行 (PID: {proc.info['pid']})")
                                    return True
                                else:
                                    log_warning(f"端口被非IDA服务进程占用 PID: {proc.info['pid']}")
                                    return False
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        pass
            except Exception as e:
                log_error(f"检查端口占用进程时出错: {e}")
                return False

        log_info("启动IDA服务...")
        
        # 构建完整的命令（包含额外参数）
        service_cmd = IDA_SERVICE_CMD.copy()
        if extra_args:
            service_cmd.extend(extra_args)
            log_debug(f"附加参数: {' '.join(extra_args)}")

        # 启动服务
        try:
            # 使用 start_new_session=True 替代 preexec_fn=os.setsid 以支持更多平台
            process = subprocess.Popen(
                service_cmd,
                start_new_session=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            _IDA_SERVICE_PID = process.pid
            log_info(f"IDA服务已在后台启动 (PID: {process.pid})。")

            # 等待服务端口监听（带超时）
            for i in range(30):  # 最多等待30秒
                log_info(f"等待IDA服务启动 ({i+1}/30)...")
                if check_ida_service_status():
                    log_success(f"IDA服务已成功启动 (端口 {IDA_SERVICE_PORT})。")
                    return True
                await asyncio.sleep(1)

            log_error("IDA服务启动超时。")
            return False

        except Exception as e:
            log_error(f"IDA服务启动失败: {e}")
            return False


def stop_ida_service():
    """停止IDA服务（同步函数，用于退出时调用）"""
    global _IDA_SERVICE_PID
    
    if _IDA_SERVICE_PID is None:
        return
    
    try:
        if psutil.pid_exists(_IDA_SERVICE_PID):
            proc = psutil.Process(_IDA_SERVICE_PID)
            
            # 尝试终止进程
            try:
                # 在 Windows 上使用 psutil 的 terminate，在 Unix 上尝试终止整个进程组
                if sys.platform == "win32":
                    # Windows: 直接终止进程
                    proc.terminate()
                    log_warning(f"已发送终止信号到IDA服务进程 (PID: {_IDA_SERVICE_PID})")
                else:
                    # Unix: 尝试终止整个进程组
                    try:
                        os.killpg(os.getpgid(_IDA_SERVICE_PID), signal.SIGTERM)
                        log_warning(f"已发送终止信号到IDA服务进程组 (PID: {_IDA_SERVICE_PID})")
                    except (ProcessLookupError, PermissionError, OSError):
                        # 如果进程组终止失败，尝试只终止进程
                        proc.terminate()
                        log_warning(f"已发送终止信号到IDA服务进程 (PID: {_IDA_SERVICE_PID})")
            except Exception as e:
                log_warning(f"发送终止信号时出错: {e}")
                proc.terminate()
            
            # 等待进程结束
            try:
                proc.wait(timeout=5)
                log_success(f"IDA服务进程已停止 (PID: {_IDA_SERVICE_PID})")
            except psutil.TimeoutExpired:
                # 如果进程还在运行，强制杀死
                proc.kill()
                log_success(f"已强制停止IDA服务进程 (PID: {_IDA_SERVICE_PID})")
    except Exception as e:
        log_warning(f"停止IDA服务时出错: {e}")
    finally:
        _IDA_SERVICE_PID = None


def _cleanup_handler():
    """退出清理处理器"""
    stop_ida_service()


def _signal_handler(signum, frame):
    """信号处理器"""
    log_info(f"收到信号 {signum}，正在清理...")
    stop_ida_service()
    sys.exit(0)


def register_cleanup():
    """注册退出清理函数"""
    global _cleanup_registered
    
    if _cleanup_registered:
        return
    
    # 注册正常退出时的清理
    atexit.register(_cleanup_handler)
    
    # 注册信号处理器（处理 Ctrl+C 等）
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    
    _cleanup_registered = True
    log_success("已注册IDA服务退出清理函数")
