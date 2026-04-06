"""
Web应用模块
提供HTTP API和WebSocket服务
"""

import asyncio
import os
import threading
import tomllib
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .monitor import get_monitor, set_pending_human_message, has_pending_human_message, has_any_pending_human_message
from ..mcp_integration.mcp_client import is_ida_pro_mcp_enabled, is_jadx_mcp_enabled, is_idalib_mcp_enabled
from ..tools.tools import is_rag_enabled
from ..skills.skill_loader import list_skill_names


# FastAPI应用实例
app = FastAPI(title="LLM Monitor")


def get_version() -> str:
    """从 pyproject.toml 读取版本号"""
    try:
        pyproject_path = Path(__file__).parent.parent.parent.parent / "pyproject.toml"
        if pyproject_path.exists():
            with pyproject_path.open("rb") as f:
                data = tomllib.load(f)
                return data.get("project", {}).get("version", "0.0.0")
    except Exception:
        pass
    
    # 回退方案：从 __version__ 导入
    try:
        from .. import __version__
        return __version__
    except Exception:
        return "0.0.0"

# WebSocket连接管理
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
    
    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        async with self._lock:
            for connection in self.active_connections:
                try:
                    await connection.send_json(message)
                except:
                    pass


manager = ConnectionManager()


def get_app() -> FastAPI:
    """获取FastAPI应用实例"""
    return app


# 读取HTML模板
def get_html_template() -> str:
    """获取HTML模板内容"""
    template_path = Path(__file__).parent / "templates" / "index.html"
    if template_path.exists():
        return template_path.read_text(encoding="utf-8")
    return "<html><body><h1>Template not found</h1></body></html>"


@app.get("/", response_class=HTMLResponse)
async def index():
    """返回监控页面"""
    return HTMLResponse(content=get_html_template())


@app.get("/api/rounds")
async def get_rounds():
    """获取所有轮次列表"""
    monitor = get_monitor()
    rounds = monitor.get_rounds()
    findings = monitor.get_major_findings()
    return {
        "rounds": [
            {
                "round_num": r["round_num"],
                "llm_name": r["llm_name"],
                "llm_model": r["llm_model"],
                "start_time": r["start_time"],
                "end_time": r["end_time"],
                "status": r["status"],
                "message_count": len(r["messages"]),
                "thread_id": r.get("thread_id", 0)
            }
            for r in rounds
        ],
        "findings_count": len(findings)
    }


@app.get("/api/rounds/{round_num}")
async def get_round_detail(round_num: int, thread_id: int = 0):
    """获取指定轮次的详细信息"""
    monitor = get_monitor()
    round_data = monitor.get_round(round_num, thread_id)
    if round_data is None:
        return {"error": "Round not found"}
    return round_data


@app.get("/api/findings")
async def get_major_findings():
    """获取所有重大发现"""
    monitor = get_monitor()
    findings = monitor.get_major_findings()
    return {"findings": findings}


# 用户消息请求模型
class HumanMessageRequest(BaseModel):
    """用户消息请求"""
    message: str
    thread_id: int = 0  # 目标线程ID，0表示当前选中的线程


@app.post("/api/human-message")
async def send_human_message(request: HumanMessageRequest):
    """
    接收用户输入的消息，存储到全局变量中
    该消息将在下一次LLM调用前追加到消息列表中
    """
    if not request.message or not request.message.strip():
        return {"success": False, "error": "消息内容不能为空"}
    
    set_pending_human_message(request.message.strip(), request.thread_id)
    
    # 广播更新通知有新消息
    await manager.broadcast({
        "type": "human_message",
        "message": request.message.strip(),
        "thread_id": request.thread_id
    })
    
    return {"success": True, "message": "消息已设置，将在下一次LLM调用时追加", "thread_id": request.thread_id}


@app.get("/api/human-message/status")
async def get_human_message_status(thread_id: int = 0):
    """获取当前是否有待处理的用户消息"""
    if thread_id > 0:
        return {"has_pending_message": has_pending_human_message(thread_id), "thread_id": thread_id}
    else:
        return {"has_pending_message": has_any_pending_human_message(), "thread_id": 0}


@app.get("/api/feature-status")
async def get_feature_status():
    """获取功能启用状态"""
    return {
        "ida_enabled": is_ida_pro_mcp_enabled() or is_idalib_mcp_enabled(),
        "jadx_enabled": is_jadx_mcp_enabled(),
        "rag_enabled": is_rag_enabled(),
        "skills_count": len(list_skill_names())
    }


@app.get("/api/version")
async def get_version_api():
    """获取应用版本号"""
    return {"version": get_version()}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket连接处理"""
    await manager.connect(websocket)
    try:
        while True:
            # 保持连接，等待客户端消息
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        await manager.disconnect(websocket)


async def broadcast_update():
    """广播更新到所有WebSocket连接"""
    monitor = get_monitor()
    findings = monitor.get_major_findings()
    await manager.broadcast({
        "type": "update",
        "findings_count": len(findings)
    })


def notify_update():
    """通知更新（非异步调用）"""
    asyncio.run(broadcast_update())


# Web服务器启动函数
def start_web_server(host: str = "0.0.0.0", port: int = 8000):
    """
    在后台线程启动Web服务器
    
    Args:
        host: 监听地址
        port: 监听端口
    """
    import uvicorn
    
    def run_server():
        uvicorn.run(app, host=host, port=port, log_level="warning")
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    return server_thread