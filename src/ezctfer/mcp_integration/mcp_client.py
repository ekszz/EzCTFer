"""
MCP Client - 连接外部 MCP 服务并获取工具
支持 stdio 和 SSE 传输协议

设计原则：
- 工具获取是静态的（一次性），不需要保持连接
- 工具调用时通过闭包捕获的 session 进行通信
- 连接由调用者管理
"""

import asyncio
import json
import logging
import os
import sys
import threading
from _thread import LockType
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Literal, Optional, TextIO

import anyio
from anyio.abc import Process
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from anyio.streams.text import TextReceiveStream

from ..config.log import log_info, log_debug, log_warning, log_error, log_success

logger = logging.getLogger(__name__)


# 模块级配置：是否启用 IDA Pro MCP
_enable_ida_pro_mcp: bool = False

# 模块级配置：是否启用 JADX MCP
_enable_jadx_mcp: bool = False

# 模块级配置：是否启用 idalib_mcp
_enable_idalib_mcp: bool = False

_server_execution_locks: dict[str, LockType] = {}
_server_execution_locks_guard = threading.Lock()


def set_enable_ida_pro_mcp(enable: bool) -> None:
    """
    设置是否启用 IDA Pro MCP
    
    Args:
        enable: True 表示启用 IDA Pro MCP，False 表示禁用
    """
    global _enable_ida_pro_mcp
    _enable_ida_pro_mcp = enable
    # if enable:
    #     log_info("🔧 IDA Pro MCP 已启用")
    # else:
    #     log_info("🔧 IDA Pro MCP 已禁用")


def is_ida_pro_mcp_enabled() -> bool:
    """
    检查是否启用 IDA Pro MCP
    
    Returns:
        True 表示启用，False 表示禁用
    """
    return _enable_ida_pro_mcp


def set_enable_jadx_mcp(enable: bool) -> None:
    """
    设置是否启用 JADX MCP
    
    Args:
        enable: True 表示启用 JADX MCP，False 表示禁用
    """
    global _enable_jadx_mcp
    _enable_jadx_mcp = enable
    # if enable:
    #     log_info("🔧 JADX MCP 已启用")
    # else:
    #     log_info("🔧 JADX MCP 已禁用")


def is_jadx_mcp_enabled() -> bool:
    """
    检查是否启用 JADX MCP
    
    Returns:
        True 表示启用，False 表示禁用
    """
    return _enable_jadx_mcp


def set_enable_idalib_mcp(enable: bool) -> None:
    """
    设置是否启用 idalib_mcp
    
    Args:
        enable: True 表示启用 idalib_mcp，False 表示禁用
    """
    global _enable_idalib_mcp
    _enable_idalib_mcp = enable
    # if enable:
    #     log_info("🔧 idalib_mcp 已启用")
    # else:
    #     log_info("🔧 idalib_mcp 已禁用")


def is_idalib_mcp_enabled() -> bool:
    """
    检查是否启用 idalib_mcp
    
    Returns:
        True 表示启用，False 表示禁用
    """
    return _enable_idalib_mcp

from langchain_core.tools import BaseTool, StructuredTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from pydantic import BaseModel, Field, create_model


class MCPToolWrapper:
    """
    MCP 工具包装器
    将 MCP 工具封装为 LangChain 可识别的 StructuredTool
    关键：保留完整的参数 schema 描述给 LLM
    """
    
    def __init__(
        self,
        tool_name: str,
        tool_description: str,
        input_schema: dict[str, Any],
        call_func: Callable[[dict[str, Any]], Any],
        execution_lock: LockType | None = None,
    ):
        """
        初始化 MCP 工具包装器
        
        Args:
            tool_name: 工具名称
            tool_description: 工具描述
            input_schema: 输入参数的 JSON Schema（包含完整的类型和描述信息）
            call_func: 调用工具的函数
        """
        self.tool_name = tool_name
        self.tool_description = tool_description
        self.input_schema = input_schema
        self.call_func = call_func
        self.execution_lock = execution_lock
        
        # 根据 JSON Schema 创建 Pydantic 模型
        self.args_model = self._create_args_model()
        
        # 创建 LangChain StructuredTool
        self.langchain_tool = self._create_langchain_tool()
    
    def _json_schema_to_python_type(self, schema: dict[str, Any]) -> tuple[type, Any]:
        """将 JSON Schema 类型转换为 Python 类型"""
        json_type = schema.get("type", "string")
        has_default = "default" in schema
        default_value = schema.get("default", ...)
        
        type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }
        
        if "enum" in schema:
            base_type = type_mapping.get(json_type, str)
            return (base_type, default_value) if has_default else (base_type, ...)
        
        if json_type == "object":
            if "properties" in schema:
                nested_model = self._create_nested_model(schema)
                return (nested_model, default_value) if has_default else (nested_model, ...)
            return (dict, default_value) if has_default else (dict, ...)
        
        if json_type == "array":
            if "items" in schema:
                item_schema = schema["items"]
                if item_schema.get("type") == "object" and "properties" in item_schema:
                    item_model = self._create_nested_model(item_schema)
                    return (list[item_model], default_value) if has_default else (list[item_model], ...)
            return (list, default_value) if has_default else (list, ...)
        
        if "anyOf" in schema or "oneOf" in schema:
            # 处理 anyOf/oneOf 类型
            # 尝试找到最合适的类型（优先选择数组类型，因为更具体）
            variants = schema.get("anyOf") or schema.get("oneOf")
            
            # 查找数组类型
            array_type = None
            string_type = None
            other_types = []
            
            for variant in variants:
                variant_type = variant.get("type")
                if variant_type == "array":
                    # 处理数组类型
                    if "items" in variant:
                        item_schema = variant["items"]
                        item_type = type_mapping.get(item_schema.get("type", "string"), str)
                        array_type = list[item_type]
                    else:
                        array_type = list
                elif variant_type == "string":
                    string_type = str
                elif variant_type:
                    other_types.append(type_mapping.get(variant_type, str))
            
            # 优先返回数组类型（如果有），否则返回字符串类型
            # 这样 LLM 会更倾向于使用正确的数组格式
            if array_type:
                return (array_type, default_value) if has_default else (array_type, ...)
            elif string_type:
                return (string_type, default_value) if has_default else (string_type, ...)
            else:
                return (Any, default_value) if has_default else (Any, ...)
        
        python_type = type_mapping.get(json_type, str)
        return (python_type, default_value) if has_default else (python_type, ...)
    
    def _create_nested_model(self, schema: dict[str, Any], model_name: str = "NestedModel") -> type[BaseModel]:
        """根据嵌套的 JSON Schema 创建 Pydantic 模型"""
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        
        fields = {}
        for prop_name, prop_schema in properties.items():
            python_type, default = self._json_schema_to_python_type(prop_schema)
            description = prop_schema.get("description", "")
            
            if prop_name in required:
                if description:
                    fields[prop_name] = (python_type, Field(..., description=description))
                else:
                    fields[prop_name] = (python_type, ...)
            else:
                if description:
                    fields[prop_name] = (python_type, Field(default=default if default is not ... else None, description=description))
                else:
                    fields[prop_name] = (python_type, default if default is not ... else None)
        
        return create_model(model_name, **fields)
    
    def _create_args_model(self) -> type[BaseModel]:
        """根据工具的 input_schema 创建 Pydantic 模型"""
        properties = self.input_schema.get("properties", {})
        required = set(self.input_schema.get("required", []))
        
        fields = {}
        for prop_name, prop_schema in properties.items():
            python_type, default = self._json_schema_to_python_type(prop_schema)
            description = prop_schema.get("description", "")
            
            if prop_name in required:
                if description:
                    fields[prop_name] = (python_type, Field(..., description=description))
                else:
                    fields[prop_name] = (python_type, ...)
            else:
                if description:
                    fields[prop_name] = (python_type, Field(default=default if default is not ... else None, description=description))
                else:
                    fields[prop_name] = (python_type, default if default is not ... else None)
        
        return create_model(f"{self.tool_name}Args", **fields)
    
    def _create_langchain_tool(self) -> BaseTool:
        """创建 LangChain StructuredTool"""
        def invoke_async_call(kwargs: dict[str, Any]) -> str:
            try:
                try:
                    loop = asyncio.get_running_loop()
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, self._async_call(kwargs))
                        return future.result()
                except RuntimeError:
                    return asyncio.run(self._async_call(kwargs))
            except Exception as e:
                return f"Error calling MCP tool {self.tool_name}: {_describe_exception(e)}"

        def sync_call_func(**kwargs) -> str:
            """同步调用 MCP 工具"""
            if self.execution_lock is None:
                return invoke_async_call(kwargs)

            with self.execution_lock:
                return invoke_async_call(kwargs)
        
        full_description = self._build_full_description()
        
        return StructuredTool(
            name=self.tool_name,
            description=full_description,
            func=sync_call_func,
            args_schema=self.args_model,
        )
    
    def _build_full_description(self) -> str:
        """构建完整的工具描述，包含参数说明"""
        description_parts = [self.tool_description]
        
        properties = self.input_schema.get("properties", {})
        required = self.input_schema.get("required", [])
        
        if properties:
            description_parts.append("\n\nArgs:")
            for prop_name, prop_schema in properties.items():
                prop_type = prop_schema.get("type", "any")
                prop_desc = prop_schema.get("description", "")
                is_required = prop_name in required
                
                enum_values = prop_schema.get("enum", [])
                enum_str = f" (enum: {enum_values})" if enum_values else ""
                
                nested_info = ""
                nested_fields_detail = ""
                
                # 处理 anyOf/oneOf 类型
                if "anyOf" in prop_schema or "oneOf" in prop_schema:
                    variants = prop_schema.get("anyOf") or prop_schema.get("oneOf")
                    variant_types = []
                    
                    for variant in variants:
                        v_type = variant.get("type", "any")
                        if v_type == "array" and "items" in variant:
                            items = variant["items"]
                            items_type = items.get("type", "any")
                            
                            # 检查 items 是否也是 anyOf
                            if "anyOf" in items or "oneOf" in items:
                                item_variants = items.get("anyOf") or items.get("oneOf")
                                item_type_names = [iv.get("type", "any") for iv in item_variants]
                                v_type = f"array of ({' or '.join(item_type_names)})"
                            elif items_type == "object" and "properties" in items:
                                # 提取对象字段详情
                                nested_props = items.get("properties", {})
                                nested_required = items.get("required", [])
                                fields_info = []
                                for fk, fv in nested_props.items():
                                    fk_type = fv.get("type", "any")
                                    fk_desc = fv.get("description", "")
                                    fk_req = "required" if fk in nested_required else "optional"
                                    # 检查默认值
                                    if "default" in fv:
                                        fk_desc = f"{fk_desc} (default: {fv['default']})"
                                    fields_info.append(f"{fk}: {fk_type} ({fk_req})" + (f" - {fk_desc}" if fk_desc else ""))
                                nested_fields_detail = f"\n        Object fields: {{{', '.join(fields_info)}}}"
                                v_type = f"array of objects"
                            else:
                                v_type = f"array of {items_type}"
                        elif v_type == "object" and "properties" in variant:
                            # 单个对象类型
                            nested_props = variant.get("properties", {})
                            nested_required = variant.get("required", [])
                            fields_info = []
                            for fk, fv in nested_props.items():
                                fk_type = fv.get("type", "any")
                                fk_desc = fv.get("description", "")
                                fk_req = "required" if fk in nested_required else "optional"
                                if "default" in fv:
                                    fk_desc = f"{fk_desc} (default: {fv['default']})"
                                fields_info.append(f"{fk}: {fk_type} ({fk_req})" + (f" - {fk_desc}" if fk_desc else ""))
                            nested_fields_detail = f"\n        Object fields: {{{', '.join(fields_info)}}}"
                            v_type = "object"
                        variant_types.append(v_type)
                    
                    nested_info = f" (can be: {' or '.join(variant_types)})"
                    # 优先显示数组类型
                    for variant in variants:
                        if variant.get("type") == "array":
                            prop_type = "array"
                            if "items" in variant:
                                items = variant["items"]
                                items_type = items.get("type", "any")
                                if items_type == "object":
                                    nested_info = f" of objects{nested_info}"
                                else:
                                    nested_info = f" of {items_type}{nested_info}"
                            break
                
                elif prop_type == "object" and "properties" in prop_schema:
                    nested_props = prop_schema.get("properties", {})
                    nested_required = prop_schema.get("required", [])
                    fields_info = []
                    for fk, fv in nested_props.items():
                        fk_type = fv.get("type", "any")
                        fk_desc = fv.get("description", "")
                        fk_req = "required" if fk in nested_required else "optional"
                        if "default" in fv:
                            fk_desc = f"{fk_desc} (default: {fv['default']})"
                        fields_info.append(f"{fk}: {fk_type} ({fk_req})" + (f" - {fk_desc}" if fk_desc else ""))
                    nested_fields_detail = f"\n        Fields: {{{', '.join(fields_info)}}}"
                    nested_info = " with fields: " + ", ".join(f"{k} ({nested_props[k].get('type', 'any')})" for k in nested_props)
                
                elif prop_type == "array" and "items" in prop_schema:
                    items = prop_schema["items"]
                    items_type = items.get("type", "any")
                    
                    # 检查 items 是否也是 anyOf
                    if "anyOf" in items or "oneOf" in items:
                        item_variants = items.get("anyOf") or items.get("oneOf")
                        item_type_names = [iv.get("type", "any") for iv in item_variants]
                        nested_info = f" of ({' or '.join(item_type_names)})"
                    elif items_type == "object" and "properties" in items:
                        nested_props = items.get("properties", {})
                        nested_required = items.get("required", [])
                        fields_info = []
                        for fk, fv in nested_props.items():
                            fk_type = fv.get("type", "any")
                            fk_desc = fv.get("description", "")
                            fk_req = "required" if fk in nested_required else "optional"
                            if "default" in fv:
                                fk_desc = f"{fk_desc} (default: {fv['default']})"
                            fields_info.append(f"{fk}: {fk_type} ({fk_req})" + (f" - {fk_desc}" if fk_desc else ""))
                        nested_fields_detail = f"\n        Object fields: {{{', '.join(fields_info)}}}"
                        nested_info = " of objects"
                    else:
                        nested_info = f" of {items_type}"
                
                # 检查是否有默认值并添加到描述
                default_info = ""
                if "default" in prop_schema:
                    default_info = f" (default: {prop_schema['default']})"
                    prop_desc = prop_desc.rstrip(".") + default_info if prop_desc else f"Default: {prop_schema['default']}"
                
                required_str = "required" if is_required else "optional"
                type_str = f"{prop_type}{nested_info}{enum_str}"
                description_parts.append(f"    {prop_name} ({type_str}, {required_str}): {prop_desc}{nested_fields_detail}")
        
        return "\n".join(description_parts)
    
    async def _async_call(self, arguments: dict[str, Any]) -> str:
        """异步调用 MCP 工具"""
        return await self.call_func(arguments)


async def fetch_mcp_tools_from_server(
    server_name: str,
    server_config: dict[str, Any]
) -> list[BaseTool]:
    """
    从单个 MCP 服务器获取工具列表
    
    使用 async with 模式正确管理连接生命周期
    
    Args:
        server_name: 服务器名称
        server_config: 服务器配置
        
    Returns:
        LangChain 工具列表
    """
    tools = []
    
    try:
        command = server_config.get("command")
        args = server_config.get("args", [])
        env = server_config.get("env", {})
        
        if not command:
            log_warning(f"MCP 服务器 {server_name} 缺少 command 配置")
            return []
        
        log_info(f"🔌 正在连接 MCP 服务器: {server_name}")
        log_debug(f"   Command: {command}")
        log_debug(f"   Args: {args}")
        
        merged_env = os.environ.copy()
        merged_env.update(env)
        
        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=merged_env,
        )
        
        # 使用 async with 正确管理生命周期
        async with stdio_client(server_params) as (read_stream, write_stream):
            log_success(f"已连接到 MCP 服务器: {server_name}")
            
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                log_success(f"会话初始化成功: {server_name}")
                
                # 获取工具列表
                result = await session.list_tools()
                log_info(f"📦 从 {server_name} 获取到 {len(result.tools)} 个工具")
                
                for tool in result.tools:
                    input_schema = tool.inputSchema or {"type": "object", "properties": {}}
                    tools.append((tool.name, tool.description, input_schema))
                
                return tools
            
    except Exception as e:
        log_error(f"从 {server_name} 获取工具失败: {e}")
        logger.exception(f"从 {server_name} 获取工具失败")
        return []


def _describe_exception(exc: BaseException) -> str:
    """展开 ExceptionGroup，输出更有用的错误信息。"""
    seen: set[int] = set()
    messages: list[str] = []

    def collect(current: BaseException | None) -> None:
        if current is None:
            return

        obj_id = id(current)
        if obj_id in seen:
            return
        seen.add(obj_id)

        if isinstance(current, BaseExceptionGroup):
            for sub_exc in current.exceptions:
                collect(sub_exc)
            return

        exc_name = type(current).__name__
        exc_text = str(current).strip()
        formatted = f"{exc_name}: {exc_text}" if exc_text else exc_name
        if formatted not in messages:
            messages.append(formatted)

        collect(current.__cause__)
        collect(current.__context__)

    collect(exc)
    return " | ".join(messages) if messages else str(exc)


def _get_server_execution_lock(server_name: str) -> LockType:
    """为同一个 MCP 服务器复用同一把执行锁。"""
    with _server_execution_locks_guard:
        lock = _server_execution_locks.get(server_name)
        if lock is None:
            lock = threading.Lock()
            _server_execution_locks[server_name] = lock
        return lock


def _get_mcp_config_path() -> str:
    """
    获取 MCP 配置文件路径
    
    优先级：
    1. 环境变量 MCP_CONFIG_FILEPATH
    2. 当前工作目录下的 mcp.json
    
    Returns:
        MCP 配置文件路径
    """
    # 优先使用环境变量
    env_path = os.getenv("MCP_CONFIG_FILEPATH")
    if env_path and env_path.strip():
        return env_path.strip()
    
    # 默认从当前工作目录获取 mcp.json
    return str(Path.cwd() / "mcp.json")


async def get_mcp_tools_async(config_path: str | None = None) -> list[BaseTool]:
    """
    异步获取 MCP 工具列表
    
    这个函数会：
    1. 连接到 MCP 服务器
    2. 获取工具定义
    3. 创建带有重新连接能力的工具包装器
    4. 断开连接
    
    工具调用时会自动重新连接到 MCP 服务器
    
    Args:
        config_path: MCP 配置文件路径，默认从环境变量 MCP_CONFIG_FILEPATH 或当前工作目录下的 mcp.json
        
    Returns:
        LangChain 工具列表
    """
    # 获取配置文件路径
    if config_path is None:
        config_path = _get_mcp_config_path()
    
    # 加载配置
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        log_warning(f"MCP 配置文件未找到: {config_path}")
        return []
    except json.JSONDecodeError as e:
        log_warning(f"MCP 配置文件格式错误: {e}")
        return []
    
    servers = config.get("mcpServers", {})
    
    if not servers:
        log_info("ℹ️ 没有配置 MCP 服务器")
        return []
    
    log_debug("\n🔌 开始初始化 MCP 服务...")
    
    global _mcp_server_tool_counts_cache

    all_tools = []
    server_tool_counts: dict[str, int] = {}
    
    for server_name, server_config in servers.items():
        # 检查是否需要过滤 ida_pro_mcp
        if server_name == "ida_pro_mcp" and not _enable_ida_pro_mcp:
            server_tool_counts[server_name] = 0
            log_debug(f"⏭️ 跳过 MCP 服务器: {server_name} (需要 --ida 参数启用)")
            continue
        
        # 检查是否需要过滤 jadx_mcp
        if server_name == "jadx_mcp" and not _enable_jadx_mcp:
            server_tool_counts[server_name] = 0
            log_debug(f"⏭️ 跳过 MCP 服务器: {server_name} (需要 --jadx 参数启用)")
            continue
        
        # 检查是否需要过滤 idalib_mcp
        if server_name == "idalib_mcp" and not _enable_idalib_mcp:
            server_tool_counts[server_name] = 0
            log_debug(f"⏭️ 跳过 MCP 服务器: {server_name} (需要 --ida 带参数启用)")
            continue
        
        transport = server_config.get("transport", "stdio")
        
        # SSE 传输处理
        if transport == "sse":
            url = server_config.get("url")
            headers = server_config.get("headers", {})
            timeout = server_config.get("timeout", 60)  # 增加默认超时到60秒
            sse_read_timeout = server_config.get("sse_read_timeout", 300)
            max_retries = server_config.get("max_retries", 3)  # 最大重试次数
            retry_delay = server_config.get("retry_delay", 2)  # 重试延迟（秒）
            server_execution_lock = _get_server_execution_lock(server_name)
            
            if not url:
                server_tool_counts[server_name] = 0
                log_warning(f"MCP 服务器 {server_name} (SSE) 缺少 url 配置")
                continue
            
            log_info(f"🔌 正在连接 MCP 服务器: {server_name} (SSE: {url})")
            
            # 带重试机制的连接逻辑
            retry_count = 0
            last_error = None
            
            while retry_count < max_retries:
                try:
                    async with sse_client(
                        url=url,
                        headers=headers,
                        timeout=timeout,
                        sse_read_timeout=sse_read_timeout
                    ) as (read_stream, write_stream):
                        log_success(f"已连接到 MCP 服务器: {server_name} (SSE)")
                        
                        async with ClientSession(read_stream, write_stream) as session:
                            await session.initialize()
                            log_success(f"会话初始化成功: {server_name} (SSE)")
                            
                            # 获取工具列表
                            result = await session.list_tools()
                            server_tool_counts[server_name] = len(result.tools)
                            log_info(f"📦 从 {server_name} 获取到 {len(result.tools)} 个工具")
                            
                            for tool in result.tools:
                                input_schema = tool.inputSchema or {"type": "object", "properties": {}}
                                
                                # 创建工具调用函数（带重试机制）
                                async def call_tool(
                                    arguments: dict[str, Any],
                                    tool_name: str = tool.name,
                                    svr_url: str = url,
                                    svr_headers: dict[str, str] = headers,
                                    svr_timeout: int = timeout,
                                    svr_sse_timeout: int = sse_read_timeout,
                                    svr_max_retries: int = max_retries,
                                    svr_retry_delay: int = retry_delay
                                ) -> str:
                                    """重新连接并调用 MCP 工具 (SSE) - 带重试机制"""
                                    tool_retry_count = 0
                                    tool_last_error = None
                                    
                                    while tool_retry_count < svr_max_retries:
                                        try:
                                            async with sse_client(
                                                url=svr_url,
                                                headers=svr_headers,
                                                timeout=svr_timeout,
                                                sse_read_timeout=svr_sse_timeout
                                            ) as (rs, ws):
                                                async with ClientSession(rs, ws) as sess:
                                                    await sess.initialize()
                                                    result = await sess.call_tool(tool_name, arguments=arguments)
                                                    
                                                    if result.content:
                                                        output_parts = []
                                                        for content_block in result.content:
                                                            if hasattr(content_block, 'text'):
                                                                output_parts.append(content_block.text)
                                                            elif hasattr(content_block, 'data'):
                                                                output_parts.append(str(content_block.data))
                                                            else:
                                                                output_parts.append(str(content_block))
                                                        return "\n".join(output_parts)
                                                    return "Tool executed successfully with no output"
                                        except Exception as e:
                                            tool_last_error = e
                                            tool_retry_count += 1
                                            if tool_retry_count < svr_max_retries:
                                                log_warning(
                                                    f"工具 {tool_name} 调用失败，第 {tool_retry_count} 次重试... "
                                                    f"(错误: {_describe_exception(e)})"
                                                )
                                                await asyncio.sleep(svr_retry_delay)

                                    return (
                                        f"Error calling tool {tool_name} after {svr_max_retries} retries: "
                                        f"{_describe_exception(tool_last_error)}"
                                    )

                                wrapper = MCPToolWrapper(
                                    tool_name=tool.name,
                                    tool_description=tool.description,
                                    input_schema=input_schema,
                                    call_func=call_tool,
                                    execution_lock=server_execution_lock,
                                )
                                
                                all_tools.append(wrapper.langchain_tool)
                            
                            # 成功连接，退出重试循环
                            break
                    
                except Exception as e:
                    last_error = e
                    retry_count += 1
                    
                    if retry_count < max_retries:
                        log_warning(
                            f"连接 MCP 服务器 {server_name} (SSE) 失败，第 {retry_count} 次重试... "
                            f"(错误: {_describe_exception(e)})"
                        )
                        logger.warning(
                            f"连接 MCP 服务器 {server_name} (SSE) 失败 "
                            f"(尝试 {retry_count}/{max_retries}): {_describe_exception(e)}"
                        )
                        await asyncio.sleep(retry_delay)
                    else:
                        server_tool_counts.setdefault(server_name, 0)
                        log_error(
                            f"连接 MCP 服务器 {server_name} (SSE) 失败 "
                            f"(已重试 {max_retries} 次): {_describe_exception(e)}"
                        )
                        logger.exception(f"连接 MCP 服务器 {server_name} (SSE) 失败")
                        break
        
        # stdio 传输处理
        elif transport == "stdio":
            command = server_config.get("command")
            args = server_config.get("args", [])
            env = server_config.get("env", {})
            server_execution_lock = _get_server_execution_lock(server_name)
            
            if not command:
                server_tool_counts[server_name] = 0
                log_warning(f"MCP 服务器 {server_name} 缺少 command 配置")
                continue
            
            log_info(f"🔌 正在连接 MCP 服务器: {server_name}")
            
            merged_env = os.environ.copy()
            merged_env.update(env)
            
            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=merged_env,
            )
            
            try:
                async with stdio_client(server_params) as (read_stream, write_stream):
                    log_success(f"已连接到 MCP 服务器: {server_name}")
                    
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        log_success(f"会话初始化成功: {server_name}")
                        
                        result = await session.list_tools()
                        server_tool_counts[server_name] = len(result.tools)
                        log_info(f"📦 从 {server_name} 获取到 {len(result.tools)} 个工具")
                        
                        for tool in result.tools:
                            input_schema = tool.inputSchema or {"type": "object", "properties": {}}
                            
                            async def call_tool(
                                arguments: dict[str, Any],
                                tool_name: str = tool.name,
                                svr_params: StdioServerParameters = server_params
                            ) -> str:
                                """重新连接并调用 MCP 工具"""
                                try:
                                    async with stdio_client(svr_params) as (rs, ws):
                                        async with ClientSession(rs, ws) as sess:
                                            await sess.initialize()
                                            result = await sess.call_tool(tool_name, arguments=arguments)
                                            
                                            if result.content:
                                                output_parts = []
                                                for content_block in result.content:
                                                    if hasattr(content_block, 'text'):
                                                        output_parts.append(content_block.text)
                                                    elif hasattr(content_block, 'data'):
                                                        output_parts.append(str(content_block.data))
                                                    else:
                                                        output_parts.append(str(content_block))
                                                return "\n".join(output_parts)
                                            return "Tool executed successfully with no output"
                                except Exception as e:
                                    return f"Error calling tool {tool_name}: {_describe_exception(e)}"
                            
                            wrapper = MCPToolWrapper(
                                tool_name=tool.name,
                                tool_description=tool.description,
                                input_schema=input_schema,
                                call_func=call_tool,
                                execution_lock=server_execution_lock,
                            )
                            
                            all_tools.append(wrapper.langchain_tool)
                        
            except Exception as e:
                server_tool_counts.setdefault(server_name, 0)
                log_error(f"连接 MCP 服务器 {server_name} 失败: {e}")
                logger.exception(f"连接 MCP 服务器 {server_name} 失败")
        
        # 不支持的传输类型
        else:
            server_tool_counts[server_name] = 0
            log_warning(f"不支持的传输类型: {transport} (服务器: {server_name})")
    
    _mcp_server_tool_counts_cache = server_tool_counts
    log_success(f"MCP 初始化完成，共获取 {len(all_tools)} 个工具\n")
    
    return all_tools


def get_mcp_tools(config_path: str | None = None) -> list[BaseTool]:
    """
    同步获取 MCP 工具列表
    
    Args:
        config_path: MCP 配置文件路径，默认从环境变量 MCP_CONFIG_FILEPATH 或当前工作目录下的 mcp.json
        
    Returns:
        LangChain 工具列表
    """
    # 获取配置文件路径
    if config_path is None:
        config_path = _get_mcp_config_path()
    
    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, get_mcp_tools_async(config_path))
            return future.result()
    except RuntimeError:
        return asyncio.run(get_mcp_tools_async(config_path))


# 保持向后兼容的类
class MCPClient:
    """
    MCP 客户端（向后兼容）
    
    注意：推荐直接使用 get_mcp_tools_async() 或 get_mcp_tools() 函数
    """
    
    def __init__(self, config_path: str | None = None):
        # 获取配置文件路径
        if config_path is None:
            config_path = _get_mcp_config_path()
        self.config_path = config_path
        self._tools: list[BaseTool] = []
        self._initialized = False
    
    async def initialize(self) -> list[BaseTool]:
        """初始化并获取工具"""
        if self._initialized:
            return self._tools
        
        self._tools = await get_mcp_tools_async(self.config_path)
        self._initialized = True
        return self._tools
    
    def get_tools(self) -> list[BaseTool]:
        """获取已初始化的工具列表"""
        return self._tools
    
    async def close(self):
        """关闭连接（不需要实现，每次调用独立连接）"""
        self._tools.clear()
        self._initialized = False


# 全局缓存
_mcp_tools_cache: Optional[list[BaseTool]] = None
_mcp_server_tool_counts_cache: dict[str, int] = {}
_mcp_tools_cache_lock = threading.Lock()


async def load_mcp_tools_async(config_path: str | None = None, force_reload: bool = False) -> list[BaseTool]:
    """
    异步加载 MCP 工具（带缓存）
    
    Args:
        config_path: MCP 配置文件路径，默认从环境变量 MCP_CONFIG_FILEPATH 或当前工作目录下的 mcp.json
        force_reload: 是否强制重新加载
        
    Returns:
        LangChain 工具列表
    """
    global _mcp_tools_cache
    
    if _mcp_tools_cache is not None and not force_reload:
        return _mcp_tools_cache
    
    # 获取配置文件路径
    if config_path is None:
        config_path = _get_mcp_config_path()
    
    _mcp_tools_cache = await get_mcp_tools_async(config_path)
    return _mcp_tools_cache


def load_mcp_tools(config_path: str | None = None, force_reload: bool = False) -> list[BaseTool]:
    """
    同步加载 MCP 工具（带缓存）
    
    Args:
        config_path: MCP 配置文件路径，默认从环境变量 MCP_CONFIG_FILEPATH 或当前工作目录下的 mcp.json
        force_reload: 是否强制重新加载
        
    Returns:
        LangChain 工具列表
    """
    global _mcp_tools_cache
    
    if _mcp_tools_cache is not None and not force_reload:
        return _mcp_tools_cache

    with _mcp_tools_cache_lock:
        if _mcp_tools_cache is not None and not force_reload:
            return _mcp_tools_cache

        # 获取配置文件路径
        if config_path is None:
            config_path = _get_mcp_config_path()

        _mcp_tools_cache = get_mcp_tools(config_path)
        return _mcp_tools_cache


def get_mcp_server_tool_counts() -> dict[str, int]:
    """获取每个 MCP 服务器已加载的工具数量。"""
    return dict(_mcp_server_tool_counts_cache)
