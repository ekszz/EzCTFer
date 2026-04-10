"""
CTF Solver - 多LLM协作的CTF解题框架
使用 LangChain 和 LangGraph 实现多轮次Agent解题
"""

import random
import re
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from ..llm import get_llm_manager, LLMInstance
from ..config.config_loader import ApiType, get_config
from ..config.log import (
    log_info,
    log_debug,
    log_warning,
    log_error,
    log_success,
    log_color,
    log_separator,
    console_print,
    prompt_input,
    Colors,
)
from ..tools.tools import (
    get_tools,
    get_important_info,
    clear_state,
    is_flag_found,
    get_found_flag,
    FlagFoundException,
    is_dual_thread_mode,
    is_quiet_mode,
    is_stop_requested,
    get_unread_findings,
    register_thread,
    get_flag_finder_thread_id,
    is_no_writeup,
)
from ..config.prompts import build_ctf_system_prompt, get_summary_prompt, get_writeup_prompt
from ..web.monitor import get_monitor, ChatMessage, get_pending_human_message, has_pending_human_message
from ..mcp_integration import load_mcp_tools


def get_all_tools() -> list[BaseTool]:
    """
    获取所有可用工具（内置工具 + MCP 工具）
    
    Returns:
        合并后的工具列表
    """
    # 获取内置工具（根据 RAG 启用状态动态获取）
    all_tools = list(get_tools())
    
    # 获取 MCP 工具
    try:
        mcp_tools = load_mcp_tools()
        if mcp_tools:
            all_tools.extend(mcp_tools)
            log_debug(f"📦 已加载 {len(mcp_tools)} 个 MCP 工具")
    except Exception as e:
        log_warning(f"加载 MCP 工具失败: {e}")
    
    return all_tools


# 最大迭代轮数（每个LLM）
# 注意：LangGraph 的 recursion_limit 是图执行步数，每次工具调用约消耗2步
# 所以设置为 120 约等于 60 次工具调用
MAX_ITERATIONS = 120
_PROMPT_STYLE = f"{Colors.WHITE_BG}{Colors.BLACK}"


class CTFSolver:
    """
    多LLM协作的CTF解题器
    """
    
    def __init__(self, task_description: str, max_iterations: int = MAX_ITERATIONS, thread_id: int = 0, thread_name: str = ""):
        """
        初始化CTF解题器
        
        Args:
            task_description: CTF题目描述
            max_iterations: 每个LLM的最大迭代轮数
            thread_id: 线程ID（双线程模式下使用）
            thread_name: 线程名称（双线程模式下使用）
        """
        self.task_description = task_description
        self.max_iterations = max_iterations
        self.llm_manager = get_llm_manager()
        self.thread_id = thread_id
        self.thread_name = thread_name
        
        # 双线程模式下注册线程
        if is_dual_thread_mode():
            register_thread(thread_id, thread_name)
        # 单线程模式下清除全局状态（只在第一个solver初始化时清除）
        elif thread_id == 0:
            clear_state()
        
        # 记录每个LLM的总结
        self.llm_summaries: list[dict] = []
        self.all_tools = get_all_tools()
    
    def _build_system_prompt(self, llm_name: str, is_new_llm: bool = True) -> str:
        """
        构建系统提示词
        
        Args:
            llm_name: 当前LLM名称
            is_new_llm: 是否是新切换的LLM
            
        Returns:
            系统提示词
        """
        current_info = get_important_info()
        return build_ctf_system_prompt(
            task_description=self.task_description,
            findings=current_info,
            llm_name=llm_name,
            is_new_llm=is_new_llm
        )
    
    def _build_summary_prompt(self) -> str:
        """构建总结提示词"""
        return get_summary_prompt()
    
    def _get_llm_index(self) -> int:
        """
        获取 LLM 索引
        
        根据当前模式和配置参数选择 LLM 索引：
        - 双线程模式：使用 DUAL_THREAD_0_LLM（线程1）或 DUAL_THREAD_1_LLM（线程2）
        - 单线程模式：使用 SINGLE_THREAD_LLM
        - 如果未配置或配置无效，则随机选择
        
        Returns:
            LLM 索引（0-based，内部数组索引）
        """
        config = get_config()
        configured_index = None  # 原始序号（对应 .env 中的 LLM_X）
        
        if is_dual_thread_mode():
            # 双线程模式：根据线程 ID 选择配置
            if self.thread_id == 1:
                configured_index = config.dual_thread_0_llm
            elif self.thread_id == 2:
                configured_index = config.dual_thread_1_llm
        else:
            # 单线程模式
            configured_index = config.single_thread_llm
        
        # 如果配置了原始序号，查找对应的 LLM
        if configured_index is not None:
            llm_instance = self.llm_manager.get_llm_by_original_index(configured_index)
            if llm_instance:
                # 找到对应的 LLM，返回其在数组中的索引
                llm_index = self.llm_manager.llms.index(llm_instance)
                log_debug(f"📌 使用配置指定的 LLM: {llm_instance.name} (序号: {configured_index})", self.thread_id)
                return llm_index
            else:
                # 配置的序号无效，输出警告并回退到随机选择
                log_warning(f"⚠️ 配置的 LLM 序号 {configured_index} 无效，回退到随机选择", self.thread_id)
        
        # 未配置或配置无效，随机选择
        llm_index = random.randint(0, len(self.llm_manager) - 1)
        return llm_index

    def run_single_llm(self, llm_index: int, initial_message: str = None, round_num: int = 1) -> tuple[bool, str]:
        """
        运行单个LLM进行解题 - 重要
        
        Args:
            llm_index: LLM索引
            initial_message: 初始消息（可选）
            round_num: 当前轮次号
            
        Returns:
            (是否找到flag, 总结内容)
        """
        llm_instance = self.llm_manager.get_llm_by_index(llm_index)
        if not llm_instance:
            return False, f"LLM索引 {llm_index} 不存在"
        
        log_info(f"🤖 切换到 LLM: {llm_instance.name} (模型: {llm_instance.config.model})", self.thread_id)
        
        # 判断是否是新LLM（有之前的发现）
        is_new_llm = len(get_important_info()) > 0
        
        # 构建系统提示
        system_prompt = self._build_system_prompt(llm_instance.name, is_new_llm)
        
        # 构建输入消息
        if initial_message:
            input_message = initial_message
        else:
            input_message = "请开始分析这道CTF题目，逐步探索并尝试找到flag。"
        
        # 获取监控器并开始新轮次
        monitor = get_monitor()
        monitor.start_round(round_num, llm_instance.name, llm_instance.config.model, self.thread_id)
        
        # 记录用户输入消息
        monitor.add_message(ChatMessage(
            role="user",
            content=input_message,
            thread_id=self.thread_id
        ))
        
        # 使用 MemorySaver 作为 checkpointer，支持状态管理和中断恢复
        checkpointer = MemorySaver()
        
        # 收集所有消息用于总结（在 try 块外初始化，确保异常处理时也能访问）
        all_messages = []
        
        try:
            # 获取所有工具（内置工具 + MCP 工具）
            all_tools = self.all_tools
            
            # 判断是否是 Anthropic 类型的 API
            is_anthropic = llm_instance.config.api_type == ApiType.ANTHROPIC
            
            # 创建 React Agent
            # 对于 Anthropic 兼容 API，不使用 prompt 参数（某些 API 不支持系统消息格式）
            if is_anthropic:
                # 不使用 prompt 参数，将系统提示词放在第一条消息中
                agent = create_react_agent(
                    model=llm_instance.client,
                    tools=all_tools,
                    interrupt_after=["tools"],
                    checkpointer=checkpointer,
                )
            else:
                # OpenAI 兼容 API 使用 prompt 参数
                agent = create_react_agent(
                    model=llm_instance.client,
                    tools=all_tools,
                    interrupt_after=["tools"],
                    checkpointer=checkpointer,
                    prompt=system_prompt
                )
            
            # 配置：包含 recursion_limit 和 thread_id（用于状态管理）
            thread_id = f"ctf_round_{round_num}"
            config = {
                "recursion_limit": self.max_iterations,
                "configurable": {"thread_id": thread_id}
            }
            
            # 对于 Anthropic 兼容 API，将系统提示词直接放在第一条用户消息中
            # 某些兼容 API 不支持 SystemMessage 格式
            if is_anthropic:
                # 将系统提示词合并到第一条用户消息中
                messages = [
                    HumanMessage(content=f"[系统指令]\n{system_prompt}\n\n[用户请求]\n{input_message}")
                ]
            else:
                messages = [HumanMessage(content=input_message)]
            
            # 初始输入
            current_input = {"messages": messages}
            
            # 迭代计数器：追踪累积的迭代次数
            # 由于 interrupt_after 会导致每次中断后重新调用 stream，
            # recursion_limit 可能会在每次 stream 调用时重置，
            # 所以需要手动追踪总迭代次数以确保不超过限制
            total_iterations = 0
            
            # 循环执行，直到 Agent 完成
            while True:
                # 检查停止信号（包括 Ctrl+C 和双线程模式的停止信号）
                if is_stop_requested():
                    log_info(f"\n⏹️ 收到停止信号，结束当前线程", self.thread_id)
                    summary = self._request_summary(llm_instance, all_messages)
                    monitor.complete_current_round(f"被Ctrl+C中断了。\n\n总结: {summary}", self.thread_id)
                    return True, f"收到信息，当前线程停止"
                
                # 检查是否超过最大迭代次数
                if total_iterations >= self.max_iterations:
                    log_warning(f"达到最大迭代次数限制 ({self.max_iterations})，强制结束本轮", self.thread_id)
                    break
                # 执行 stream，使用 stream_mode="values" 获取事件
                # checkpointer 通过 config 传入，支持中断恢复
                events = agent.stream(
                    current_input,
                    config=config,
                )
                
                for event in events:
                    # 检查是否是中断事件
                    if "__interrupt__" in event:
                        # interrupt_data = event["__interrupt__"]
                        # print(f"\n⏸️ 捕获到中断: {interrupt_data}")
                        
                        # 收集所有待注入的消息（用户消息 + 另一线程的finding消息）
                        message_parts = []
                        
                        # 检查是否有待处理的用户消息
                        if has_pending_human_message(self.thread_id):
                            human_msg = get_pending_human_message(self.thread_id)
                            if human_msg:
                                log_info(f"\n📩 追加用户消息到对话: {human_msg[:50]}...", self.thread_id)
                                message_parts.append(f"[用户输入]\n{human_msg}")
                        
                        # 双线程模式下检查是否有来自另一线程的 finding 消息
                        if is_dual_thread_mode():
                            unread_findings = get_unread_findings(self.thread_id)
                            if unread_findings:
                                log_debug(f"\n🔗 收到另一线程的发现 ({len(unread_findings)} 条)", self.thread_id)
                                combined_findings = "\n".join([f"- {f}" for f in unread_findings])
                                message_parts.append(f"[来自另一队伍的重大发现，以下内容不需要重复记录]\n{combined_findings}")
                        
                        # 如果有待注入的消息，合并为一条 HumanMessage
                        if message_parts:
                            combined_message = "\n\n".join(message_parts)
                            log_success(f"注入消息到对话，恢复执行", self.thread_id)
                            
                            # 使用 Command(update=...) 恢复执行并注入消息
                            current_input = Command(
                                update={
                                    "messages": [
                                        HumanMessage(content=combined_message)
                                    ]
                                }
                            )
                            
                            # 记录到监控器
                            monitor.add_message(ChatMessage(
                                role="user",
                                content=combined_message,
                                thread_id=self.thread_id
                            ))
                        else:
                            # 没有待处理消息，使用中断值恢复
                            # resume_value = interrupt_data.get("value") if isinstance(interrupt_data, dict) else interrupt_data
                            current_input = Command(resume={})

                        break  # 跳出 for 循环，继续 while 循环
                    else:
                        # 处理正常事件
                        self._process_stream_event(event, all_messages)
                        
                        # 递增迭代计数
                        total_iterations += 1
                        
                        # 检查是否找到flag
                        if is_flag_found():
                            flag = get_found_flag()
                            return self._handle_flag_found(llm_instance, all_messages, flag)
                        
                        # 检查是否达到迭代限制
                        if total_iterations >= self.max_iterations:
                            log_warning(f"达到最大迭代次数限制 ({self.max_iterations})，准备结束本轮", self.thread_id)
                            break
                else:
                    # for 循环正常结束（没有 break），说明 stream 完成
                    # 收集所有待注入的消息（用户消息 + 另一线程的finding消息）
                    message_parts = []
                    
                    # 检查是否有待处理的用户消息
                    if has_pending_human_message(self.thread_id):
                        human_msg = get_pending_human_message(self.thread_id)
                        if human_msg:
                            log_info(f"\n📩 检测到用户消息: {human_msg[:50]}...", self.thread_id)
                            message_parts.append(f"[用户输入]\n{human_msg}")
                    
                    # 双线程模式下检查是否有来自另一线程的 finding 消息
                    if is_dual_thread_mode():
                        unread_findings = get_unread_findings(self.thread_id)
                        if unread_findings:
                            log_debug(f"\n🔗 收到另一线程的发现 ({len(unread_findings)} 条)", self.thread_id)
                            combined_findings = "\n".join([f"- {f}" for f in unread_findings])
                            message_parts.append(f"[来自另一队伍的重大发现，以下内容不需要重复记录]\n{combined_findings}")
                    
                    # 如果有待注入的消息，合并为一条 HumanMessage
                    if message_parts:
                        combined_message = "\n\n".join(message_parts)
                        log_success(f"注入消息到对话，继续执行", self.thread_id)
                        current_input = {
                            "messages": [
                                HumanMessage(content=combined_message)
                            ]
                        }
                        # 记录到监控器
                        monitor.add_message(ChatMessage(
                            role="user",
                            content=combined_message,
                            thread_id=self.thread_id
                        ))
                        # 继续下一轮 while 循环
                    elif total_iterations <= self.max_iterations // 2:
                        log_info(f"本轮迭代数量{total_iterations}少于最大限制数量的一半，继续探索...")
                        current_input = {
                            "messages": [
                                HumanMessage(content="请继续探索")
                            ]
                        }
                        # 记录到监控器
                        monitor.add_message(ChatMessage(
                            role="user",
                            content="请继续探索",
                            thread_id=self.thread_id
                        ))
                        # 继续下一轮 while 循环
                    else:
                        # 没有待处理消息，正常结束
                        break
            
            # 检查是否找到flag
            if is_flag_found():
                flag = get_found_flag()
                return self._handle_flag_found(llm_instance, all_messages, flag)
            
            # 请求总结
            summary = self._request_summary(llm_instance, all_messages)
            
            # 完成当前轮次
            monitor.complete_current_round(summary, self.thread_id)
            
            return False, summary
            
        except FlagFoundException as e:
            # 捕获FlagFoundException，表示找到了flag，正常返回
            flag = e.flag
            return self._handle_flag_found(llm_instance, all_messages, flag)
            
        except Exception as e:
            error_msg = f"Agent执行出错: {str(e)}"
            log_error(error_msg, self.thread_id)
            import traceback
            traceback.print_exc()
            monitor.error_current_round(error_msg, self.thread_id)
            return False, error_msg

    def _handle_flag_found(self, llm_instance: LLMInstance, all_messages: list, flag: str) -> tuple[bool, str]:
        summary = self._request_summary(llm_instance, all_messages)
        finder_thread_id = get_flag_finder_thread_id()
        should_generate_writeup = (not is_dual_thread_mode()) or (finder_thread_id == self.thread_id)
        monitor = get_monitor()

        if should_generate_writeup:
            if not self._confirm_generate_writeup():
                monitor.complete_current_round(f"找到flag: {flag}\n\n总结: {summary}\n", self.thread_id)
                return True, f"已找到flag: {flag}\n\n总结: {summary}\n"

            writeup = self._generate_writeup(llm_instance, all_messages, flag)
            monitor.complete_current_round(f"找到flag: {flag}\n\n总结: {summary}\n\n--- Writeup ---\n{writeup}", self.thread_id)
            return True, f"已找到flag: {flag}\n\n总结: {summary}\n\n--- Writeup ---\n{writeup}"

        monitor.complete_current_round(f"找到flag: {flag}\n\n总结: {summary}", self.thread_id)
        return True, f"已找到flag: {flag}\n\n总结: {summary}"

    def _confirm_generate_writeup(self) -> bool:
        if is_no_writeup():
            return False
        if is_quiet_mode():
            return True

        while True:
            try:
                thread_label = f"Thread {self.thread_id}" if self.thread_id else "Thread 0"
                user_input = prompt_input(
                    [""],
                    f"🔄 Generate {thread_label} Writeup? | Enter y (confirm) / n (reject): ",
                    style=_PROMPT_STYLE,
                ).strip().lower()
                if user_input == "y":
                    return True
                if user_input == "n":
                    console_print("⚠️ Writeup skipped.")
                    return False
                console_print("⚠️ Invalid input. Please enter y or n.")
            except (EOFError, KeyboardInterrupt):
                console_print("\n⚠️ Writeup skipped due to user interrupt.")
                return False

    def _process_stream_event(self, event, all_messages: list) -> None:
        """
        处理流式事件，实时输出thinking和response
        
        Args:
            event: 流式事件（默认 stream 模式格式）
            all_messages: 收集所有消息的列表
        """
        monitor = get_monitor()
        
        # 默认 stream 模式下，事件格式为 {"节点名称": {"messages": [...]}, ...}
        # 跳过中断事件（在主循环中单独处理）
        if not isinstance(event, dict) or "__interrupt__" in event:
            return
        
        # 遍历每个节点的数据
        for node_name, node_data in event.items():
            # 跳过非字典数据
            if not isinstance(node_data, dict):
                continue
            
            # 获取消息列表
            messages = node_data.get("messages", [])
            
            pending_tool_messages: list[ToolMessage] = []

            def flush_pending_tool_messages() -> None:
                if not pending_tool_messages:
                    return
                self._record_tool_messages_to_monitor(pending_tool_messages, monitor)
                pending_tool_messages.clear()

            for msg in messages:
                # 使用消息ID来判断是否已处理
                msg_id = getattr(msg, 'id', None) or id(msg)
                
                # 检查是否已处理过该消息
                if any((getattr(m, 'id', None) or id(m)) == msg_id for m in all_messages):
                    continue
                
                # 记录消息
                all_messages.append(msg)
                
                # 处理AIMessage
                if isinstance(msg, AIMessage):
                    flush_pending_tool_messages()
                    self._print_ai_message(msg)
                    # 记录AI消息到监控器
                    self._record_ai_message_to_monitor(msg, monitor)
                
                # 处理ToolMessage
                elif isinstance(msg, ToolMessage):
                    pending_tool_messages.append(msg)
                else:
                    flush_pending_tool_messages()

            flush_pending_tool_messages()
    
    def _extract_text_content(self, content) -> str:
        """
        从消息内容中提取文本
        处理 OpenAI 和 Anthropic 不同的内容格式
        
        Args:
            content: 消息内容，可能是字符串或列表
            
        Returns:
            提取的文本内容
        """
        if content is None:
            return ""
        
        # 如果是字符串，直接返回
        if isinstance(content, str):
            return content
        
        # 如果是列表（Anthropic 格式），提取文本块
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    # Anthropic 文本块格式: {"type": "text", "text": "..."}
                    if block.get('type') == 'text':
                        text_parts.append(block.get('text', ''))
                    # 也处理可能的字符串类型
                    elif 'text' in block:
                        text_parts.append(block['text'])
                elif isinstance(block, str):
                    text_parts.append(block)
            return '\n'.join(text_parts)
        
        # 其他情况，尝试转换为字符串
        return str(content)
    
    def _extract_thinking_content(self, msg: AIMessage) -> str | None:
        if hasattr(msg, 'additional_kwargs') and msg.additional_kwargs:
            thinking = (msg.additional_kwargs.get('reasoning_content')
                       or msg.additional_kwargs.get('reasoning')
                       or msg.additional_kwargs.get('thinking')
                       or msg.additional_kwargs.get('thought'))
            if thinking:
                return thinking if isinstance(thinking, str) else str(thinking)

        content = getattr(msg, 'content', None)
        if not isinstance(content, list):
            return None

        parts = []
        for block in content:
            if not isinstance(block, dict):
                continue

            if block.get('type') not in ('reasoning', 'thinking', 'reasoning_content'):
                continue

            text = block.get('text')
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())

            summary = block.get('summary')
            if isinstance(summary, list):
                for item in summary:
                    if not isinstance(item, dict):
                        continue
                    summary_text = item.get('text')
                    if isinstance(summary_text, str) and summary_text.strip():
                        parts.append(summary_text.strip())

            block_content = block.get('content')
            if isinstance(block_content, str) and block_content.strip():
                parts.append(block_content.strip())

        if parts:
            return '\n\n'.join(parts)
        return None

    def _split_leading_think_block(self, text: str) -> tuple[str | None, str]:
        """
        拆分响应开头的 <think></think> 块。

        Args:
            text: 原始响应文本

        Returns:
            (thinking内容, 去掉think块后的正文)
        """
        if not text:
            return None, text

        match = re.match(r'^\s*<think>(.*?)</think>\s*', text, re.DOTALL)
        if not match:
            return None, text

        thinking = match.group(1).strip()
        remaining_text = text[match.end():].lstrip()
        return thinking or None, remaining_text

    def _print_ai_message(self, msg: AIMessage) -> None:
        """
        打印AI消息的thinking和response
        
        Args:
            msg: AI消息
        """
        # Extract thinking content.
        reasoning = self._extract_thinking_content(msg)
        # if reasoning:
        #     log_info(f"Thinking:\n{reasoning}\n", self.thread_id)
        
        # 输出response text（处理不同的内容格式）
        text_content = self._extract_text_content(msg.content)
        if text_content:
            log_debug(f"🤖 Response:\n{text_content}\n", self.thread_id)
        
        # 检查是否有工具调用 - 只显示工具名称
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tool_call in msg.tool_calls:
                tool_name = tool_call.get('name', 'unknown')
                log_debug(f"🔧 调用工具: {tool_name}", self.thread_id)
    
    def _record_ai_message_to_monitor(self, msg: AIMessage, monitor) -> None:
        """
        将AI消息记录到监控器
        
        Args:
            msg: AI消息
            monitor: 监控器实例
        """
        # Extract thinking content.
        thinking = self._extract_thinking_content(msg)
        
        # 提取文本内容（处理不同的内容格式）
        text_content = self._extract_text_content(msg.content)
        think_block, text_content = self._split_leading_think_block(text_content)
        if not thinking and think_block:
            thinking = think_block
        
        # 处理工具调用
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            tool_calls = []
            for tool_call in msg.tool_calls:
                tool_calls.append({
                    "name": tool_call.get('name', 'unknown'),
                    "args": tool_call.get('args', {}),
                    "call_id": tool_call.get('id'),
                })

            monitor.add_message(ChatMessage(
                role="assistant",
                content=text_content,
                tool_calls=tool_calls,
                thinking=thinking,
                thread_id=self.thread_id
            ))
        elif text_content or thinking:
            # 普通AI消息
            monitor.add_message(ChatMessage(
                role="assistant",
                content=text_content,  # 使用提取的文本内容
                thinking=thinking,
                thread_id=self.thread_id
            ))

    def _record_tool_messages_to_monitor(self, tool_messages: list[ToolMessage], monitor) -> None:
        """
        将一组连续的工具结果记录到监控器，合并为单条 tool 消息。

        Args:
            tool_messages: 连续的 ToolMessage 列表
            monitor: 监控器实例
        """
        if not tool_messages:
            return

        tool_results = []
        for msg in tool_messages:
            tool_results.append({
                "name": getattr(msg, 'name', None),
                "content": str(msg.content)[:2000] if msg.content else "",
                "call_id": getattr(msg, 'tool_call_id', None),
            })

        monitor.add_message(ChatMessage(
            role="tool",
            content="",
            tool_results=tool_results,
            thread_id=self.thread_id
        ))
    
    def _request_summary(self, llm_instance: LLMInstance, messages: list) -> str:
        """
        请求LLM总结本次解题过程
        
        Args:
            llm_instance: LLM实例
            messages: 消息列表
            
        Returns:
            总结内容
        """
        log_info("\n📝 正在生成本轮总结...", self.thread_id)
        
        summary_prompt = self._build_summary_prompt()
        
        try:
            # 判断是否是 Anthropic 类型的 API
            is_anthropic = llm_instance.config.api_type == ApiType.ANTHROPIC
            
            # 构建包含对话历史的消息列表
            if is_anthropic:
                # 对于 Anthropic 兼容 API，将系统提示词放在第一条用户消息中
                summary_messages = [
                    HumanMessage(content=f"{summary_prompt}\n\n请根据以下对话过程，提供简洁的总结：")
                ]
            else:
                # OpenAI 兼容 API 使用 SystemMessage
                summary_messages = [SystemMessage(content=summary_prompt)]
            
            # 添加完整的对话历史（不简化，全量归纳总结）
            for msg in messages:
                if isinstance(msg, AIMessage):
                    # 添加完整的AI消息
                    text_content = self._extract_text_content(msg.content)
                    summary_messages.append(AIMessage(content=f"[AI]: {text_content}"))
                elif isinstance(msg, ToolMessage):
                    # 添加完整的工具调用结果
                    summary_messages.append(AIMessage(content=f"[工具结果]: {msg.content}"))
                elif isinstance(msg, HumanMessage):
                    summary_messages.append(HumanMessage(content=f"[用户]: {msg.content}"))
            
            # 添加总结请求（OpenAI 兼容 API 需要）
            if not is_anthropic:
                summary_messages.append(HumanMessage(content="请根据以上对话过程，提供简洁的总结："))
            
            response = llm_instance.client.invoke(summary_messages)
            summary = self._extract_text_content(response.content)
            
            log_info(f"📋 本轮总结:\n{summary}\n", self.thread_id)
            return summary
        except Exception as e:
            log_warning(f"总结生成失败: {e}", self.thread_id)
            import traceback
            traceback.print_exc()
            return "无法生成本轮总结"
    
    def _generate_writeup(self, llm_instance: LLMInstance, messages: list, flag: str) -> str:
        """
        生成CTF解题报告（Writeup）
        
        Args:
            llm_instance: LLM实例
            messages: 消息列表
            flag: 找到的flag
            
        Returns:
            Writeup内容
        """
        log_info("\n📝 正在生成解题报告（Writeup）...", self.thread_id)
        
        writeup_prompt = get_writeup_prompt()
        
        try:
            # 判断是否是 Anthropic 类型的 API
            is_anthropic = llm_instance.config.api_type == ApiType.ANTHROPIC
            
            # 构建包含对话历史的消息列表
            if is_anthropic:
                # 对于 Anthropic 兼容 API，将系统提示词放在第一条用户消息中
                writeup_messages = [
                    HumanMessage(content=f"{writeup_prompt}\n\n题目描述：{self.task_description}\n\n最终获得的Flag：{flag}\n\n请根据以下完整解题过程，撰写详细的Writeup：")
                ]
            else:
                # OpenAI 兼容 API 使用 SystemMessage
                writeup_messages = [SystemMessage(content=writeup_prompt)]
            
            # 添加完整的对话历史
            for msg in messages:
                if isinstance(msg, AIMessage):
                    # 添加完整的AI消息
                    text_content = self._extract_text_content(msg.content)
                    writeup_messages.append(AIMessage(content=f"[AI]: {text_content}"))
                    
                elif isinstance(msg, ToolMessage):
                    # 添加完整的工具调用结果
                    tool_name = getattr(msg, 'name', 'unknown')
                    writeup_messages.append(AIMessage(content=f"[工具调用结果-{tool_name}]: {msg.content}"))
                    
                elif isinstance(msg, HumanMessage):
                    writeup_messages.append(HumanMessage(content=f"[用户]: {msg.content}"))
            
            # 添加Writeup请求（OpenAI 兼容 API 需要）
            if not is_anthropic:
                writeup_messages.append(HumanMessage(
                    content=f"题目描述：{self.task_description}\n\n最终获得的Flag：{flag}\n\n请根据以上完整解题过程，撰写详细的Writeup："
                ))
            
            response = llm_instance.client.invoke(writeup_messages)
            writeup = self._extract_text_content(response.content)
            
            log_success(f"📄 Writeup 已生成:\n{writeup}\n", self.thread_id)
            return writeup
        except Exception as e:
            log_warning(f"Writeup生成失败: {e}", self.thread_id)
            import traceback
            traceback.print_exc()
            return f"无法生成Writeup: {str(e)}"
    
    def solve(self, max_rounds: int = 60) -> tuple[bool, str]:
        """
        执行多LLM协作解题
        
        Args:
            max_rounds: 最大轮换轮数（每个LLM执行一次为一轮）
            
        Returns:
            (是否成功找到flag, flag内容或总结)
        """
        # 双线程模式下，在工作线程内部重新注册（确保 threading.local 正确设置）
        if is_dual_thread_mode():
            register_thread(self.thread_id, self.thread_name)
        
        log_debug("🎯 开始多LLM协作解题", self.thread_id)
        log_info(f"📝 题目描述: {self.task_description}", self.thread_id)
        log_debug(f"🔄 每个LLM最大迭代次数: {self.max_iterations}", self.thread_id)
        log_debug(f"👥 可用LLM数量: {len(self.llm_manager)}", self.thread_id)
        
        if len(self.llm_manager) == 0:
            return False, "没有可用的LLM"
        
        round_num = 0
        current_message = None
        
        while round_num < max_rounds:
            round_num += 1
            
            # 选择LLM：优先使用配置参数，否则随机选择
            llm_index = self._get_llm_index()
            
            log_info(f"\n🔄 第 {round_num} 轮", self.thread_id)
            
            # 运行当前LLM
            found, summary = self.run_single_llm(llm_index, current_message, round_num)
            
            # 记录本轮结果
            llm_instance = self.llm_manager.get_llm_by_index(llm_index)
            self.llm_summaries.append({
                "round": round_num,
                "llm_name": llm_instance.name if llm_instance else "unknown",
                "summary": summary,
                "findings_count": len(get_important_info())
            })
            
            # 检查是否找到flag
            if found:
                flag = get_found_flag()
                return True, flag
            
            # 为下一个LLM准备消息，包含上一轮的总结
            current_info = get_important_info()
            
            # 构建消息，包含上一轮LLM的总结
            summary_section = f"\n\n📋 上一轮LLM({llm_instance.name if llm_instance else 'unknown'})的总结：\n{summary}\n" if summary else ""
            
            # 用户最初的提示词（始终包含在每轮消息中）
            original_prompt_section = f"\n\n📌 用户最初的提示词：\n{self.task_description}\n"
            
            if current_info:
                findings_text = "\n".join([f"  - {info}" for info in current_info])
                current_message = f"""{original_prompt_section}
之前的分析已经发现了以下重要信息：
{findings_text}
{summary_section}
请基于这些发现和上一轮的总结继续分析，尝试找到flag。"""
            else:
                current_message = f"""{original_prompt_section}
之前的分析没有重大发现。{summary_section}
请尝试继续分析这道题目。"""
        
        final_summary = self._generate_final_summary()
        return False, final_summary
    
    def _generate_final_summary(self) -> str:
        """生成最终总结"""
        current_info = get_important_info()
        
        summary_lines = ["=== 最终总结 ===", ""]
        summary_lines.append(f"总轮数: {len(self.llm_summaries)}")
        summary_lines.append(f"重大发现数量: {len(current_info)}")
        summary_lines.append("")
        
        if current_info:
            summary_lines.append("重大发现:")
            for i, info in enumerate(current_info, 1):
                summary_lines.append(f"  {i}. {info}")
            summary_lines.append("")
        
        summary_lines.append("各轮总结:")
        for s in self.llm_summaries:
            summary_lines.append(f"\n--- 第 {s['round']} 轮 ({s['llm_name']}) ---")
            summary_lines.append(s['summary'])
        
        return "\n".join(summary_lines)


def solve_ctf(task_description: str, max_iterations: int = MAX_ITERATIONS, max_rounds: int = 60) -> tuple[bool, str]:
    """
    便捷函数：解决CTF题目
    
    Args:
        task_description: 题目描述
        max_iterations: 每个LLM的最大迭代次数
        max_rounds: 最大轮换轮数
        
    Returns:
        (是否成功找到flag, flag内容或总结)
    """
    solver = CTFSolver(task_description, max_iterations)
    return solver.solve(max_rounds)
