"""
DeepSeek 聊天模型修复版
解决 langchain_deepseek 在消息序列化时丢失 reasoning_content 字段的问题
"""

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_deepseek import ChatDeepSeek as _ChatDeepSeek


class ChatDeepSeekFixed(_ChatDeepSeek):
    """
    修复版 DeepSeek 聊天模型
    
    解决问题：
    - langchain_deepseek 在序列化 AIMessage 时，不会将 additional_kwargs 中的
      reasoning_content 字段包含在请求中，导致 DeepSeek API 报错：
      "Missing reasoning_content field in the assistant message"
    
    修复方法：
    - 重写 _get_request_payload 方法，在序列化 assistant 消息时，
      将 additional_kwargs 中的 reasoning_content 添加到请求体中
    """
    
    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        """
        获取 API 请求的 payload
        
        重写此方法以正确处理 reasoning_content 字段
        """
        # 调用父类方法获取基本 payload
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        
        # 处理 assistant 消息中的 reasoning_content
        # 需要将 AIMessage.additional_kwargs 中的 reasoning_content 添加到请求中
        if hasattr(input_, '__iter__') and not isinstance(input_, str):
            # input_ 可能是消息列表
            messages = list(input_)
            for i, msg in enumerate(messages):
                if isinstance(msg, AIMessage):
                    # 检查是否有 reasoning_content
                    reasoning_content = msg.additional_kwargs.get('reasoning_content')
                    if reasoning_content and i < len(payload.get('messages', [])):
                        # 将 reasoning_content 添加到请求体中
                        payload['messages'][i]['reasoning_content'] = reasoning_content
        
        return payload