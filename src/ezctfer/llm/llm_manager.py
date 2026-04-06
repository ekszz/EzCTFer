"""
LLM 管理模块
使用数组管理多个 LLM 实例
支持通过 httpx 配置自动重试机制
"""

from collections.abc import Mapping
from typing import Any
from dataclasses import dataclass
import hashlib

import anthropic
import httpx
from langchain_core.language_models import BaseChatModel
from langchain_anthropic import ChatAnthropic

from ..config import LLMConfig, get_config
from ..config.config_loader import ApiType
from .chat_deepseek_fixed import ChatDeepSeekFixed
from .chat_openai_compatible import ChatOpenAICompatible

# 自定义 User-Agent，模拟 Claude CLI
CLAUDE_USER_AGENT = "claude-cli/2.0.76 (external, sdk-ts)"

# 默认重试配置
DEFAULT_MAX_RETRIES = 3  # 最大重试次数
DEFAULT_RETRY_BACKOFF_FACTOR = 0.5  # 重试退避因子（秒）
DEFAULT_SSL_VERIFY = False  # 忽略 SSL 证书校验


def build_default_prompt_cache_key(config: LLMConfig) -> str:
    """Build a stable prompt cache key for the OpenAI Responses API."""
    cache_identity = f"{config.api_type.value}|{config.api_url}|{config.model}"
    cache_hash = hashlib.sha256(cache_identity.encode("utf-8")).hexdigest()[:16]
    return f"ezctfer:{config.api_type.value}:{cache_hash}"


def merge_llm_extra(
    base_kwargs: dict[str, Any],
    extra_kwargs: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """
    合并 LLM 额外配置。

    - 字典会递归合并
    - extra 中的 null 会删除默认值
    - 其他类型直接覆盖
    """
    merged = dict(base_kwargs)
    if not isinstance(extra_kwargs, Mapping):
        return merged

    for key, value in extra_kwargs.items():
        current = merged.get(key)
        if value is None:
            merged.pop(key, None)
        elif isinstance(value, Mapping) and isinstance(current, Mapping):
            merged[key] = merge_llm_extra(dict(current), value)
        else:
            merged[key] = value

    return merged


def build_openai_compatible_kwargs(
    config: LLMConfig,
    http_client: httpx.Client,
    async_http_client: httpx.AsyncClient,
) -> dict[str, Any]:
    """构建 OpenAI 兼容客户端的初始化参数。"""
    prompt_cache_key = build_default_prompt_cache_key(config)
    base_kwargs = {
        "model": config.model,
        "api_key": config.api_key,
        "base_url": config.api_url,
        "request_timeout": config.timeout,
        "default_headers": {"User-Agent": CLAUDE_USER_AGENT},
        "model_kwargs": {"prompt_cache_key": prompt_cache_key},
        "http_client": http_client,
        "http_async_client": async_http_client,
    }

    merged_kwargs = merge_llm_extra(base_kwargs, config.extra)
    uses_responses_api = bool(
        merged_kwargs.get("use_responses_api")
        or merged_kwargs.get("reasoning") is not None
        or merged_kwargs.get("use_previous_response_id")
    )

    return merged_kwargs


def create_http_client_with_retries(
    timeout: int = 120,
    max_retries: int = DEFAULT_MAX_RETRIES,
    verify: bool = DEFAULT_SSL_VERIFY,
    proxy: str | None = None,
) -> httpx.Client:
    """
    创建带重试机制的 httpx 客户端
    
    Args:
        timeout: 请求超时时间（秒）
        max_retries: 最大重试次数
        proxy: 当前 LLM 使用的代理地址；未配置时保持 httpx 默认代理行为
        
    Returns:
        配置了重试机制的 httpx.Client
    """
    transport_kwargs: dict[str, Any] = {
        "retries": max_retries,
        "verify": verify,
    }
    client_kwargs: dict[str, Any] = {
        "timeout": timeout,
        "verify": verify,
    }
    if proxy:
        transport_kwargs["proxy"] = proxy
        transport_kwargs["trust_env"] = False
        client_kwargs["trust_env"] = False

    # 创建带重试的 transport
    transport = httpx.HTTPTransport(**transport_kwargs)
    
    # 创建客户端
    client = httpx.Client(
        transport=transport,
        **client_kwargs,
    )
    
    return client


def create_async_http_client_with_retries(
    timeout: int = 120,
    max_retries: int = DEFAULT_MAX_RETRIES,
    verify: bool = DEFAULT_SSL_VERIFY,
    proxy: str | None = None,
) -> httpx.AsyncClient:
    """
    创建带重试机制的异步 httpx 客户端
    
    Args:
        timeout: 请求超时时间（秒）
        max_retries: 最大重试次数
        proxy: 当前 LLM 使用的代理地址；未配置时保持 httpx 默认代理行为
        
    Returns:
        配置了重试机制的 httpx.AsyncClient
    """
    transport_kwargs: dict[str, Any] = {
        "retries": max_retries,
        "verify": verify,
    }
    client_kwargs: dict[str, Any] = {
        "timeout": timeout,
        "verify": verify,
    }
    if proxy:
        transport_kwargs["proxy"] = proxy
        transport_kwargs["trust_env"] = False
        client_kwargs["trust_env"] = False

    # 创建带重试的 transport
    transport = httpx.AsyncHTTPTransport(**transport_kwargs)
    
    # 创建异步客户端
    client = httpx.AsyncClient(
        transport=transport,
        **client_kwargs,
    )
    
    return client


@dataclass
class LLMInstance:
    """LLM 实例封装"""
    name: str
    config: LLMConfig
    client: BaseChatModel
    
    @property
    def index(self) -> int:
        """获取原始序号（对应 .env 中的 LLM_X）"""
        return self.config.index
    
    def invoke(self, *args, **kwargs) -> Any:
        """调用 LLM"""
        return self.client.invoke(*args, **kwargs)
    
    def __repr__(self) -> str:
        return f"LLMInstance(name='{self.name}', model='{self.config.model}')"


class LLMManager:
    """
    LLM 管理器
    使用数组管理多个 LLM 实例
    """
    
    _instance = None
    _llms: list[LLMInstance] = []
    _name_to_llm: dict[str, LLMInstance] = {}
    _initialized: bool = False
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def initialize(self) -> None:
        """
        初始化所有 LLM 实例
        从配置中读取 LLM 配置并创建实例
        """
        if self._initialized:
            return
        
        config = get_config()
        
        for llm_config in config.llm_configs:
            try:
                llm_instance = self._create_llm_instance(llm_config)
                self._llms.append(llm_instance)
                self._name_to_llm[llm_config.name] = llm_instance
            except Exception as e:
                print(f"警告: 创建 LLM '{llm_config.name}' 失败: {e}")
                continue
        
        self._initialized = True
    
    def _create_llm_instance(self, config: LLMConfig) -> LLMInstance:
        """
        根据 LLM 配置创建 LLM 实例
        根据 api_type 选择不同的客户端实现
        
        Args:
            config: LLM 配置
            
        Returns:
            LLMInstance: LLM 实例
        """
        client: BaseChatModel
        
        # 创建带重试机制的 httpx 客户端
        http_client = create_http_client_with_retries(
            timeout=config.timeout,
            max_retries=DEFAULT_MAX_RETRIES,
            proxy=config.proxy,
        )
        async_http_client = create_async_http_client_with_retries(
            timeout=config.timeout,
            max_retries=DEFAULT_MAX_RETRIES,
            proxy=config.proxy,
        )
        
        if config.api_type == ApiType.ANTHROPIC:
            # 使用 ChatAnthropic 客户端
            # Anthropic API 使用 anthropic_api_url 参数设置自定义 URL
            # 注意：langchain-anthropic 不支持直接传递 http_client 参数
            # 它使用内部的 _get_default_httpx_client 函数创建客户端
            # 使用 max_retries 参数配置重试机制
            client_kwargs = merge_llm_extra(
                {
                    "model": config.model,
                    "api_key": config.api_key,
                    "anthropic_api_url": config.api_url,
                    "timeout": config.timeout,
                    "max_retries": DEFAULT_MAX_RETRIES,
                    "anthropic_proxy": config.proxy,
                    # 设置自定义 User-Agent
                    "default_headers": {"User-Agent": CLAUDE_USER_AGENT},
                    # Anthropic 特殊配置：支持系统消息
                    # 某些兼容 API 需要这个参数
                },
                config.extra,
            )
            client = ChatAnthropic(**client_kwargs)
            client._client = anthropic.Client(
                api_key=client_kwargs.get("api_key", config.api_key),
                base_url=client_kwargs.get("anthropic_api_url", config.api_url),
                timeout=client_kwargs.get("timeout", config.timeout),
                max_retries=client_kwargs.get("max_retries", DEFAULT_MAX_RETRIES),
                default_headers=client_kwargs.get(
                    "default_headers",
                    {"User-Agent": CLAUDE_USER_AGENT},
                ),
                http_client=http_client,
            )
            client._async_client = anthropic.AsyncClient(
                api_key=client_kwargs.get("api_key", config.api_key),
                base_url=client_kwargs.get("anthropic_api_url", config.api_url),
                timeout=client_kwargs.get("timeout", config.timeout),
                max_retries=client_kwargs.get("max_retries", DEFAULT_MAX_RETRIES),
                default_headers=client_kwargs.get(
                    "default_headers",
                    {"User-Agent": CLAUDE_USER_AGENT},
                ),
                http_client=async_http_client,
            )
        elif config.api_type == ApiType.DEEPSEEK:
            # 使用 ChatDeepSeekFixed 客户端（修复版）
            # DeepSeek 推理模型（如 deepseek-reasoner）需要在消息中保留 reasoning_content 字段
            # 官方的 langchain_deepseek 在序列化消息时会丢失 reasoning_content，导致 API 报错
            # ChatDeepSeekFixed 修复了这个问题
            client_kwargs = merge_llm_extra(
                {
                    "model": config.model,
                    "api_key": config.api_key,
                    "base_url": config.api_url,
                    "timeout": config.timeout,
                    # 设置自定义 User-Agent
                    "default_headers": {"User-Agent": CLAUDE_USER_AGENT},
                    # 配置带重试的 httpx 客户端
                    "httpx_client": http_client,
                    "httpx_async_client": async_http_client,
                },
                config.extra,
            )
            client = ChatDeepSeekFixed(**client_kwargs)
        elif config.api_type == ApiType.OPENAI:
            # 默认使用 ChatOpenAI 作为通用的 LLM 客户端
            # 它支持自定义 base_url，可以连接到各种兼容 OpenAI API 的服务
            # Responses 相关能力通过 LLM_X_EXTRA 控制。
            client_kwargs = build_openai_compatible_kwargs(
                config,
                http_client,
                async_http_client,
            )
            client = ChatOpenAICompatible(**client_kwargs)
        else:
            raise ValueError(f"不支持的 API 类型: {config.api_type}")
        
        return LLMInstance(
            name=config.name,
            config=config,
            client=client
        )
    
    @property
    def llms(self) -> list[LLMInstance]:
        """获取所有 LLM 实例"""
        return self._llms
    
    def get_llm(self, name: str) -> LLMInstance | None:
        """
        根据名称获取 LLM 实例
        
        Args:
            name: LLM 名称
            
        Returns:
            LLMInstance | None: LLM 实例，不存在则返回 None
        """
        return self._name_to_llm.get(name)
    
    def get_llm_names(self) -> list[str]:
        """获取所有 LLM 名称"""
        return list(self._name_to_llm.keys())
    
    def get_llm_by_index(self, index: int) -> LLMInstance | None:
        """
        根据索引获取 LLM 实例
        
        Args:
            index: LLM 索引（从 0 开始）
            
        Returns:
            LLMInstance | None: LLM 实例，索引越界则返回 None
        """
        if 0 <= index < len(self._llms):
            return self._llms[index]
        return None
    
    def get_llm_by_original_index(self, original_index: int) -> LLMInstance | None:
        """
        根据原始序号获取 LLM 实例
        
        Args:
            original_index: 原始序号（对应 .env 中的 LLM_X，从 1 开始）
            
        Returns:
            LLMInstance | None: LLM 实例，不存在则返回 None
        """
        for llm in self._llms:
            if llm.index == original_index:
                return llm
        return None
    
    def add_llm(self, config: LLMConfig) -> LLMInstance:
        """
        动态添加一个 LLM
        
        Args:
            config: LLM 配置
            
        Returns:
            LLMInstance: 新创建的 LLM 实例
        """
        llm_instance = self._create_llm_instance(config)
        self._llms.append(llm_instance)
        self._name_to_llm[config.name] = llm_instance
        return llm_instance
    
    def remove_llm(self, name: str) -> bool:
        """
        移除指定名称的 LLM
        
        Args:
            name: LLM 名称
            
        Returns:
            bool: 是否成功移除
        """
        llm = self._name_to_llm.pop(name, None)
        if llm is not None:
            self._llms.remove(llm)
            return True
        return False
    
    def clear(self) -> None:
        """清除所有 LLM 实例"""
        self._llms.clear()
        self._name_to_llm.clear()
        self._initialized = False
    
    def __len__(self) -> int:
        """返回 LLM 数量"""
        return len(self._llms)
    
    def __iter__(self):
        """迭代所有 LLM"""
        return iter(self._llms)
    
    def __repr__(self) -> str:
        names = self.get_llm_names()
        return f"LLMManager(llms={names})"


# 全局 LLM 管理器实例
llm_manager = LLMManager()


def get_llm_manager() -> LLMManager:
    """获取 LLM 管理器实例"""
    return llm_manager


def init_llms() -> None:
    """初始化所有 LLM"""
    llm_manager.initialize()
