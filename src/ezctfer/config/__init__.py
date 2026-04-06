"""
配置模块
"""

from .config_loader import (
    ApiType,
    LLMConfig,
    AppConfig,
    ConfigLoader,
    config_loader,
    get_config,
    init_config,
)
from .prompts import (
    CTF_SYSTEM_PROMPT_HEADER,
    CTF_SYSTEM_PROMPT_RULES,
    CTF_SYSTEM_PROMPT_FINDINGS,
    CTF_SYSTEM_PROMPT_NEW_LLM,
    CTF_SUMMARY_PROMPT,
    TOOL_RECORD_FINDING_DESCRIPTION,
    TOOL_SUBMIT_FLAG_DESCRIPTION,
    TOOL_RETRIEVE_KNOWLEDGE_DESCRIPTION,
    build_ctf_system_prompt,
    get_summary_prompt,
)

__all__ = [
    # Config Loader
    "ApiType",
    "LLMConfig",
    "AppConfig",
    "ConfigLoader",
    "config_loader",
    "get_config",
    "init_config",
    # Prompts
    "CTF_SYSTEM_PROMPT_HEADER",
    "CTF_SYSTEM_PROMPT_RULES",
    "CTF_SYSTEM_PROMPT_FINDINGS",
    "CTF_SYSTEM_PROMPT_NEW_LLM",
    "CTF_SUMMARY_PROMPT",
    "TOOL_RECORD_FINDING_DESCRIPTION",
    "TOOL_SUBMIT_FLAG_DESCRIPTION",
    "TOOL_RETRIEVE_KNOWLEDGE_DESCRIPTION",
    "build_ctf_system_prompt",
    "get_summary_prompt",
]
