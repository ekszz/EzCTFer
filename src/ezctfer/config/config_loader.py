"""
配置加载工具类
从项目根目录的 .env 文件读取 LLM 配置
"""

import json
import os
from typing import Any
from pathlib import Path
from dataclasses import dataclass, field

from dotenv import load_dotenv


from enum import Enum


class ApiType(Enum):
    """API 类型枚举"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    
    @classmethod
    def from_string(cls, value: str) -> "ApiType":
        """从字符串解析 API 类型，不区分大小写"""
        value = value.lower().strip()
        for api_type in cls:
            if api_type.value == value:
                return api_type
        raise ValueError(f"不支持的 API 类型: {value}")


@dataclass
class LLMConfig:
    """LLM 配置数据类"""
    name: str
    api_key: str
    api_url: str
    model: str
    timeout: int = 120  # 默认超时时间（秒）
    api_type: ApiType = ApiType.OPENAI  # API 类型，默认为 OpenAI 兼容
    index: int = 0  # 原始序号（对应 .env 中的 LLM_X）
    proxy: str | None = None  # 可选代理地址，仅作用于当前 LLM
    extra: dict[str, Any] = field(default_factory=dict)  # 额外的模型初始化参数
    
    def __post_init__(self):
        """验证配置有效性"""
        if self.proxy is not None:
            self.proxy = self.proxy.strip() or None
        if not self.api_key:
            raise ValueError(f"LLM '{self.name}' 缺少 API Key")
        if not self.api_url:
            raise ValueError(f"LLM '{self.name}' 缺少 API URL")
        if not self.model:
            raise ValueError(f"LLM '{self.name}' 缺少 Model 名称")
        if not isinstance(self.extra, dict):
            raise ValueError(f"LLM '{self.name}' 的 extra 配置必须是 JSON 对象")


@dataclass
class AppConfig:
    """应用配置数据类"""
    llm_configs: list[LLMConfig] = field(default_factory=list)
    max_iterations: int = 120  # LangGraph图执行步数，每次工具调用约消耗2步，所以120约等于60次工具调用
    max_rounds: int = 10  # 最多LLM切换轮数
    # LLM 选择配置（可选）
    single_thread_llm: int | None = None  # 单线程模式下使用的 LLM 索引
    dual_thread_0_llm: int | None = None  # 双线程模式下线程1使用的 LLM 索引
    dual_thread_1_llm: int | None = None  # 双线程模式下线程2使用的 LLM 索引
    
    def get_llm_config(self, name: str) -> LLMConfig | None:
        """根据名称获取 LLM 配置"""
        for config in self.llm_configs:
            if config.name == name:
                return config
        return None
    
    def get_all_llm_names(self) -> list[str]:
        """获取所有 LLM 名称"""
        return [config.name for config in self.llm_configs]


class ConfigLoader:
    """配置加载器"""
    
    _instance = None
    _config: AppConfig | None = None
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_config(self, env_path: str | Path | None = None) -> AppConfig:
        """
        加载配置
        
        Args:
            env_path: .env 文件路径，默认为项目根目录
            
        Returns:
            AppConfig: 应用配置对象
        """
        if self._config is not None:
            return self._config
        
        # 确定 .env 文件路径
        if env_path is None:
            # 默认从当前工作目录查找 .env 文件
            env_path = Path.cwd() / ".env"
        else:
            env_path = Path(env_path)
        
        # 加载 .env 文件
        if env_path.exists():
            load_dotenv(env_path)
        else:
            raise FileNotFoundError(f"配置文件不存在: {env_path}")
        
        # 解析 LLM 配置
        llm_configs = self._parse_llm_configs()
        
        # 解析应用配置参数
        max_iterations = int(os.getenv("MAX_ITERATIONS", "120"))
        max_rounds = int(os.getenv("MAX_ROUNDS", "10"))
        
        # 解析 LLM 选择配置（可选）
        single_thread_llm = self._parse_optional_int("SINGLE_THREAD_LLM")
        dual_thread_0_llm = self._parse_optional_int("DUAL_THREAD_0_LLM")
        dual_thread_1_llm = self._parse_optional_int("DUAL_THREAD_1_LLM")
        
        self._config = AppConfig(
            llm_configs=llm_configs,
            max_iterations=max_iterations,
            max_rounds=max_rounds,
            single_thread_llm=single_thread_llm,
            dual_thread_0_llm=dual_thread_0_llm,
            dual_thread_1_llm=dual_thread_1_llm
        )
        return self._config
    
    def _parse_optional_int(self, env_key: str) -> int | None:
        """
        解析可选的整数环境变量
        
        Args:
            env_key: 环境变量名
            
        Returns:
            整数值或 None（如果未配置或无效）
        """
        value = os.getenv(env_key)
        if value is None or value.strip() == "":
            return None
        try:
            return int(value)
        except ValueError:
            return None

    def _parse_optional_json_dict(self, env_key: str) -> dict[str, Any]:
        """
        解析可选的 JSON 对象环境变量

        Args:
            env_key: 环境变量名

        Returns:
            dict[str, Any]: 解析后的 JSON 对象；未配置时返回空字典
        """
        value = os.getenv(env_key)
        if value is None or value.strip() == "":
            return {}

        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{env_key} 不是合法的 JSON: {exc.msg}") from exc

        if not isinstance(parsed, dict):
            raise ValueError(f"{env_key} 必须是 JSON 对象")

        return parsed
    
    def _parse_llm_configs(self) -> list[LLMConfig]:
        """
        从环境变量解析 LLM 配置
        支持动态数量的 LLM，通过序号区分 (LLM_1_, LLM_2_, ...)
        自动发现所有配置的序号并按从小到大排序
        """
        import re
        
        # 收集所有 LLM 配置的序号
        indices = set()
        pattern = re.compile(r"^LLM_(\d+)_NAME$")
        
        for key in os.environ:
            match = pattern.match(key)
            if match:
                indices.add(int(match.group(1)))
        
        if not indices:
            return []
        
        # 按序号从小到大排序
        sorted_indices = sorted(indices)
        
        configs = []
        for index in sorted_indices:
            name = os.getenv(f"LLM_{index}_NAME")
            if name is None:
                continue
            
            api_key = os.getenv(f"LLM_{index}_API_KEY", "")
            api_url = os.getenv(f"LLM_{index}_API_URL", "")
            model = os.getenv(f"LLM_{index}_MODEL", "")
            timeout = int(os.getenv(f"LLM_{index}_TIMEOUT", "120"))  # 默认120秒
            api_type_str = os.getenv(f"LLM_{index}_API_TYPE", "openai")  # 默认为 openai
            api_type = ApiType.from_string(api_type_str)
            proxy = os.getenv(f"LLM_{index}_PROXY")
            extra = self._parse_optional_json_dict(f"LLM_{index}_EXTRA")
            
            config = LLMConfig(
                name=name,
                api_key=api_key,
                api_url=api_url,
                model=model,
                timeout=timeout,
                api_type=api_type,
                index=index,  # 保存原始序号
                proxy=proxy,
                extra=extra,
            )
            configs.append(config)
        
        return configs
    
    @property
    def config(self) -> AppConfig:
        """获取已加载的配置"""
        if self._config is None:
            raise RuntimeError("配置尚未加载，请先调用 load_config()")
        return self._config
    
    def reload_config(self, env_path: str | Path | None = None) -> AppConfig:
        """重新加载配置"""
        self._config = None
        return self.load_config(env_path)


# 全局配置加载器实例
config_loader = ConfigLoader()


def get_config() -> AppConfig:
    """获取应用配置的快捷方法"""
    return config_loader.config


def init_config(env_path: str | Path | None = None) -> AppConfig:
    """初始化配置的快捷方法"""
    return config_loader.load_config(env_path)
