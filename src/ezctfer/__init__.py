"""
LTeam - 基于 LangChain 的多 LLM CTF 解题框架
"""

__version__ = "1.1.1"


# tiktoken 编码缓存（模块级，供 _tiktoken_get_token_ids 使用）
_tiktoken_enc = None


def _tiktoken_get_token_ids(text: str):
    """用 tiktoken 的 GPT-2 编码替换 langchain_core 对 GPT2TokenizerFast 的依赖。

    选用 ``gpt2`` 编码以与 langchain 原回退（``GPT2TokenizerFast("gpt2")``）保持
    同一套 BPE、计数一致；若更看重现代 OpenAI 模型的近似精度，可改为 ``cl100k_base``。
    """
    global _tiktoken_enc
    if _tiktoken_enc is None:
        import tiktoken

        _tiktoken_enc = tiktoken.get_encoding("gpt2")
    return _tiktoken_enc.encode(text)


def _setup_fast_start() -> None:
    """启动加速：未启用 RAG 时，阻止 torch/transformers 等重型库在导入期被加载。

    问题：``langchain_core.language_models.base`` 在模块顶层执行
    ``from transformers import GPT2TokenizerFast``（仅 try/except 守卫），
    会连带加载 transformers + torch，启动耗时数秒，且与是否使用 RAG 无关——
    只要导入 ``langchain_openai`` / ``langchain_anthropic`` / ``langchain_deepseek``
    就会触发（实测 transformers≈5.7s、torch≈2.5s）。

    方案：在 ``ezctfer`` 包被导入的最早期（早于 ``__main__`` 中的任何顶层 import）
    注册一个 ``sys.meta_path`` finder，对 torch / transformers / sentence_transformers /
    lightrag 主动抛 ``ImportError``，使 langchain 与 ``rag_service`` 里的 try/except
    统一走到 None 分支，从而彻底跳过这些重型库的加载；同时用 tiktoken 替换
    langchain_core 的 GPT-2 token 计数回退实现，保证 ``get_num_tokens`` 等仍可用。

    仅在未传 ``--rag`` / ``--init-rag`` 时生效；这两个场景需要重型库，会放行原行为。
    如需强制关闭本优化（用于排查），设置环境变量 ``EZCTFER_NO_FAST_START=1``。
    """
    import os
    import sys

    if os.environ.get("EZCTFER_NO_FAST_START"):
        return
    # --rag / --init-rag 需要 torch/transformers/sentence_transformers/lightrag，放行。
    if any(token in ("--rag", "--init-rag") for token in sys.argv):
        return

    _blocked = {"torch", "transformers", "sentence_transformers", "lightrag"}

    class _HeavyLibBlocker:
        """对屏蔽列表中的顶层包及其子模块抛 ImportError。"""

        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".", 1)[0] in _blocked:
                raise ImportError(
                    f"ezctfer fast-start: 未启用 --rag，已跳过重型依赖 '{fullname}'"
                )
            return None

    if not any(isinstance(finder, _HeavyLibBlocker) for finder in sys.meta_path):
        sys.meta_path.insert(0, _HeavyLibBlocker())

    # 用 tiktoken 替换 langchain_core 的 GPT-2 token 计数回退，避免依赖 transformers。
    try:
        import langchain_core.language_models.base as _lc_base

        if not getattr(_lc_base, "_ezctfer_tiktoken_patched", False):
            _lc_base._get_token_ids_default_method = _tiktoken_get_token_ids
            _lc_base._ezctfer_tiktoken_patched = True
    except Exception:
        # langchain_core 未安装或结构不符时，静默跳过，不影响启动。
        pass


_setup_fast_start()
