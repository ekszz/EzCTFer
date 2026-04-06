from collections.abc import Mapping
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI


THINKING_FIELDS = ("reasoning_content", "reasoning", "thinking", "thought")


class ChatOpenAICompatible(ChatOpenAI):
    @staticmethod
    def _extract_thinking_fields(data: Mapping[str, Any] | None) -> dict[str, Any]:
        if not isinstance(data, Mapping):
            return {}

        extracted: dict[str, Any] = {}
        for field in THINKING_FIELDS:
            value = data.get(field)
            if value:
                extracted[field] = value
        return extracted

    @classmethod
    def _merge_thinking_fields(
        cls,
        message: AIMessage | AIMessageChunk,
        data: Mapping[str, Any] | None,
    ) -> None:
        for field, value in cls._extract_thinking_fields(data).items():
            if not message.additional_kwargs.get(field):
                message.additional_kwargs[field] = value

    def _create_chat_result(
        self,
        response: dict | Any,
        generation_info: dict | None = None,
    ) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)

        response_dict = response if isinstance(response, dict) else response.model_dump()
        choices = response_dict.get("choices") or []
        for generation, choice in zip(result.generations, choices):
            message = getattr(generation, "message", None)
            if isinstance(message, AIMessage):
                self._merge_thinking_fields(message, choice.get("message"))

        return result

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk,
            default_chunk_class,
            base_generation_info,
        )
        if generation_chunk is None:
            return None

        message = generation_chunk.message
        if not isinstance(message, AIMessageChunk):
            return generation_chunk

        choices = chunk.get("choices", []) or chunk.get("chunk", {}).get("choices", [])
        if choices:
            self._merge_thinking_fields(message, choices[0].get("delta"))

        return generation_chunk
