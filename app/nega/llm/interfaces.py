# nega/llm/interfaces.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

Message = Dict[str, str]

@dataclass
class LLMConfig:
    provider: str
    model: str
    base_url: str
    timeout_s: float = 30.0 # why need 
    max_retries: int = 2
    temperature: float = 0.2
    top_p: float = 0.9
    # 预留额外参数
    extra: Dict[str, Any] = field(default_factory=dict)

class LLMProvider(Protocol):
    """所有 LLM Provider 的统一接口（面向 orchestrator / 业务层）。"""

    def chat(self, messages: List[Message], **kwargs: Any) -> str:
        """同步返回最终文本（最小实现）。后续可扩展 streaming / tool calling 等。"""
        ...