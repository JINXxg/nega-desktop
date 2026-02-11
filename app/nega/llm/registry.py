# nega/llm/registry.py
from __future__ import annotations

from typing import Callable, Dict, Type # what is the function of Callable and Type

from .interfaces import LLMProvider, LLMConfig

_PROVIDER_REGISTRY: Dict[str, Type[LLMProvider]] = {}

# Type描述类对象，Callable描述可调用对象（函数/方法/类的构造器等）
def register_provider(name: str) -> Callable[[Type[LLMProvider]], Type[LLMProvider]]:
    """装饰器：把 Provider 类注册到全局表，后续通过 provider name 实例化。"""
    def _decorator(cls: Type[LLMProvider]) -> Type[LLMProvider]: # here need to repeat the decorator design pattern
        if name in _PROVIDER_REGISTRY:
            raise KeyError(f"LLMProvider '{name}' already registered: {_PROVIDER_REGISTRY[name]}" )
        _PROVIDER_REGISTRY[name] = cls
        return cls
    return _decorator

def create_provider(config: LLMConfig) -> LLMProvider:
        """根据 config.yaml 的 provider 字符串创建实例。"""
        if config.provider not in _PROVIDER_REGISTRY:
            known = ",".join(_PROVIDER_REGISTRY.keys()) # what is the function of this part
            raise KeyError(f"Unknown LLMProvider '{config.provider}'. known: {known}")
        cls = _PROVIDER_REGISTRY[config.provider]
        return cls(config) # cls is what
  