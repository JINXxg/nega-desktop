# nega/llm/providers/dashscope_openai.py 
from __future__ import annotations

import os
from typing import Any, List

from openai import OpenAI

from ..interfaces import LLMConfig, Message
from ..registry import register_provider
#这里的目的就是把类对象注册进全局的_PROVIDER_REGISTRY字典中，以便后续通过名称创建实例
@register_provider("dashscope_openai") # this is a class decorator
class DashscopeOpenAIProvider:
    """
    阿里 DashScope 的 OpenAI 兼容模式 Provider。

    - API Key: 环境变量 DASHSCOPE_API_KEY  (官方推荐)  :contentReference[oaicite:3]{index=3}
    - base_url 示例（北京）：https://dashscope.aliyuncs.com/compatible-mode/v1 :contentReference[oaicite:4]{index=4}
    """
     
    def __init__(self, config: LLMConfig):
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing DASHSCOPE_API_KEY set it in .env ")
        
        self.config = config
        self.client = OpenAI(
              api_key = api_key,
              base_url = config.base_url,
              timeout = config.timeout_s,
              max_retries = config.max_retries
        )
    
    def chat(self, messages: List[Message], **kwargs: Any) -> str:
        temperature = float(kwargs.get("temperature", self.config.temperature))
        top_p = float(kwargs.get("top_p", self.config.top_p))

        response = self.client.chat.completions.create(
            model = self.config.model,
            messages = messages,
            temperature = temperature,
            top_p = top_p
        )

        return response.choices[0].message.content or "" 
