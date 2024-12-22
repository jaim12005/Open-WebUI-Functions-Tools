"""
title: OpenAI o1 Integration for OpenWebUI
author: Balaxxe
version: 1.4
license: MIT
requirements: pydantic>=2.0.0, requests>=2.0.0, aiohttp>=3.0.0
environment_variables: 
    - OPENAI_API_KEY (required)

Supports:
- o1 and o1-mini models
- Developer messages (replacing system messages)
- Reasoning tokens tracking and allocation
- Vision processing (o1 only)
- Streaming responses
- Markdown formatting control
- Configurable token limits

Key Features:
- Internal reasoning: Models use reasoning tokens to "think" before responding
- Context windows: o1 (200k tokens), o1-mini (128k tokens)
- Max completion tokens: o1 (100k), o1-mini (65k)
- Recommended reasoning buffer: 25,000 tokens minimum

Note: o1 models are being rolled out gradually to Tier 5 customers only.
Note 2: Tools are not supported. 
"""

import os
import requests
import json
from typing import List, Union, Generator, Iterator, Dict, Optional
from pydantic import BaseModel, Field
from open_webui.utils.misc import pop_system_message
import aiohttp


class Pipe:
    API_VERSION = "2024-03"
    MODEL_URL = "https://api.openai.com/v1/chat/completions"
    SUPPORTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/gif", "image/webp"]

    MODEL_CONFIGS = {
        "o1": {
            "id": "o1",
            "context_window": 200000,
            "max_completion_tokens": 100000,
            "supports_vision": True,
            "supports_reasoning": True,
            "supports_developer_messages": True,
            "supports_tools": False,
        },
        "o1-mini": {
            "id": "o1-mini",
            "context_window": 128000,
            "max_completion_tokens": 65536,
            "supports_vision": False,
            "supports_reasoning": True,
            "supports_developer_messages": True,
            "supports_tools": False,
        },
    }
    REQUEST_TIMEOUT = (3.05, 300)

    class Valves(BaseModel):
        OPENAI_API_KEY: str = Field(
            default=os.getenv("OPENAI_API_KEY", ""), description="Your OpenAI API key"
        )
        REASONING_BUFFER: int = Field(
            default=25000,
            description="Minimum tokens reserved for reasoning (5000-50000)",
            ge=5000,
            le=50000,
        )
        REASONING_RATIO: float = Field(
            default=0.7,
            description="Ratio of max_completion_tokens to allocate for reasoning (0.6-0.9)",
            ge=0.6,
            le=0.9,
        )
        MAX_COMPLETION_TOKENS: Optional[int] = Field(
            default=None,
            description="Override model's default max completion tokens (None=use model default)",
            ge=1,
            le=100000,
            nullable=True,
        )
        ENABLE_MARKDOWN: bool = Field(
            default=False, description="Enable markdown formatting in responses"
        )
        ENABLE_DEVELOPER_MESSAGES: bool = Field(
            default=True,
            description="Convert system messages to developer messages for o1 models",
        )

    def __init__(self):
        self.type = "manifold"
        self.id = "openai-o1"
        self.valves = self.Valves()
        self.request_id = None

    def get_o1_models(self) -> List[dict]:
        return [
            {
                "id": f"openai/{name}",
                "name": self.MODEL_CONFIGS[name]["id"],
                "context_length": self.MODEL_CONFIGS[name]["context_window"],
                "supports_vision": self.MODEL_CONFIGS[name]["supports_vision"],
                "supports_reasoning": True,
            }
            for name in self.MODEL_CONFIGS.keys()
        ]

    def pipes(self) -> List[dict]:
        return self.get_o1_models()

    def process_content(self, content: Union[str, List[dict]]) -> List[dict]:
        if isinstance(content, str):
            return [{"type": "text", "text": content}]

        processed_content = []
        for item in content:
            if item["type"] == "text":
                processed_content.append({"type": "text", "text": item["text"]})
            elif item["type"] == "image_url":
                processed_content.append(self.process_image(item))
        return processed_content

    def process_image(self, image_data):
        if image_data["image_url"]["url"].startswith("data:image"):
            mime_type, base64_data = image_data["image_url"]["url"].split(",", 1)
            media_type = mime_type.split(":")[1].split(";")[0]

            if media_type not in self.SUPPORTED_IMAGE_TYPES:
                raise ValueError(f"Unsupported media type: {media_type}")

            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_data,
                },
            }
        else:
            return {
                "type": "image",
                "source": {"type": "url", "url": image_data["image_url"]["url"]},
            }

    def _estimate_tokens(self, messages: List[dict]) -> int:
        total_tokens = 0
        for message in messages:
            if isinstance(message.get("content"), str):
                total_tokens += len(message["content"]) // 4
            elif isinstance(message.get("content"), list):
                for content in message["content"]:
                    if content.get("type") == "text":
                        total_tokens += len(content["text"]) // 4
                    elif content.get("type") == "image":
                        total_tokens += 1000

        total_tokens += len(messages) * 4
        return total_tokens

    def _process_messages(self, messages: List[dict]) -> List[dict]:
        processed_messages = []
        for message in messages:
            processed_content = []
            content = message.get("content", "")

            role = message["role"]
            if (
                role == "system"
                and self.valves.ENABLE_DEVELOPER_MESSAGES
                and self.MODEL_CONFIGS[message.get("model", "o1")][
                    "supports_developer_messages"
                ]
            ):
                role = "developer"

            if isinstance(content, str):
                processed_content = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                for item in content:
                    if item["type"] == "text":
                        processed_content.append({"type": "text", "text": item["text"]})
                    elif item["type"] == "image_url":
                        if self.MODEL_CONFIGS[message.get("model", "o1")][
                            "supports_vision"
                        ]:
                            processed_content.append(self.process_image(item))

            processed_messages.append(
                {
                    "role": role,
                    "content": (
                        processed_content
                        if len(processed_content) > 1
                        else processed_content[0]["text"]
                    ),
                }
            )

        return processed_messages

    def _handle_response(
        self, response: requests.Response
    ) -> tuple[dict, Optional[dict]]:
        if response.status_code != 200:
            error_msg = f"Error: HTTP {response.status_code}"
            try:
                error_data = response.json().get("error", {})
                error_msg += f": {error_data.get('message', response.text)}"
            except:
                error_msg += f": {response.text}"

            self.request_id = response.headers.get("x-request-id")
            if self.request_id:
                error_msg += f" (Request ID: {self.request_id})"

            return {"content": error_msg, "format": "text"}, None

        result = response.json()
        reasoning_metrics = None
        if "usage" in result:
            usage = result["usage"]
            if "completion_tokens_details" in usage:
                details = usage["completion_tokens_details"]
                reasoning_metrics = {
                    "reasoning_tokens": details.get("reasoning_tokens", 0),
                    "completion_tokens": details.get("accepted_prediction_tokens", 0),
                    "rejected_tokens": details.get("rejected_prediction_tokens", 0),
                }

        return result, reasoning_metrics

    async def _stream_with_ui(
        self, url: str, headers: dict, payload: dict, body: dict, __event_emitter__=None
    ) -> Generator:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    self.request_id = response.headers.get("x-request-id")

                    if response.status != 200:
                        error_msg = (
                            f"Error: HTTP {response.status}: {await response.text()}"
                        )
                        if self.request_id:
                            error_msg += f" (Request ID: {self.request_id})"
                        if __event_emitter__:
                            await __event_emitter__(
                                {
                                    "type": "status",
                                    "data": {
                                        "description": error_msg,
                                        "done": True,
                                    },
                                }
                            )
                        yield error_msg
                        return

                    total_reasoning_tokens = 0
                    total_completion_tokens = 0

                    async for line in response.content:
                        if line and line.startswith(b"data: "):
                            try:
                                data = json.loads(line[6:])

                                if "choices" in data:
                                    choice = data["choices"][0]
                                    if (
                                        "delta" in choice
                                        and "content" in choice["delta"]
                                    ):
                                        yield choice["delta"]["content"]

                                    if choice.get("finish_reason") == "stop":
                                        if "usage" in data:
                                            usage = data["usage"]
                                            if "completion_tokens_details" in usage:
                                                details = usage[
                                                    "completion_tokens_details"
                                                ]
                                                total_reasoning_tokens = details.get(
                                                    "reasoning_tokens", 0
                                                )
                                                total_completion_tokens = details.get(
                                                    "accepted_prediction_tokens", 0
                                                )

                                        if __event_emitter__:
                                            if (
                                                total_reasoning_tokens
                                                or total_completion_tokens
                                            ):
                                                await __event_emitter__(
                                                    {
                                                        "type": "metrics",
                                                        "data": {
                                                            "reasoning_tokens": total_reasoning_tokens,
                                                            "completion_tokens": total_completion_tokens,
                                                        },
                                                    }
                                                )
                                            await __event_emitter__(
                                                {
                                                    "type": "status",
                                                    "data": {
                                                        "description": "Reasoning complete",
                                                        "done": True,
                                                    },
                                                }
                                            )
                                        break

                            except json.JSONDecodeError:
                                continue

        except Exception as e:
            error_msg = f"Stream error: {str(e)}"
            if self.request_id:
                error_msg += f" (Request ID: {self.request_id})"
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": error_msg, "done": True},
                    }
                )
            yield error_msg

    async def pipe(
        self, body: Dict, __event_emitter__=None
    ) -> Union[str, Generator, Iterator]:
        if not self.valves.OPENAI_API_KEY:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Error: OPENAI_API_KEY is required",
                            "done": True,
                        },
                    }
                )
            return {"content": "Error: OPENAI_API_KEY is required", "format": "text"}

        try:
            developer_message, messages = pop_system_message(body["messages"])
            model_name = body["model"].split("/")[-1]
            model_config = self.MODEL_CONFIGS[model_name]

            input_tokens = self._estimate_tokens(messages)
            available_tokens = model_config["context_window"] - input_tokens
            max_completion = min(
                available_tokens,
                self.valves.MAX_COMPLETION_TOKENS
                or model_config["max_completion_tokens"],
            )

            reasoning_tokens = int(max_completion * self.valves.REASONING_RATIO)
            if reasoning_tokens < 25000:
                error_msg = (
                    "The reasoning ratio is set too low for o1 models. These models require "
                    "significant token allocation for reasoning (at least 25,000 tokens). "
                    "Please increase the REASONING_RATIO valve to at least 0.6"
                )
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": error_msg, "done": True},
                        }
                    )
                return {"content": error_msg, "format": "text"}

            max_completion = max_completion - max(
                reasoning_tokens, self.valves.REASONING_BUFFER
            )

            payload = {
                "model": model_config["id"],
                "messages": self._process_messages(messages),
                "max_completion_tokens": max_completion,
                "stream": body.get("stream", False),
                "response_format": body.get("response_format"),
            }

            if self.valves.ENABLE_MARKDOWN and developer_message:
                developer_message = f"{developer_message}\nFormatting reenabled"

            if developer_message:
                payload["messages"].insert(
                    0,
                    {
                        "role": (
                            "developer"
                            if self.valves.ENABLE_DEVELOPER_MESSAGES
                            and model_config["supports_developer_messages"]
                            else "system"
                        ),
                        "content": str(developer_message),
                    },
                )

            payload = {k: v for k, v in payload.items() if v is not None}

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Thinking...", "done": False},
                    }
                )

            headers = {
                "Authorization": f"Bearer {self.valves.OPENAI_API_KEY}",
                "Content-Type": "application/json",
            }

            try:
                if payload["stream"]:
                    return self._stream_with_ui(
                        self.MODEL_URL, headers, payload, body, __event_emitter__
                    )

                response = await self._send_request(self.MODEL_URL, headers, payload)
                if response.status_code != 200:
                    return {
                        "content": f"Error: HTTP {response.status_code}: {response.text}",
                        "format": "text",
                    }

                result, reasoning_metrics = self._handle_response(response)
                response_text = result["choices"][0]["message"]["content"]

                if __event_emitter__ and reasoning_metrics:
                    await __event_emitter__(
                        {
                            "type": "metrics",
                            "data": {
                                "reasoning_tokens": reasoning_metrics[
                                    "reasoning_tokens"
                                ],
                                "completion_tokens": reasoning_metrics[
                                    "completion_tokens"
                                ],
                            },
                        }
                    )

                return response_text

            except requests.exceptions.RequestException as e:
                error_msg = f"Request failed: {str(e)}"
                if self.request_id:
                    error_msg += f" (Request ID: {self.request_id})"

                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": error_msg, "done": True},
                        }
                    )
                return {"content": error_msg, "format": "text"}

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            if self.request_id:
                error_msg += f" (Request ID: {self.request_id})"

            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": error_msg, "done": True}}
                )
            return {"content": error_msg, "format": "text"}
