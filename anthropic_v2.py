"""
title: Anthropic API Integration for OpenWebUI
author: Balaxxe
version: 2.2
license: MIT
requirements: pydantic>=2.0.0, requests>=2.0.0
environment_variables: 
    - ANTHROPIC_API_KEY (required)

Supports:
- All Claude 3 models
- Streaming responses
- Image processing
- Prompt caching (server-side)
- Function calling
- PDF processing
- Cache Control
"""

import os
import requests
import json
import time
import hashlib
import logging
from datetime import datetime
from typing import (
    List,
    Union,
    Generator,
    Iterator,
    Dict,
    Optional,
    AsyncIterator,
)
from pydantic import BaseModel, Field
from open_webui.utils.misc import pop_system_message
import aiohttp


class Pipe:
    API_VERSION = "2023-06-01"
    MODEL_URL = "https://api.anthropic.com/v1/messages"
    SUPPORTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/gif", "image/webp"]
    SUPPORTED_PDF_MODELS = ["claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20240620"]
    MAX_IMAGE_SIZE = 5 * 1024 * 1024
    MAX_PDF_SIZE = 32 * 1024 * 1024
    TOTAL_MAX_IMAGE_SIZE = 100 * 1024 * 1024
    PDF_BETA_HEADER = "pdfs-2024-09-25"
    FUNCTION_BETA_HEADER = "function-calling-2023-09-25"
    # TODO: Fetch model max tokens from the API if available
    MODEL_MAX_TOKENS = {
        "claude-3-opus-20240229": 4096,
        "claude-3-sonnet-20240229": 4096,
        "claude-3-haiku-20240307": 4096,
        "claude-3-5-sonnet-20240620": 8192,
        "claude-3-5-sonnet-20241022": 8192,
        "claude-3-5-haiku-20241022": 8192,
        "claude-3-opus-latest": 4096,
        "claude-3-5-sonnet-latest": 8192,
        "claude-3-5-haiku-latest": 8192,
    }
    BETA_HEADER = "prompt-caching-2024-07-31"
    REQUEST_TIMEOUT = (3.05, 60)

    class Valves(BaseModel):
        ANTHROPIC_API_KEY: str = Field(
            default=os.getenv("ANTHROPIC_API_KEY", ""),
            description="Your Anthropic API key",
        )

    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.type = "manifold"
        self.id = "anthropic"
        self.valves = self.Valves()
        self.request_id = None
        self.default_stop_sequences = ["\n\nHuman:", "<|end-of-output|>"]

    def get_anthropic_models(self) -> List[dict]:
        return [
            {
                "id": f"anthropic/{name}",
                "name": name,
                "context_length": 200000,
                "supports_vision": name != "claude-3-5-haiku-20241022",
            }
            for name in [
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307",
                "claude-3-5-sonnet-20240620",
                "claude-3-5-sonnet-20241022",
                "claude-3-5-haiku-20241022",
                "claude-3-opus-latest",
                "claude-3-5-sonnet-latest",
                "claude-3-5-haiku-latest",
            ]
        ]

    def pipes(self) -> List[dict]:
        return self.get_anthropic_models()

    def process_content(self, content: Union[str, List[dict]]) -> List[dict]:
        if isinstance(content, str):
            return [{"type": "text", "text": content}]

        processed_content = []
        for item in content:
            if item["type"] == "text":
                processed_content.append({"type": "text", "text": item["text"]})
            elif item["type"] == "image_url":
                processed_content.append(self.process_image(item))
            elif item["type"] == "pdf_url":
                model_name = item.get("model", "").split("/")[-1]
                if model_name not in self.SUPPORTED_PDF_MODELS:
                    raise ValueError(
                        f"PDF support is only available for models: {', '.join(self.SUPPORTED_PDF_MODELS)}"
                    )
                processed_content.append(self.process_pdf(item))
        return processed_content

    def process_image(self, image_data):
        if image_data["image_url"]["url"].startswith("data:image"):
            mime_type, base64_data = image_data["image_url"]["url"].split(",", 1)
            media_type = mime_type.split(":")[1].split(";")[0]

            if media_type not in self.SUPPORTED_IMAGE_TYPES:
                raise ValueError(f"Unsupported media type: {media_type}")

            # TODO: Optimize image processing to avoid reading the entire base64 data into memory
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

    def process_pdf(self, pdf_data):
        if pdf_data["pdf_url"]["url"].startswith("data:application/pdf"):
            mime_type, base64_data = pdf_data["pdf_url"]["url"].split(",", 1)

            document = {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": base64_data,
                },
            }
            # TODO: Optimize PDF processing to avoid reading the entire base64 data into memory
            if pdf_data.get("cache_control"):
                document["cache_control"] = pdf_data["cache_control"]

            return document
        else:
            document = {
                "type": "document",
                "source": {"type": "url", "url": pdf_data["pdf_url"]["url"]},
            }

            if pdf_data.get("cache_control"):
                document["cache_control"] = pdf_data["cache_control"]

            return document

    async def pipe(
        self, body: Dict, __event_emitter__=None
    ) -> Union[str, Generator, Iterator]:
        if not self.valves.ANTHROPIC_API_KEY:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Error: ANTHROPIC_API_KEY is required",
                            "done": True,
                        },
                    }
                )
            return {"content": "Error: ANTHROPIC_API_KEY is required", "format": "text"}

        try:
            system_message, messages = pop_system_message(body["messages"])

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Processing request...", "done": False},
                    }
                )

            model_name = body["model"].split("/")[-1]
            max_tokens_limit = self.MODEL_MAX_TOKENS.get(model_name, 4096)

            stop_param = body.get("stop", None)
            stop_sequences = stop_param if isinstance(stop_param, list) else []
            if not stop_sequences:
                stop_sequences = self.default_stop_sequences

            payload = {
                "model": model_name,
                "messages": self._process_messages(messages),
                "max_tokens": min(
                    body.get("max_tokens", max_tokens_limit), max_tokens_limit
                ),
                "temperature": (
                    float(body.get("temperature"))
                    if body.get("temperature") is not None
                    else None
                ),
                "top_k": (
                    int(body.get("top_k")) if body.get("top_k") is not None else None
                ),
                "top_p": (
                    float(body.get("top_p")) if body.get("top_p") is not None else None
                ),
                "stream": body.get("stream"),
                "metadata": body.get("metadata", {}),
                "stop": stop_sequences
            }

            if system_message:
                payload["system"] = str(system_message)

            if "tools" in body:
                payload["tools"] = body["tools"]
                if "tool_choice" in body:
                    payload["tool_choice"] = body["tool_choice"]

            if "response_format" in body:
                payload["response_format"] = {
                    "type": body["response_format"].get("type")
                }

            headers = {
                "x-api-key": self.valves.ANTHROPIC_API_KEY,
                "anthropic-version": self.API_VERSION,
                "content-type": "application/json",
            }

            beta_headers = []
            if any(isinstance(m["content"], list) for m in body["messages"]):
                if "tools" in body:
                    beta_headers.append(self.FUNCTION_BETA_HEADER)

            if beta_headers:
                headers["anthropic-beta"] = ",".join(beta_headers)

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

                result, _ = self._handle_response(response)
                response_text = result["content"][0]["text"]

                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "Request completed successfully",
                                "done": True,
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

                    async for line in response.content:
                        if line and line.startswith(b"data: "):
                            try:
                                data = json.loads(line[6:])
                                if (
                                    data["type"] == "content_block_delta"
                                    and "text" in data["delta"]
                                ):
                                    yield data["delta"]["text"]
                                elif data["type"] == "message_stop":
                                    if __event_emitter__:
                                        await __event_emitter__(
                                            {
                                                "type": "status",
                                                "data": {
                                                    "description": "Request completed",
                                                    "done": True,
                                                },
                                            }
                                        )
                                    break
                            except json.JSONDecodeError as e:
                                logging.error(
                                    f"Failed to parse streaming response: {e}"
                                )
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

    def _process_messages(self, messages: List[dict]) -> List[dict]:
        processed_messages = []
        for message in messages:
            processed_content = []
            for content in self.process_content(message["content"]):
                if (
                    message.get("role") == "assistant"
                    and content.get("type") == "tool_calls"
                ):
                    content["cache_control"] = {"type": "ephemeral"}
                elif (
                    message.get("role") == "user"
                    and content.get("type") == "tool_results"
                ):
                    content["cache_control"] = {"type": "ephemeral"}
                elif content.get("type") == "image":
                    if content["source"]["type"] == "base64":
                        image_size = len(content["source"]["data"]) * 3 / 4
                        if image_size > self.MAX_IMAGE_SIZE:
                            raise ValueError(
                                f"Image size exceeds 5MB limit: {image_size / (1024 * 1024):.2f}MB"
                            )
                        if (
                            content["source"]["media_type"]
                            not in self.SUPPORTED_IMAGE_TYPES
                        ):
                            raise ValueError(
                                f"Unsupported media type: {content['source']['media_type']}"
                            )
                processed_content.append(content)
            processed_messages.append(
                {"role": message["role"], "content": processed_content}
            )
        return processed_messages

    async def _send_request(
        self, url: str, headers: dict, payload: dict
    ) -> requests.Response:
        retry_count = 0
        base_delay = 1  # Start with 1 second delay
        max_retries = 3

        while retry_count < max_retries:
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.REQUEST_TIMEOUT,
                )
                if response.status_code == 429:
                    retry_after = int(
                        response.headers.get(
                            "retry-after", base_delay * (2**retry_count)
                        )
                    )
                    logging.warning(
                        f"Rate limit hit. Retrying in {retry_after} seconds. Retry count: {retry_count + 1}"
                    )
                    time.sleep(retry_after)
                    retry_count += 1
                    continue
                return response
            except requests.exceptions.RequestException as e:
                logging.error(f"Request failed: {str(e)}")
                raise
        logging.error("Max retries exceeded for rate limit.")
        return requests.Response()

    def _handle_response(self, response):
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
        usage = result.get("usage", {})
        cache_metrics = {
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        }
        return result, cache_metrics
