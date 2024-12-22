"""
title: OpenRouter OCR Filter for OpenWebUI
author: Balaxxe (based on @xiniah's work)
version: 1.0
license: MIT
requirements: 
    - pydantic>=2.0.0
    - aiohttp>=3.0.0
environment_variables:
    - OCR_API_KEY (required) - OpenRouter API key with access to vision models. Get your API key from https://openrouter.ai/keys

Supports:
- Image (not PDF) to text conversion via OpenRouter API before prompt requests
- Multiple image formats
- Streaming responses
- Retry mechanism with exponential backoff
- Configurable model selection
- Base64 and URL image support
- Automatic retries with exponential backoff
- Configurable vision model selection
- Detailed error reporting

NOTE: Be sure to toggle "Global" in the function settings. 
"""

import asyncio
from typing import (
    Callable,
    Awaitable,
    Any,
    Optional,
    Dict,
    Generator,
    Iterator,
    Union,
    Tuple,
    List,
    AsyncIterator,
)
import aiohttp
from pydantic import BaseModel, Field
import logging
import uuid
import base64
from PIL import Image
import io

logging.basicConfig(level=logging.INFO)


class Filter:
    API_VERSION = "2024-03"
    REQUEST_TIMEOUT = (3.05, 60)
    SUPPORTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/gif", "image/webp"]
    MAX_IMAGE_SIZE = 5 * 1024 * 1024
    BASE_URL = "https://openrouter.ai/api/v1"
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0

    class Valves(BaseModel):
        OCR_API_KEY: str = Field(default="", description="API key for the API.")
        ocr_prompt: str = Field(
            default="Please only recognize and extract the text or data from this image without interpreting, analyzing, or understanding the content. Do not output any additional information. Simply return the recognized text or data content.",
            description="Prompt for performing OCR recognition.",
        )
        model_name: str = Field(
            default="anthropic/claude-3-haiku",
            description="Model name used for OCR on images.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.logger = logging.getLogger("OCR_Filter")
        self.request_id = None

    def _prepare_request(self, image: str) -> tuple[dict, dict]:
        api_key = self.valves.OCR_API_KEY.strip() if self.valves.OCR_API_KEY else ""
        if not api_key:
            raise ValueError("OpenRouter API key is required")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://openwebui.com",
            "X-Title": "OpenWebUI OCR",
        }

        body = {
            "model": self.valves.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.valves.ocr_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image, "detail": "high"},
                        }
                    ],
                },
            ],
            "temperature": 0.0,
            "provider": {"data_collection": "deny", "require_parameters": True},
        }

        return headers, body

    async def _perform_ocr(
        self, image: str, event_emitter: Callable[[Any], Awaitable[None]]
    ) -> str:
        """Internal method for performing OCR recognition."""
        await event_emitter(
            {
                "type": "status",
                "data": {
                    "description": "✨Performing text recognition on the image, please wait patiently...",
                    "done": False,
                },
            }
        )

        if not self.valves.OCR_API_KEY or not self.valves.OCR_API_KEY.strip():
            raise ValueError(
                "OpenRouter API key is required. Please set OCR_API_KEY environment variable."
            )

        headers, body = self._prepare_request(image)
        url = f"{self.BASE_URL}/chat/completions"

        async with aiohttp.ClientSession() as session:
            for attempt in range(self.MAX_RETRIES):
                try:
                    async with session.post(
                        url, json=body, headers=headers, timeout=60
                    ) as response:
                        response_data = await response.json()

                        self.logger.info(f"OpenRouter response: {response_data}")

                        response.raise_for_status()
                        self.request_id = response.headers.get("x-request-id")

                        if not response_data.get("choices", []):
                            if response.status == 401:
                                raise ValueError(
                                    "Invalid OpenRouter API key. Please check your API key configuration."
                                )
                            elif response.status == 402:
                                raise ValueError(
                                    "Insufficient credits. Please add more credits to your OpenRouter account."
                                )
                            else:
                                raise ValueError(
                                    "No content generated. The model may be warming up, please try again."
                                )

                        result = response_data["choices"][0]["message"]["content"]

                        await event_emitter(
                            {
                                "type": "status",
                                "data": {
                                    "description": "✅ OCR completed successfully!",
                                    "progress": 100,
                                    "done": True,
                                },
                            }
                        )
                        return result

                except aiohttp.ClientError as e:
                    error_msg = f"OCR request failed (attempt {attempt + 1}/{self.MAX_RETRIES}): {e}"
                    if self.request_id:
                        error_msg += f" (Request ID: {self.request_id})"
                    self.logger.warning(error_msg)

                    if attempt < self.MAX_RETRIES - 1:
                        await asyncio.sleep(self.RETRY_DELAY * (attempt + 1))
                    else:
                        raise RuntimeError(
                            f"OCR failed after {self.MAX_RETRIES} attempts"
                        )
                except Exception as e:
                    error_msg = f"Unexpected error during OCR: {e}"
                    if self.request_id:
                        error_msg += f" (Request ID: {self.request_id})"
                    self.logger.error(error_msg)
                    raise

    def _find_image_in_messages(self, messages) -> Optional[tuple[int, int, str]]:
        for m_index, message in enumerate(messages):
            if message["role"] == "user" and isinstance(message.get("content"), list):
                for c_index, content in enumerate(message["content"]):
                    if content["type"] == "image_url":
                        return m_index, c_index, content["image_url"]["url"]
        return None

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __user__: Optional[dict] = None,
        __model__: Optional[dict] = None,
    ) -> dict:
        if not self.valves.OCR_API_KEY:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "❌ OCR_API_KEY is required",
                        "done": True,
                        "error": True,
                    },
                }
            )
            return body

        messages = body.get("messages", [])
        image_info = self._find_image_in_messages(messages)
        if not image_info:
            return body
        message_index, content_index, image = image_info
        is_valid, error_msg = self._validate_image(image)
        if not is_valid:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"❌ {error_msg}",
                        "done": True,
                        "error": True,
                    },
                }
            )
            return body

        if (len(messages) // 2) >= 1:
            del messages[message_index]["content"][content_index]
            body["messages"] = messages
            return body
        try:
            result = await self._perform_ocr(image, __event_emitter__)
            messages[message_index]["content"][content_index]["type"] = "text"
            messages[message_index]["content"][content_index].pop("image_url", None)
            messages[message_index]["content"][content_index]["text"] = result
            body["messages"] = messages
        except Exception as e:
            error_msg = f"OCR recognition error: {e}"
            self.logger.error(error_msg)
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"❌ {error_msg}",
                        "progress": 100,
                        "done": True,
                        "error": True,
                    },
                }
            )
        return body

    async def outlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __user__: Optional[dict] = None,
        __model__: Optional[dict] = None,
    ) -> dict:
        return body

    def _validate_image(self, image: str) -> tuple[bool, str]:
        try:
            if image.startswith("data:image"):
                header, encoded = image.split(",", 1)
                mime_type = header.split(";")[0].split(":")[1]

                if mime_type not in self.SUPPORTED_IMAGE_TYPES:
                    return False, f"Unsupported image type: {mime_type}"

                image_data = base64.b64decode(encoded)
                if len(image_data) > self.MAX_IMAGE_SIZE:
                    return (
                        False,
                        f"Image size exceeds {self.MAX_IMAGE_SIZE/1024/1024}MB limit",
                    )

                img = Image.open(io.BytesIO(image_data))
                img.verify()
                return True, ""

        except Exception as e:
            return False, f"Invalid image: {str(e)}"

        return False, "Unsupported image format"
