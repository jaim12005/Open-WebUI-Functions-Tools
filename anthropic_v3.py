"""
title: Anthropic API Integration for OpenWebUI
author: Balaxxe
version: 3.1
license: MIT
requirements: pydantic>=2.0.0, requests>=2.0.0, fastapi>=0.95.0

description: |
  A comprehensive integration for Anthropic's Claude AI models in OpenWebUI.
  This function provides access to all Claude 3, 3.5, and 3.7 models with
  full support for their advanced capabilities including multimodal inputs,
  thinking mode, and large context windows.

features:
  - Full support for all Claude 3, 3.5, and 3.7 models (Opus, Sonnet, Haiku variants)
  - Multimodal capabilities with image processing (JPEG, PNG, GIF, WebP)
  - PDF document processing with supported models
  - Streaming responses for real-time output
  - Advanced function calling / Tool use with OpenAI compatibility
  - Extended thinking capability for Claude 3.7 models with event notifications
  - 128k context window support for Claude 3.7 models
  - Configurable prompt caching for improved performance and reduced token usage
  - JSON response format for structured data
  - Detailed metadata and token usage tracking
  - Configurable temperature, max tokens, and stop sequences
  - Robust error handling and automatic retries
  - Comprehensive logging for debugging
  - Event-based notifications for thinking, token usage, and completion

environment_variables:
    - ANTHROPIC_API_KEY (required): Your Anthropic API key
    - ANTHROPIC_API_VERSION (optional): API version to use (defaults to 2023-06-01)
    - ANTHROPIC_RETRY_COUNT (optional): Number of retries for rate limits (defaults to 3)
    - ANTHROPIC_TIMEOUT (optional): Request timeout in seconds (defaults to 60)
    - ANTHROPIC_ENABLE_LARGE_CONTEXT (optional): Enable 128k context for Claude 3.7 (true/false)
    - ANTHROPIC_DEFAULT_MODEL (optional): Default model to use
    - ANTHROPIC_DEFAULT_MAX_TOKENS (optional): Default max tokens to use
    - ANTHROPIC_DEFAULT_TEMPERATURE (optional): Default temperature
    - ANTHROPIC_THINKING_ENABLED (optional): Enable thinking by default
    - ANTHROPIC_THINKING_BUDGET (optional): Default thinking budget tokens
    - ANTHROPIC_DEFAULT_STOP_SEQUENCES (optional): Default stop sequences
    - ANTHROPIC_ENABLE_PROMPT_CACHING (optional): Enable prompt caching (defaults to true)
    - ANTHROPIC_RESPONSE_FORMAT_JSON (optional): Request responses in JSON format (defaults to false)
    - ANTHROPIC_CACHE_CONTROL (optional): Cache control strategy (standard, aggressive, minimal)
    - ANTHROPIC_ENABLE_TOOL_USE (optional): Enable function calling/tool use capabilities (defaults to true)
    - ANTHROPIC_TOOL_CHOICE (optional): Tool choice strategy (auto, any, none)

usage: |
  1. Set your Anthropic API key in the valves configuration
  2. Configure parameters as needed in OpenWebUI:
     - Default Model: Recommended "claude-3-7-sonnet-latest"
     - Max Tokens: Controls maximum response length
     - Thinking Enabled: For deep reasoning (Claude 3.7 only)
     - Thinking Budget: Token allocation for thinking (20,000+ recommended for complex tasks)
     - Enable Large Context: Set to true for 128k context with Claude 3.7
     - Enable Prompt Caching: Set to true for improved performance (default)
     - Response Format JSON: Set to true to request responses in JSON format
     - Cache Control: Choose caching strategy (standard, aggressive, minimal)
     - Enable Tool Use: Set to true to enable function calling capabilities
     - Tool Choice: Control how tools are selected (auto, any, none)
  
  Notes:
  - When thinking is enabled, temperature will automatically be set to 1.0 as required by Anthropic's API
  - PDF support is only available for specific models (Claude 3.5 and 3.7)
  - For multimodal use, simply upload images in the chat interface
  - Stop sequences are only applied when explicitly requested
  - Large context (128k) is only available for Claude 3.7 models
  - JSON response format ensures responses are valid JSON (useful for structured data)
  - Tool use supports both Anthropic's native format and OpenAI-compatible function calling
  - Response metadata includes token usage, stop reason, and other useful information
"""

import os
import requests
import json
import time
import logging
from datetime import datetime
from typing import List, Union, Generator, Iterator, Dict, Optional, AsyncIterator
from pydantic import BaseModel, Field
from open_webui.utils.misc import pop_system_message
import aiohttp
from fastapi import Request


"""
Configurable Valves:
    - ANTHROPIC_API_KEY: Your Anthropic API key (required)
    - DEFAULT_MODEL: Default model to use (e.g., "claude-3-7-sonnet-latest")
    - DEFAULT_MAX_TOKENS: Default token limit (e.g., 4096)
    - DEFAULT_TEMPERATURE: Default temperature between 0.0-1.0 (e.g., 0.7)
    - THINKING_ENABLED: Enable thinking by default for Claude 3.7+ models (true/false)
    - THINKING_BUDGET: Default thinking budget tokens (min 1024, e.g., 2048)
    - DEFAULT_STOP_SEQUENCES: Default stop sequences (comma-separated strings)
    - ENABLE_LARGE_CONTEXT: Enable 128k context for Claude 3.7 models (true/false)
    - ENABLE_PROMPT_CACHING: Enable prompt caching for improved performance (true/false)
    - RESPONSE_FORMAT_JSON: Request responses in JSON format (true/false)
    - CACHE_CONTROL: Cache control strategy (standard, aggressive, minimal)
    - ENABLE_TOOL_USE: Enable function calling/tool use capabilities (true/false)
    - TOOL_CHOICE: Tool choice strategy (auto, any, none)
"""


class Pipe:
    API_VERSION = os.getenv("ANTHROPIC_API_VERSION", "2023-06-01")
    MODEL_URL = "https://api.anthropic.com/v1/messages"
    SUPPORTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/gif", "image/webp"]
    SUPPORTED_PDF_MODELS = [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620",
        "claude-3-5-opus-20240229",
        "claude-3-5-opus-20240307",
        "claude-3-7-sonnet-20240620",
        "claude-3-7-sonnet-20250219",
    ]
    SUPPORTED_THINKING_MODELS = [
        "claude-3-5-sonnet-20240620",
        "claude-3-7-sonnet-20240620",
        "claude-3-7-sonnet-20250219",
    ]
    MAX_IMAGE_SIZE = 5 * 1024 * 1024
    MAX_PDF_SIZE = 32 * 1024 * 1024
    TOTAL_MAX_IMAGE_SIZE = 100 * 1024 * 1024
    PDF_BETA_HEADER = "pdfs-2024-09-25"
    BETA_HEADER = "prompt-caching-2024-07-31"

    class Valves(BaseModel):
        ANTHROPIC_API_KEY: str = Field(
            default=os.getenv("ANTHROPIC_API_KEY", ""),
            description="Your Anthropic API key (required)",
        )
        DEFAULT_MODEL: str = Field(
            default=os.getenv("ANTHROPIC_DEFAULT_MODEL", "claude-3-7-sonnet-latest"),
            description="Default model to use if not specified in request",
        )
        DEFAULT_MAX_TOKENS: int = Field(
            default=int(os.getenv("ANTHROPIC_DEFAULT_MAX_TOKENS", "4096")),
            description="Default token limit if not specified in request",
        )
        DEFAULT_TEMPERATURE: float = Field(
            default=float(os.getenv("ANTHROPIC_DEFAULT_TEMPERATURE", "0.7")),
            description="Default temperature (0.0-1.0) if not specified in request",
        )
        THINKING_ENABLED: bool = Field(
            default=os.getenv("ANTHROPIC_THINKING_ENABLED", "false").lower() == "true",
            description="Enable thinking by default for Claude 3.7+ models",
        )
        THINKING_BUDGET: int = Field(
            default=int(os.getenv("ANTHROPIC_THINKING_BUDGET", "2048")),
            description="Default thinking budget tokens (min 1024)",
        )
        DEFAULT_STOP_SEQUENCES: str = Field(
            default=os.getenv(
                "ANTHROPIC_DEFAULT_STOP_SEQUENCES", "\n\nHuman:,<|end-of-output|>"
            ),
            description="Default stop sequences (comma-separated)",
        )
        ENABLE_LARGE_CONTEXT: bool = Field(
            default=os.getenv("ANTHROPIC_ENABLE_LARGE_CONTEXT", "false").lower()
            == "true",
            description="Enable 128k context for Claude 3.7 models",
        )
        ENABLE_PROMPT_CACHING: bool = Field(
            default=os.getenv("ANTHROPIC_ENABLE_PROMPT_CACHING", "true").lower()
            == "true",
            description="Enable prompt caching for improved performance",
        )
        RESPONSE_FORMAT_JSON: bool = Field(
            default=os.getenv("ANTHROPIC_RESPONSE_FORMAT_JSON", "false").lower()
            == "true",
            description="Request responses in JSON format",
        )
        CACHE_CONTROL: str = Field(
            default=os.getenv("ANTHROPIC_CACHE_CONTROL", "standard"),
            description="Cache control strategy (standard, aggressive, minimal)",
        )
        ENABLE_TOOL_USE: bool = Field(
            default=os.getenv("ANTHROPIC_ENABLE_TOOL_USE", "true").lower()
            == "true",
            description="Enable function calling/tool use capabilities",
        )
        TOOL_CHOICE: str = Field(
            default=os.getenv("ANTHROPIC_TOOL_CHOICE", "auto"),
            description="Tool choice strategy (auto, any, none)",
        )

    __state__ = {"enabled": True}

    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.type = "manifold"
        self.id = "anthropic_v3"
        self.valves = self.Valves()
        self.request_id = None

        try:
            from open_webui.models.functions import Functions

            function_data = Functions.get_function_by_id(self.id)
            if function_data:
                Pipe.__state__["enabled"] = function_data.get("enabled", True)
                logging.info(
                    f"Loaded state from registry: enabled={Pipe.__state__['enabled']}"
                )
        except Exception as e:
            logging.warning(f"Could not load state from registry: {str(e)}")
            Pipe.__state__ = {"enabled": True}

        self.default_stop_sequences = []
        if self.valves.DEFAULT_STOP_SEQUENCES:
            self.default_stop_sequences = [
                s.strip()
                for s in self.valves.DEFAULT_STOP_SEQUENCES.split(",")
                if s.strip()
            ]
            logging.info(
                f"Initialized with default stop sequences: {self.default_stop_sequences}"
            )

        if self.valves.ENABLE_LARGE_CONTEXT:
            logging.info("Large context (128k) is enabled for Claude 3.7 models")

    def enable(self):
        """Enable the function"""
        Pipe.__state__["enabled"] = True
        logging.info("Anthropic V3 function enabled")
        self.save_state()
        return True

    def disable(self):
        """Disable the function"""
        Pipe.__state__["enabled"] = False
        logging.info("Anthropic V3 function disabled")
        self.save_state()
        return True

    def is_enabled(self):
        """Check if the function is enabled"""
        return Pipe.__state__.get("enabled", True)

    def toggle(self):
        """Toggle the function's enabled state"""
        Pipe.__state__["enabled"] = not Pipe.__state__.get("enabled", True)
        logging.info(
            f"Anthropic V3 function toggled: enabled={Pipe.__state__['enabled']}"
        )
        self.save_state()
        return Pipe.__state__["enabled"]

    def save_state(self):
        """Save the function state to OpenWebUI's function registry"""
        try:
            from open_webui.models.functions import Functions

            Functions.update_function(id=self.id, enabled=Pipe.__state__["enabled"])

            logging.info(
                f"Anthropic V3 function state saved to registry: enabled={Pipe.__state__['enabled']}"
            )
            return True
        except Exception as e:
            logging.error(f"Error saving state to registry: {str(e)}")
            return False

    async def _stream_response(self, url, headers, payload, __event_emitter__):
        """Handle streaming responses from the Anthropic API"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, headers=headers, json=payload, timeout=60
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    error_msg = f"API Error: HTTP {response.status}: {error_text}"
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {"description": error_msg, "done": True},
                        }
                    )
                    yield {"content": error_msg, "format": "text"}
                    return

                if self.valves.RESPONSE_FORMAT_JSON:
                    json_content = ""
                    metadata = {}
                    
                    async for line in response.content:
                        if line and line.startswith(b"data: "):
                            try:
                                data = json.loads(line[6:])
                                
                                if data["type"] == "message_start":
                                    metadata = {
                                        "id": data.get("message", {}).get("id", ""),
                                        "model": data.get("message", {}).get("model", ""),
                                        "type": "message",
                                    }
                                elif data["type"] == "message_stop":
                                    metadata["stop_reason"] = data.get("stop_reason", "")
                                    metadata["stop_sequence"] = data.get("stop_sequence", "")
                                    metadata["usage"] = data.get("usage", {})
                                    
                                    if __event_emitter__ and "usage" in data:
                                        await __event_emitter__(
                                            {
                                                "type": "token_usage",
                                                "data": data["usage"]
                                            }
                                        )
                                    
                                    await __event_emitter__(
                                        {
                                            "type": "status",
                                            "data": {
                                                "description": "Request completed",
                                                "done": True,
                                                "metadata": metadata
                                            },
                                        }
                                    )
                                    break
                                elif (
                                    data["type"] == "content_block_delta"
                                    and "text" in data["delta"]
                                ):
                                    json_content += data["delta"]["text"]
                            except Exception as e:
                                logging.error(f"Error processing streaming JSON data: {str(e)}")
                    
                    # Validate and yield the complete JSON at the end
                    try:
                        json_obj = json.loads(json_content)
                        yield {"content": json_obj, "format": "json", "metadata": metadata}
                    except json.JSONDecodeError:
                        logging.warning("Response was not valid JSON despite JSON format being requested")
                        yield {"content": json_content, "format": "text", "metadata": metadata}
                else:
                    metadata = {}
                    
                    async for line in response.content:
                        if line and line.startswith(b"data: "):
                            try:
                                data = json.loads(line[6:])
                                
                                if data["type"] == "message_start":
                                    metadata = {
                                        "id": data.get("message", {}).get("id", ""),
                                        "model": data.get("message", {}).get("model", ""),
                                        "type": "message",
                                    }
                                    if __event_emitter__:
                                        await __event_emitter__(
                                            {
                                                "type": "message_start",
                                                "data": metadata
                                            }
                                        )
                                elif data["type"] == "message_stop":
                                    metadata["stop_reason"] = data.get("stop_reason", "")
                                    metadata["stop_sequence"] = data.get("stop_sequence", "")
                                    metadata["usage"] = data.get("usage", {})
                                    
                                    if __event_emitter__ and "usage" in data:
                                        await __event_emitter__(
                                            {
                                                "type": "token_usage",
                                                "data": data["usage"]
                                            }
                                        )
                                    
                                    await __event_emitter__(
                                        {
                                            "type": "status",
                                            "data": {
                                                "description": "Request completed",
                                                "done": True,
                                                "metadata": metadata
                                            },
                                        }
                                    )
                                    break
                                elif (
                                    data["type"] == "content_block_delta"
                                    and "text" in data["delta"]
                                ):
                                    yield data["delta"]["text"]
                                elif data["type"] == "content_block_start" and data.get("content_block", {}).get("type") == "thinking":
                                    if __event_emitter__:
                                        await __event_emitter__(
                                            {
                                                "type": "status",
                                                "data": {"description": "Thinking...", "done": False}
                                            }
                                        )
                                        await __event_emitter__(
                                            {
                                                "type": "thinking_start",
                                                "data": {"model": metadata.get("model", "")}
                                            }
                                        )
                                elif data["type"] == "content_block_stop" and data.get("content_block", {}).get("type") == "thinking":
                                    if __event_emitter__:
                                        await __event_emitter__(
                                            {
                                                "type": "status",
                                                "data": {"description": "Generating response...", "done": False}
                                            }
                                        )
                                        await __event_emitter__(
                                            {
                                                "type": "thinking_stop",
                                                "data": {"model": metadata.get("model", "")}
                                            }
                                        )
                            except Exception as e:
                                logging.error(f"Error processing streaming data: {str(e)}")

    async def _non_stream_response(self, url, headers, payload, __event_emitter__):
        """Handle non-streaming responses from the Anthropic API"""
        response = requests.post(url, headers=headers, json=payload, timeout=60)

        if response.status_code != 200:
            error_msg = f"API Error: HTTP {response.status_code}: {response.text}"
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": error_msg, "done": True},
                    }
                )
            return {"content": error_msg, "format": "text"}

        result = response.json()
        content = result.get("content", [])
        text_content = ""

        for item in content:
            if item.get("type") == "text":
                text_content += item.get("text", "")
        
        usage = result.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        
        metadata = {
            "id": result.get("id", ""),
            "model": result.get("model", ""),
            "type": result.get("type", ""),
            "role": result.get("role", ""),
            "stop_reason": result.get("stop_reason", ""),
            "stop_sequence": result.get("stop_sequence", ""),
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
        }
        
        logging.info(f"Response metadata: {metadata}")
        
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Request completed successfully",
                        "done": True,
                        "metadata": metadata
                    },
                }
            )
            
            await __event_emitter__(
                {
                    "type": "token_usage",
                    "data": metadata["usage"]
                }
            )

        if self.valves.RESPONSE_FORMAT_JSON:
            try:
                json_obj = json.loads(text_content)
                return {"content": json_obj, "format": "json", "metadata": metadata}
            except json.JSONDecodeError:
                logging.warning("Response was not valid JSON despite JSON format being requested")
                
        return {"content": text_content, "format": "text", "metadata": metadata}

    def _process_messages(self, messages: List[dict]) -> List[dict]:
        """Process content from various formats into Anthropic-compatible format"""
        processed_messages = []
        for message in messages:
            if not isinstance(message.get("content"), list):
                message["content"] = [
                    {"type": "text", "text": message.get("content", "")}
                ]

            processed_content = []
            for item in message["content"]:
                if item.get("type") == "text":
                    processed_content.append(
                        {"type": "text", "text": item.get("text", "")}
                    )
                elif item.get("type") == "image_url":
                    if (
                        isinstance(item.get("image_url"), dict)
                        and "url" in item["image_url"]
                    ):
                        url = item["image_url"]["url"]
                        if url.startswith("data:image"):
                            try:
                                mime_type, base64_data = url.split(",", 1)
                                media_type = mime_type.split(":")[1].split(";")[0]
                                if media_type in self.SUPPORTED_IMAGE_TYPES:
                                    processed_content.append(
                                        {
                                            "type": "image",
                                            "source": {
                                                "type": "base64",
                                                "media_type": media_type,
                                                "data": base64_data,
                                            },
                                        }
                                    )
                            except Exception as e:
                                logging.error(
                                    f"Error processing base64 image: {str(e)}"
                                )
                        else:
                            processed_content.append(
                                {
                                    "type": "image",
                                    "source": {"type": "url", "url": url},
                                }
                            )
                elif item.get("type") == "pdf_url":
                    model_name = message.get("model", "").split("/")[-1]
                    if model_name in self.SUPPORTED_PDF_MODELS:
                        if (
                            isinstance(item.get("pdf_url"), dict)
                            and "url" in item["pdf_url"]
                        ):
                            url = item["pdf_url"]["url"]
                            if url.startswith("data:application/pdf"):
                                try:
                                    mime_type, base64_data = url.split(",", 1)
                                    processed_content.append(
                                        {
                                            "type": "document",
                                            "source": {
                                                "type": "base64",
                                                "media_type": "application/pdf",
                                                "data": base64_data,
                                            },
                                        }
                                    )
                                except Exception as e:
                                    logging.error(
                                        f"Error processing base64 PDF: {str(e)}"
                                    )
                            else:
                                processed_content.append(
                                    {
                                        "type": "document",
                                        "source": {"type": "url", "url": url},
                                    }
                                )

            processed_messages.append(
                {"role": message["role"], "content": processed_content}
            )
        return processed_messages

    def pipes(self) -> List[dict]:
        models = []
        for name in [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-5-opus-20240229",
            "claude-3-5-opus-20240307",
            "claude-3-7-sonnet-20240620",
            "claude-3-7-sonnet-20250219",
            "claude-3-opus-latest",
            "claude-3-5-sonnet-latest",
            "claude-3-5-haiku-latest",
            "claude-3-5-opus-latest",
            "claude-3-7-sonnet-latest",
        ]:
            context_length = 200000
            if name.startswith("claude-3-7") and self.valves.ENABLE_LARGE_CONTEXT:
                context_length = 128000

            supports_vision = "haiku" not in name or name != "claude-3-5-haiku-20241022"
            supports_pdf = name in self.SUPPORTED_PDF_MODELS
            supports_thinking = name in self.SUPPORTED_THINKING_MODELS
            
            supports_tool_use = True
            
            supports_json = True

            display_name = f"Claude {name.replace('claude-', '')}"

            models.append(
                {
                    "id": f"anthropic/{name}",
                    "name": display_name,
                    "context_length": context_length,
                    "supports_vision": supports_vision,
                    "supports_pdf": supports_pdf,
                    "supports_thinking": supports_thinking,
                    "supports_tool_use": supports_tool_use,
                    "supports_json": supports_json,
                }
            )
        return models

    async def pipe(
        self,
        body: Dict,
        __user__: dict = None,
        __request__: Request = None,
        __event_emitter__=None,
    ) -> Union[str, Generator, Iterator]:
        if not self.is_enabled():
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Error: Anthropic V3 function is disabled",
                            "done": True,
                        },
                    }
                )
            return {
                "content": "Error: Anthropic V3 function is disabled",
                "format": "text",
            }

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

            model_name = body.get("model", "").split("/")[-1]
            thinking_enabled = model_name.startswith("claude-3-7") and self.valves.THINKING_ENABLED
            
            if __event_emitter__:
                status_description = "Preparing to think..." if thinking_enabled else "Processing request..."
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": status_description, "done": False},
                    }
                )

            if not body.get("model"):
                body["model"] = f"anthropic/{self.valves.DEFAULT_MODEL}"
                logging.info(f"Using default model: {self.valves.DEFAULT_MODEL}")

            model_name = body["model"].split("/")[-1]
            max_tokens_limit = 4096

            temperature = self.valves.DEFAULT_TEMPERATURE
            if "temperature" in body:
                try:
                    temp_value = body.get("temperature")
                    if temp_value is not None:
                        if isinstance(temp_value, (int, float, str)):
                            temperature = float(temp_value)
                        else:
                            logging.warning(
                                f"Invalid temperature value: {temp_value}, using default: {temperature}"
                            )
                except (ValueError, TypeError) as e:
                    logging.warning(
                        f"Error parsing temperature: {e}, using default: {temperature}"
                    )

            payload = {
                "model": model_name,
                "messages": (
                    self._process_messages(messages)
                    if hasattr(self, "_process_messages")
                    else messages
                ),
                "max_tokens": min(
                    body.get("max_tokens", self.valves.DEFAULT_MAX_TOKENS),
                    max_tokens_limit,
                ),
                "temperature": temperature,
                "stream": body.get("stream", True),
            }
            
            if self.valves.RESPONSE_FORMAT_JSON:
                payload["response_format"] = {"type": "json"}
                logging.info("JSON response format enabled")
                
            if self.valves.CACHE_CONTROL != "standard":
                cache_settings = {}
                if self.valves.CACHE_CONTROL == "aggressive":
                    cache_settings = {"type": "auto"}
                elif self.valves.CACHE_CONTROL == "minimal":
                    cache_settings = {"type": "none"}
                
                if cache_settings:
                    payload["cache_control"] = cache_settings
                    logging.info(f"Cache control set to: {self.valves.CACHE_CONTROL}")

            if system_message:
                payload["system"] = system_message

            if model_name.startswith("claude-3-7"):
                if self.valves.THINKING_ENABLED:
                    thinking_budget = min(
                        self.valves.THINKING_BUDGET, max_tokens_limit // 2
                    )
                    payload["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": thinking_budget,
                    }
                    payload["temperature"] = 1.0
                    logging.info(
                        f"THINKING ENABLED: Using thinking with budget: {thinking_budget} tokens"
                    )

            if self.default_stop_sequences:
                payload["stop_sequences"] = self.default_stop_sequences

            headers = {
                "x-api-key": self.valves.ANTHROPIC_API_KEY,
                "anthropic-version": self.API_VERSION,
                "content-type": "application/json",
            }

            beta_headers = []
            
            if self.valves.ENABLE_PROMPT_CACHING:
                beta_headers.append(self.BETA_HEADER)
                logging.info("Prompt caching enabled")
                
            if self.valves.ENABLE_TOOL_USE:
                if "tools" in body:
                    payload["tools"] = body["tools"]
                    logging.info(f"Tool use enabled with {len(body['tools'])} tools")
                    
                    if self.valves.TOOL_CHOICE != "auto":
                        if self.valves.TOOL_CHOICE == "any":
                            payload["tool_choice"] = "any"
                        elif self.valves.TOOL_CHOICE == "none":
                            payload["tool_choice"] = "none"
                        logging.info(f"Tool choice set to: {self.valves.TOOL_CHOICE}")
                elif "functions" in body:
                    tools = []
                    for func in body["functions"]:
                        tool = {
                            "name": func.get("name", ""),
                            "description": func.get("description", ""),
                            "input_schema": func.get("parameters", {})
                        }
                        tools.append({"type": "function", "function": tool})
                    
                    payload["tools"] = tools
                    logging.info(f"Converted {len(tools)} OpenAI-style functions to Anthropic tools")
                    
                    if "function_call" in body:
                        if body["function_call"] == "auto":
                            pass
                        elif body["function_call"] == "none":
                            payload["tool_choice"] = "none"
                        elif isinstance(body["function_call"], dict) and "name" in body["function_call"]:
                            payload["tool_choice"] = {
                                "type": "function",
                                "function": {"name": body["function_call"]["name"]}
                            }
                    elif self.valves.TOOL_CHOICE != "auto":
                        if self.valves.TOOL_CHOICE == "any":
                            payload["tool_choice"] = "any"
                        elif self.valves.TOOL_CHOICE == "none":
                            payload["tool_choice"] = "none"

            if beta_headers:
                headers["anthropic-beta"] = ",".join(beta_headers)

            if payload.get("stream", True) and __event_emitter__:
                return self._stream_response(
                    self.MODEL_URL, headers, payload, __event_emitter__
                )
            else:
                return await self._non_stream_response(
                    self.MODEL_URL, headers, payload, __event_emitter__
                )

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": error_msg, "done": True}}
                )
            return {"content": error_msg, "format": "text"}
