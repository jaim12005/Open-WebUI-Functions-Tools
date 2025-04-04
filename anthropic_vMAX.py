"""
title: Anthropic API Integration for OpenWebUI
author: Balaxxe
version: 1.0
license: MIT
requirements: pydantic>=2.0.0, requests>=2.0.0, aiohttp>=3.8.0

description: |
  A comprehensive integration for Anthropic's Claude AI models in OpenWebUI.
  This function provides access to Claude 3, 3.5, and 3.7 models.
  It supports advanced capabilities including multimodal inputs (images/PDFs),
  extended thinking, tool use, prompt caching, and large context windows.
  Maximum output token limits are dynamically set based on the specific model
  and operating mode (e.g., normal, thinking, 128k beta). Includes fix for loading state from registry.

features:
  - Support for Claude 3, 3.5, and 3.7 models.
  - Dynamic max output token limits based on model/mode (4k/8k/64k/128k).
  - Multimodal capabilities: Image (JPEG, PNG, GIF, WebP) and PDF processing (for supported models, requires beta header).
  - Streaming responses for real-time output.
  - Advanced Tool Use: Supports Anthropic's native format and provides compatibility for OpenAI's function/tool calling format. Configurable tool choice.
  - Extended Thinking: Utilizes the 'thinking' parameter for supported models (e.g., Claude 3.7) with event notifications. Requires specific models.
  - 200k context window support for all listed models.
  - Extended Output: Supports up to 128K output tokens for specific models (e.g., 3.7 Sonnet) when enabled via valve and beta header. Cannot be used with Thinking simultaneously.
  - Prompt Caching: Configurable via valve to potentially improve performance (requires beta header).
  - JSON Response Format: Option to enforce JSON output structure.
  - Detailed Metadata: Includes token usage (including cache tokens), stop reason, etc. in response metadata and events.
  - Configurable Parameters: Temperature, max output tokens (request/valve can cap below model max), stop sequences.
  - Error Handling: Includes basic retries for non-streaming requests on transient errors.
  - Comprehensive Logging: Detailed logs for debugging parameter calculation and API interaction.
  - Event Emission: Provides status, token usage, and thinking start/stop events to OpenWebUI.
  - Correctly loads enabled/disabled state from OpenWebUI registry.

environment_variables:
    - ANTHROPIC_API_KEY (required): Your Anthropic API key.
    - ANTHROPIC_API_VERSION (optional): API version (defaults to 2023-06-01).
    - ANTHROPIC_RETRY_COUNT (optional): Retries for transient errors (non-streaming, defaults to 3).
    - ANTHROPIC_TIMEOUT (optional): Request timeout in seconds (defaults to 120).
    - ANTHROPIC_DEFAULT_MODEL (optional): Default model if not specified in request (e.g., "claude-3-5-sonnet-latest").
    - ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS (optional): Default max output tokens. Set to 0 to use the model's dynamic maximum; otherwise, this value acts as a cap if lower than the model's max.
    - ANTHROPIC_DEFAULT_TEMPERATURE (optional): Default temperature (0.0-1.0, e.g., 0.7).
    - ANTHROPIC_THINKING_ENABLED (optional): Enable thinking by default for supported models (true/false).
    - ANTHROPIC_THINKING_BUDGET (optional): Default thinking budget tokens (min 1024, capped below final max_tokens).
    - ANTHROPIC_DEFAULT_STOP_SEQUENCES (optional): Default stop sequences (comma-separated).
    - ANTHROPIC_ENABLE_PROMPT_CACHING (optional): Enable prompt caching beta feature (true/false, defaults true).
    - ANTHROPIC_ENABLE_EXTENDED_OUTPUT (optional): Enable the 128K extended output beta feature for supported models (true/false).
    - ANTHROPIC_RESPONSE_FORMAT_JSON (optional): Force JSON output format (true/false).
    - ANTHROPIC_CACHE_CONTROL (optional): Cache control strategy (standard, aggressive, minimal).
    - ANTHROPIC_ENABLE_TOOL_USE (optional): Enable tool use capabilities (true/false, defaults true).
    - ANTHROPIC_TOOL_CHOICE (optional): Tool choice strategy (auto, any, none, or specific tool name).

usage: |
  1. Set your Anthropic API key in valves.
  2. Configure other valves as needed. Key interactions:
     - Max Output Tokens: The function determines the model's maximum capability (4k/8k/64k/128k) based on model, thinking mode, and 'Enable Extended Output' valve. The value in 'Default Max Output Tokens' (or the request body) acts as a cap *only if* it's lower than this determined maximum (and not 0). Set 'Default Max Output Tokens' to 0 to always use the model's determined maximum.
     - Enable Extended Output: Must be 'Enabled' to access the 128K limit (on supported models like 3.7 Sonnet). Cannot be used with Thinking.
     - Thinking Enabled: Must be 'Enabled' for thinking mode. On 3.7 Sonnet, this defaults to a 64K max output limit *unless* 'Enable Extended Output' is also true (which takes precedence for 128K but disables thinking).
     - Thinking Budget: Automatically capped to be less than the final `max_tokens` value sent to the API.

  Notes:
  - Max output limits dynamically determined: 4096 (Opus 3, Haiku 3), 8192 (Sonnet 3.5, Haiku 3.5, Sonnet 3.7 Normal), 64000 (Sonnet 3.7 Thinking), 128000 (Sonnet 3.7 w/ Beta Enabled).
  - Thinking mode forces temperature to 1.0.
  - PDF support requires specific models (e.g., 3.5/3.7 Sonnet) and beta flag activation.
  - Context window is 200k tokens for all listed models.
  - Tool use supports Anthropic & OpenAI formats.
"""

import os
import requests
import json
import time
import logging
from typing import List, Union, Generator, Iterator, Dict, Optional, AsyncIterator, Any
from pydantic import BaseModel, Field
from open_webui.utils.misc import pop_system_message
import aiohttp
from fastapi import Request


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


class Pipe:
    API_VERSION = os.getenv("ANTHROPIC_API_VERSION", "2023-06-01")
    MODEL_URL = "https://api.anthropic.com/v1/messages"
    SUPPORTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/gif", "image/webp"]

    SUPPORTED_PDF_MODELS = [
        "claude-3-5-sonnet-20240620",
        "claude-3-7-sonnet-20240620",
        "claude-3-7-sonnet-20250219",
        "claude-3-5-sonnet-latest",
        "claude-3-7-sonnet-latest",
    ]
    SUPPORTED_THINKING_MODELS = [
        "claude-3-7-sonnet-20240620",
        "claude-3-7-sonnet-20250219",
        "claude-3-7-sonnet-latest",
    ]
    EXTENDED_OUTPUT_MODEL_FAMILIES = [
        "claude-3-7-sonnet",
    ]

    MAX_IMAGE_SIZE = 5 * 1024 * 1024
    MAX_PDF_SIZE = 32 * 1024 * 1024
    TOTAL_MAX_IMAGE_SIZE = 100 * 1024 * 1024

    PDF_BETA_HEADER = "pdfs-2024-09-25"
    PROMPT_CACHING_BETA_HEADER = "prompt-caching-2024-07-31"
    EXTENDED_OUTPUT_BETA_HEADER = "output-128k-2025-02-19"

    class Valves(BaseModel):
        ANTHROPIC_API_KEY: str = Field(
            default=os.getenv("ANTHROPIC_API_KEY", ""),
            description="Your Anthropic API key (required)",
        )
        DEFAULT_MODEL: str = Field(
            default=os.getenv("ANTHROPIC_DEFAULT_MODEL", "claude-3-5-sonnet-latest"),
            description="Default model if not specified in request",
        )
        DEFAULT_MAX_OUTPUT_TOKENS: int = Field(
            default=int(os.getenv("ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS", "0")),
            description="Default max output tokens (0=model max, otherwise caps the output)",
            ge=0,
        )
        DEFAULT_TEMPERATURE: float = Field(
            default=float(os.getenv("ANTHROPIC_DEFAULT_TEMPERATURE", "0.7")),
            description="Default temperature (0.0-1.0)",
            ge=0.0,
            le=1.0,
        )
        THINKING_ENABLED: bool = Field(
            default=os.getenv("ANTHROPIC_THINKING_ENABLED", "false").lower() == "true",
            description="Enable thinking by default for supported models",
        )
        THINKING_BUDGET: int = Field(
            default=int(os.getenv("ANTHROPIC_THINKING_BUDGET", "4096")),
            description="Default thinking budget tokens (min 1024)",
            ge=1024,
        )
        DEFAULT_STOP_SEQUENCES: str = Field(
            default=os.getenv("ANTHROPIC_DEFAULT_STOP_SEQUENCES", ""),
            description="Default stop sequences (comma-separated)",
        )
        ENABLE_PROMPT_CACHING: bool = Field(
            default=os.getenv("ANTHROPIC_ENABLE_PROMPT_CACHING", "true").lower()
            == "true",
            description="Enable prompt caching beta feature",
        )
        ENABLE_EXTENDED_OUTPUT: bool = Field(
            default=os.getenv("ANTHROPIC_ENABLE_EXTENDED_OUTPUT", "false").lower()
            == "true",
            description="Enable extended output (128K tokens) beta for supported models",
        )
        RESPONSE_FORMAT_JSON: bool = Field(
            default=os.getenv("ANTHROPIC_RESPONSE_FORMAT_JSON", "false").lower()
            == "true",
            description="Request responses strictly in JSON format",
        )
        CACHE_CONTROL: str = Field(
            default=os.getenv("ANTHROPIC_CACHE_CONTROL", "standard"),
            description="Cache control strategy (standard, aggressive, minimal)",
        )
        ENABLE_TOOL_USE: bool = Field(
            default=os.getenv("ANTHROPIC_ENABLE_TOOL_USE", "true").lower() == "true",
            description="Enable function calling/tool use capabilities",
        )
        TOOL_CHOICE: str = Field(
            default=os.getenv("ANTHROPIC_TOOL_CHOICE", "auto"),
            description="Tool choice strategy (auto, any, none, or specific tool name)",
        )
        ANTHROPIC_TIMEOUT: int = Field(
            default=int(os.getenv("ANTHROPIC_TIMEOUT", "120")),
            description="Request timeout in seconds for Anthropic API calls.",
            ge=10,
        )

    __state__ = {"enabled": True}

    def __init__(self):
        self.type = "manifold"
        self.id = "anthropic_v3"
        self.valves = self.Valves()
        self.retry_count = int(os.getenv("ANTHROPIC_RETRY_COUNT", "3"))

        try:
            from open_webui.models.functions import Functions

            function_data = Functions.get_function_by_id(self.id)
            if function_data and hasattr(function_data, "enabled"):
                Pipe.__state__["enabled"] = function_data.enabled
                log.info(
                    f"Loaded state from registry: enabled={Pipe.__state__['enabled']}"
                )
            elif function_data:
                log.warning(
                    f"Loaded function data from registry but 'enabled' attribute not found. Defaulting to enabled=True."
                )
                Pipe.__state__["enabled"] = True
            else:
                log.info(
                    f"Function '{self.id}' not found in registry. Defaulting to enabled=True."
                )
                Pipe.__state__["enabled"] = True

        except ImportError:
            log.warning(
                "Could not import OpenWebUI function models. State persistence disabled. Defaulting to enabled=True."
            )
            Pipe.__state__ = {"enabled": True}
        except Exception as e:
            log.warning(
                f"Could not load state from registry: {str(e)}. Defaulting to enabled=True."
            )
            Pipe.__state__ = {"enabled": True}

        self.default_stop_sequences = []
        if self.valves.DEFAULT_STOP_SEQUENCES:
            self.default_stop_sequences = [
                s.strip()
                for s in self.valves.DEFAULT_STOP_SEQUENCES.split(",")
                if s.strip()
            ]
            log.info(
                f"Initialized with default stop sequences: {self.default_stop_sequences}"
            )

    def enable(self):
        """Enable the function"""
        Pipe.__state__["enabled"] = True
        log.info("Anthropic V3 function enabled")
        self.save_state()
        return True

    def disable(self):
        """Disable the function"""
        Pipe.__state__["enabled"] = False
        log.info("Anthropic V3 function disabled")
        self.save_state()
        return True

    def is_enabled(self):
        """Check if the function is enabled"""
        return Pipe.__state__.get("enabled", True)

    def toggle(self):
        """Toggle the function's enabled state"""
        current_state = self.is_enabled()
        Pipe.__state__["enabled"] = not current_state
        log.info(f"Anthropic V3 function toggled: enabled={Pipe.__state__['enabled']}")
        self.save_state()
        return Pipe.__state__["enabled"]

    def save_state(self):
        """Save the function state to OpenWebUI's function registry"""
        try:
            from open_webui.models.functions import Functions

            Functions.update_function(id=self.id, enabled=Pipe.__state__["enabled"])
            log.info(
                f"Anthropic V3 function state saved to registry: enabled={Pipe.__state__['enabled']}"
            )
            return True
        except Exception as e:
            log.error(f"Error saving state to registry: {str(e)}")
            return False

    async def _emit_status(
        self,
        __event_emitter__,
        description: str,
        done: bool,
        metadata: Optional[dict] = None,
    ):
        """Helper to emit status events."""
        if __event_emitter__:
            event_data = {"description": description, "done": done}
            if metadata:
                event_data["metadata"] = metadata
            await __event_emitter__({"type": "status", "data": event_data})

    async def _emit_token_usage(self, __event_emitter__, usage: dict):
        """Helper to emit token usage events."""
        if __event_emitter__:
            await __event_emitter__({"type": "token_usage", "data": usage})

    async def _emit_thinking_event(
        self, __event_emitter__, event_type: str, model_name: str
    ):
        """Helper to emit thinking start/stop events."""
        if __event_emitter__:
            status_desc = (
                "Thinking..."
                if event_type == "thinking_start"
                else "Generating response..."
            )
            await self._emit_status(__event_emitter__, status_desc, False)
            await __event_emitter__({"type": event_type, "data": {"model": model_name}})

    async def _stream_response(self, url, headers, payload, __event_emitter__):
        """Handle streaming responses from the Anthropic API"""
        log.debug(f"Streaming request payload: {json.dumps(payload, indent=2)}")
        log.debug(f"Streaming request headers: {headers}")
        model_name = payload.get("model", "unknown")
        response_format_json = payload.get("response_format", {}).get("type") == "json"
        full_json_content = ""
        final_metadata = {}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.valves.ANTHROPIC_TIMEOUT,
                ) as response:

                    if response.status != 200:
                        error_text = await response.text()
                        error_msg = f"API Error: HTTP {response.status} {response.reason}: {error_text}"
                        log.error(error_msg)
                        await self._emit_status(__event_emitter__, error_msg, True)
                        yield {"content": error_msg, "format": "error"}
                        return

                    async for line in response.content:
                        if line and line.startswith(b"data: "):
                            try:
                                data = json.loads(line[6:])
                                event_type = data.get("type")

                                if event_type == "message_start":
                                    final_metadata = {
                                        "id": data.get("message", {}).get("id", ""),
                                        "model": data.get("message", {}).get(
                                            "model", ""
                                        ),
                                        "type": "message",
                                    }
                                    if __event_emitter__:
                                        await __event_emitter__(
                                            {
                                                "type": "message_start",
                                                "data": final_metadata,
                                            }
                                        )

                                elif event_type == "message_delta":
                                    if "usage" in data:
                                        pass

                                elif event_type == "message_stop":
                                    stop_info = data.get("anthropic_meta", {}).get(
                                        "usage", {}
                                    )
                                    if not stop_info and "usage" in data:
                                        stop_info = data["usage"]

                                    final_metadata["stop_reason"] = data.get(
                                        "stop_reason",
                                        response.headers.get("anthropic-stop-reason"),
                                    )
                                    final_metadata["stop_sequence"] = data.get(
                                        "stop_sequence",
                                        response.headers.get("anthropic-stop-sequence"),
                                    )
                                    final_metadata["usage"] = {
                                        "input_tokens": stop_info.get(
                                            "input_tokens", 0
                                        ),
                                        "output_tokens": stop_info.get(
                                            "output_tokens", 0
                                        ),
                                        "cache_creation_input_tokens": stop_info.get(
                                            "cache_creation_input_tokens"
                                        ),
                                        "cache_read_input_tokens": stop_info.get(
                                            "cache_read_input_tokens"
                                        ),
                                    }
                                    final_metadata["usage"] = {
                                        k: v
                                        for k, v in final_metadata["usage"].items()
                                        if v is not None
                                    }

                                    log.info(
                                        f"Stream ended. Final Metadata: {final_metadata}"
                                    )
                                    await self._emit_token_usage(
                                        __event_emitter__, final_metadata["usage"]
                                    )
                                    await self._emit_status(
                                        __event_emitter__,
                                        "Request completed",
                                        True,
                                        final_metadata,
                                    )
                                    break

                                elif event_type == "content_block_start":
                                    content_type = data.get("content_block", {}).get(
                                        "type"
                                    )
                                    if content_type == "thinking":
                                        await self._emit_thinking_event(
                                            __event_emitter__,
                                            "thinking_start",
                                            model_name,
                                        )
                                    elif content_type == "tool_use":
                                        log.info(
                                            f"Tool use block started: {data.get('content_block', {}).get('name')}"
                                        )

                                elif event_type == "content_block_delta":
                                    delta = data.get("delta", {})
                                    delta_type = delta.get("type")
                                    if delta_type == "text_delta":
                                        text_chunk = delta.get("text", "")
                                        if response_format_json:
                                            full_json_content += text_chunk
                                        else:
                                            yield text_chunk
                                    elif delta_type == "input_json_delta":
                                        pass

                                elif event_type == "content_block_stop":
                                    content_type = data.get("content_block", {}).get(
                                        "type"
                                    )
                                    if content_type == "thinking":
                                        await self._emit_thinking_event(
                                            __event_emitter__,
                                            "thinking_stop",
                                            model_name,
                                        )
                                    elif content_type == "tool_use":
                                        log.info(f"Tool use block stopped.")

                                elif event_type == "ping":
                                    pass

                                elif event_type == "error":
                                    error_data = data.get("error", {})
                                    error_msg = f"API Stream Error: {error_data.get('type')} - {error_data.get('message')}"
                                    log.error(error_msg)
                                    await self._emit_status(
                                        __event_emitter__, error_msg, True
                                    )
                                    yield {"content": error_msg, "format": "error"}
                                    return

                            except json.JSONDecodeError as e:
                                log.error(
                                    f"Error decoding streaming line: {line.decode()}, Error: {str(e)}"
                                )
                            except Exception as e:
                                log.error(
                                    f"Error processing streaming event: {str(e)}",
                                    exc_info=True,
                                )

            if response_format_json:
                log.debug(f"Attempting to parse accumulated JSON: {full_json_content}")
                try:
                    json_obj = json.loads(full_json_content)
                    yield {
                        "content": json_obj,
                        "format": "json",
                        "metadata": final_metadata,
                    }
                except json.JSONDecodeError:
                    log.warning(
                        "Response was not valid JSON despite JSON format request. Returning raw text."
                    )
                    yield {
                        "content": full_json_content,
                        "format": "text_fallback_from_json",
                        "metadata": final_metadata,
                    }

        except aiohttp.ClientError as e:
            error_msg = f"Network Error during streaming: {str(e)}"
            log.error(error_msg, exc_info=True)
            await self._emit_status(__event_emitter__, error_msg, True)
            yield {"content": error_msg, "format": "error"}
        except Exception as e:
            error_msg = f"Unexpected error during streaming: {str(e)}"
            log.error(error_msg, exc_info=True)
            await self._emit_status(__event_emitter__, error_msg, True)
            yield {"content": error_msg, "format": "error"}

    async def _non_stream_response(self, url, headers, payload, __event_emitter__):
        """Handle non-streaming responses from the Anthropic API with retries"""
        log.debug(f"Non-streaming request payload: {json.dumps(payload, indent=2)}")
        log.debug(f"Non-streaming request headers: {headers}")
        last_error = None
        timeout = self.valves.ANTHROPIC_TIMEOUT

        for attempt in range(self.retry_count):
            log.info(
                f"Attempt {attempt + 1}/{self.retry_count} for non-streaming request..."
            )
            try:
                response = requests.post(
                    url, headers=headers, json=payload, timeout=timeout
                )

                if response.status_code == 200:
                    log.info("Non-streaming request successful.")
                    result = response.json()
                    content_blocks = result.get("content", [])
                    text_content = ""
                    tool_calls = []

                    for item in content_blocks:
                        if item.get("type") == "text":
                            text_content += item.get("text", "")
                        elif item.get("type") == "tool_use":
                            tool_calls.append(item)

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
                            "cache_creation_input_tokens": usage.get(
                                "cache_creation_input_tokens"
                            ),
                            "cache_read_input_tokens": usage.get(
                                "cache_read_input_tokens"
                            ),
                        },
                    }
                    metadata["usage"] = {
                        k: v for k, v in metadata["usage"].items() if v is not None
                    }

                    log.info(f"Response metadata: {metadata}")
                    await self._emit_token_usage(__event_emitter__, metadata["usage"])
                    await self._emit_status(
                        __event_emitter__,
                        "Request completed successfully",
                        True,
                        metadata,
                    )

                    response_payload = {}
                    response_format = "text"

                    if payload.get("response_format", {}).get("type") == "json":
                        try:
                            json_obj = json.loads(text_content)
                            response_payload = {
                                "content": json_obj,
                                "format": "json",
                                "metadata": metadata,
                            }
                            if tool_calls:
                                response_payload["tool_calls"] = tool_calls
                            return response_payload
                        except json.JSONDecodeError:
                            log.warning(
                                "Response was not valid JSON despite JSON format request. Returning text fallback."
                            )
                            response_format = "text_fallback_from_json"

                    response_payload = {
                        "content": text_content,
                        "format": response_format,
                        "metadata": metadata,
                    }
                    if tool_calls:
                        response_payload["tool_calls"] = tool_calls

                    return response_payload

                elif response.status_code in [429, 500, 502, 503, 504]:
                    error_msg = f"API Error (Attempt {attempt + 1}/{self.retry_count}): HTTP {response.status_code}: {response.text}"
                    log.warning(error_msg)
                    last_error = error_msg
                    time.sleep(1.5**attempt)
                    continue
                else:
                    error_msg = f"API Error: HTTP {response.status_code} {response.reason}: {response.text}"
                    log.error(error_msg)
                    await self._emit_status(__event_emitter__, error_msg, True)
                    return {"content": error_msg, "format": "error"}

            except requests.exceptions.Timeout:
                error_msg = f"API Error (Attempt {attempt + 1}/{self.retry_count}): Request timed out after {timeout}s"
                log.warning(error_msg)
                last_error = error_msg
                time.sleep(1.5**attempt)
                continue
            except requests.exceptions.RequestException as e:
                error_msg = f"API Error (Attempt {attempt + 1}/{self.retry_count}): Network or Request Error: {str(e)}"
                log.error(error_msg)
                last_error = error_msg
                break

        final_error_msg = f"API Error: Failed after {self.retry_count} attempts. Last error: {last_error}"
        log.error(final_error_msg)
        await self._emit_status(__event_emitter__, final_error_msg, True)
        return {"content": final_error_msg, "format": "error"}

    def _process_content_item(
        self, item: Any, model_name: str
    ) -> Optional[Union[Dict, List[Dict]]]:
        """Processes a single item from a message's content list."""
        item_type = item.get("type") if isinstance(item, dict) else None

        if item_type == "text":
            return {"type": "text", "text": item.get("text", "")}

        elif (
            item_type == "image_url"
            and isinstance(item.get("image_url"), dict)
            and "url" in item["image_url"]
        ):
            url = item["image_url"]["url"]
            if url.startswith("data:image"):
                try:
                    mime_type, base64_data = url.split(",", 1)
                    media_type = mime_type.split(":")[1].split(";")[0]
                    if media_type in self.SUPPORTED_IMAGE_TYPES:
                        log.debug(f"Processing base64 image ({media_type})")
                        return {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_data,
                            },
                        }
                    else:
                        log.warning(f"Unsupported image media type: {media_type}")
                        return None
                except Exception as e:
                    log.error(f"Error processing base64 image data URI: {str(e)}")
                    return None
            else:
                log.debug(f"Processing image URL: {url}")
                return {"type": "image", "source": {"type": "url", "url": url}}

        elif item_type == "image" and isinstance(item.get("source"), dict):
            log.debug("Passing through existing Anthropic image block")
            return item

        elif item_type in ["pdf_url", "document"]:
            if model_name not in self.SUPPORTED_PDF_MODELS:
                log.warning(
                    f"PDF provided but model {model_name} does not support it. Skipping."
                )
                return None

            data_source = None
            if item_type == "pdf_url" and isinstance(item.get("pdf_url"), dict):
                data_source = item["pdf_url"]
            elif item_type == "document" and isinstance(item.get("source"), dict):
                data_source = item["source"]
            elif item_type == "document" and isinstance(item.get("document"), dict):
                data_source = item["document"]

            if not data_source:
                log.warning(f"Could not find source data for {item_type}: {item}")
                return None

            if "url" in data_source and data_source["url"].startswith(
                "data:application/pdf"
            ):
                try:
                    _, base64_data = data_source["url"].split(",", 1)
                    log.debug("Processing base64 PDF document")
                    return {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": base64_data,
                        },
                    }
                except Exception as e:
                    log.error(f"Error processing base64 PDF data URI: {str(e)}")
                    return None
            elif "url" in data_source:
                log.debug(f"Processing PDF document URL: {data_source['url']}")
                return {
                    "type": "document",
                    "source": {"type": "url", "url": data_source["url"]},
                }
            elif (
                data_source.get("type") == "base64"
                and data_source.get("media_type") == "application/pdf"
            ):
                log.debug("Passing through existing Anthropic base64 PDF block")
                return {"type": "document", "source": data_source}
            else:
                log.warning(f"Unsupported PDF source format: {data_source}")
                return None

        elif item_type == "tool_use":
            log.debug(f"Passing through tool_use block: {item.get('name')}")
            return item

        elif item_type == "tool_result":
            log.debug(
                f"Passing through tool_result block for tool_use_id: {item.get('tool_use_id')}"
            )
            return item

        elif item_type == "tool_calls" and isinstance(item.get("tool_calls"), list):
            converted_calls = []
            for call in item["tool_calls"]:
                if call.get("type") == "function":
                    try:
                        arguments = json.loads(
                            call.get("function", {}).get("arguments", "{}")
                        )
                        converted_calls.append(
                            {
                                "type": "tool_use",
                                "id": call.get(
                                    "id", f"call_{call.get('function', {}).get('name')}"
                                ),
                                "name": call.get("function", {}).get("name"),
                                "input": arguments,
                            }
                        )
                    except json.JSONDecodeError:
                        log.error(
                            f"Failed to parse arguments for OpenAI tool call: {call}"
                        )
                    except Exception as e:
                        log.error(f"Error converting OpenAI tool call: {e}")
                else:
                    log.warning(
                        f"Skipping unsupported OpenAI tool call type: {call.get('type')}"
                    )
            return converted_calls if converted_calls else None

        else:
            log.warning(
                f"Unsupported content item type or structure: {item_type} / {item}"
            )
            return None

    def _process_messages(self, messages: List[dict], model_name: str) -> List[dict]:
        """
        Processes messages from OpenWebUI format to Anthropic API format.
        Handles various content types (text, image, pdf) and roles (user, assistant, tool).
        Maps OpenAI tool/function call formats to Anthropic tool_use/tool_result.
        """
        processed_api_messages = []
        for i, msg in enumerate(messages):
            role = msg.get("role")
            content = msg.get("content")
            processed_content_blocks = []

            if isinstance(content, str):
                processed_content_blocks.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for item in content:
                    processed_item = self._process_content_item(item, model_name)
                    if isinstance(processed_item, list):
                        processed_content_blocks.extend(processed_item)
                    elif isinstance(processed_item, dict):
                        processed_content_blocks.append(processed_item)
            elif isinstance(content, dict) and content.get("type") == "tool_result":
                processed_item = self._process_content_item(content, model_name)
                if processed_item:
                    processed_content_blocks.append(processed_item)

            if role == "assistant":
                if "tool_calls" in msg and isinstance(msg["tool_calls"], list):
                    converted_calls = self._process_content_item(
                        {"type": "tool_calls", "tool_calls": msg["tool_calls"]},
                        model_name,
                    )
                    if converted_calls:
                        processed_content_blocks.extend(converted_calls)

                if "function_call" in msg and isinstance(msg["function_call"], dict):
                    try:
                        arguments = json.loads(
                            msg["function_call"].get("arguments", "{}")
                        )
                        tool_call_id = msg.get(
                            "id", f"call_{msg['function_call']['name']}"
                        )
                        processed_content_blocks.append(
                            {
                                "type": "tool_use",
                                "id": tool_call_id,
                                "name": msg["function_call"]["name"],
                                "input": arguments,
                            }
                        )
                        log.debug(
                            f"Converted legacy function_call '{msg['function_call']['name']}' to tool_use."
                        )
                    except json.JSONDecodeError:
                        log.error(
                            f"Failed to parse arguments for legacy function call: {msg['function_call']}"
                        )
                    except Exception as e:
                        log.error(f"Error converting legacy function call: {e}")

            if not processed_content_blocks and role != "system":
                log.warning(
                    f"Skipping message {i} due to empty or unprocessed content for role '{role}'. Original content: {content}"
                )
                continue

            if role == "user":
                is_tool_result = all(
                    block.get("type") == "tool_result"
                    for block in processed_content_blocks
                )
                if is_tool_result:
                    log.debug(f"Constructing user message containing tool results.")
                    processed_api_messages.append(
                        {"role": "user", "content": processed_content_blocks}
                    )
                else:
                    log.debug("Constructing standard user message.")
                    processed_api_messages.append(
                        {"role": "user", "content": processed_content_blocks}
                    )

            elif role == "assistant":
                log.debug("Constructing assistant message.")
                valid_assistant_blocks = [
                    b
                    for b in processed_content_blocks
                    if b.get("type") != "tool_result"
                ]
                if not valid_assistant_blocks:
                    log.warning(
                        f"Skipping assistant message {i} as it only contained tool_result blocks after processing."
                    )
                    continue
                processed_api_messages.append(
                    {"role": "assistant", "content": valid_assistant_blocks}
                )

            elif role == "tool":
                tool_results = []
                tool_call_id = msg.get("tool_call_id")

                if not tool_call_id:
                    for j in range(len(processed_api_messages) - 1, -1, -1):
                        prev_msg = processed_api_messages[j]
                        if prev_msg["role"] == "assistant":
                            for block in prev_msg.get("content", []):
                                if block.get("type") == "tool_use" and block.get(
                                    "name"
                                ) == msg.get("name"):
                                    tool_call_id = block.get("id")
                                    log.warning(
                                        f"Inferred tool_call_id '{tool_call_id}' for tool result based on name match."
                                    )
                                    break
                            if tool_call_id:
                                break

                if not tool_call_id:
                    log.error(
                        f"Could not find tool_call_id for tool result message: {msg}. Skipping."
                    )
                    continue

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": (
                            content if isinstance(content, str) else json.dumps(content)
                        ),
                    }
                )

                log.debug(
                    f"Constructing user message containing tool results mapped from 'tool' role."
                )
                processed_api_messages.append({"role": "user", "content": tool_results})

            elif role == "system":
                log.warning(
                    "System message found within message list, usually handled separately."
                )
            else:
                log.warning(f"Unsupported role encountered: {role}")

            if len(processed_api_messages) >= 2:
                last_api_msg = processed_api_messages[-1]
                second_last_api_msg = processed_api_messages[-2]

                if last_api_msg["role"] == second_last_api_msg["role"]:
                    log.debug(
                        f"Merging consecutive messages for role: {last_api_msg['role']}"
                    )
                    merged_content = (
                        second_last_api_msg["content"] + last_api_msg["content"]
                    )
                    second_last_api_msg["content"] = merged_content
                    processed_api_messages.pop()

        valid_sequence = True
        for i in range(len(processed_api_messages) - 1):
            current_role = processed_api_messages[i]["role"]
            next_role = processed_api_messages[i + 1]["role"]
            if current_role == next_role:
                log.error(
                    f"API Constraint Violated: Consecutive messages with role '{current_role}' at index {i} after processing. This may cause API errors."
                )
                if isinstance(
                    processed_api_messages[i]["content"], list
                ) and isinstance(processed_api_messages[i + 1]["content"], list):
                    log.warning(
                        f"Attempting final merge for consecutive '{current_role}' roles."
                    )
                    processed_api_messages[i]["content"].extend(
                        processed_api_messages[i + 1]["content"]
                    )
                    processed_api_messages.pop(i + 1)
                    valid_sequence = False
                    break
                else:
                    valid_sequence = False
                    break

        if not valid_sequence:
            log.error(
                "Message sequence does not strictly alternate user/assistant roles after processing. API call might fail."
            )

        log.debug(
            f"Final processed messages for API: {json.dumps(processed_api_messages, indent=2)}"
        )
        return processed_api_messages

    def pipes(self) -> List[dict]:
        """Defines the models and their capabilities exposed by this function."""
        models_data = []
        model_names = [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20240620",
            "claude-3-7-sonnet-20240620",
            "claude-3-7-sonnet-20250219",
            "claude-3-opus-latest",
            "claude-3-sonnet-latest",
            "claude-3-haiku-latest",
            "claude-3-5-sonnet-latest",
            "claude-3-7-sonnet-latest",
        ]

        for name in model_names:
            context_length = 200000

            model_id = name

            supports_vision = True
            supports_pdf = model_id in self.SUPPORTED_PDF_MODELS
            supports_thinking = self._supports_thinking(model_id)
            supports_extended_output = self._supports_extended_output(model_id)

            supports_tool_use = True
            supports_json = True

            display_name = f"Anthropic {name.replace('claude-', '').replace('-latest', ' Latest').replace('-', ' ').title()}"
            if "202" in display_name:
                parts = display_name.split(" ")
                for i, part in enumerate(parts):
                    if part.isdigit() and len(part) == 8:
                        parts[i] = f"({part[:4]}-{part[4:6]}-{part[6:]})"
                display_name = " ".join(parts)

            models_data.append(
                {
                    "id": f"anthropic/{model_id}",
                    "name": display_name,
                    "context_length": context_length,
                    "supports_vision": supports_vision,
                    "supports_pdf": supports_pdf,
                    "supports_thinking": supports_thinking,
                    "supports_tool_use": supports_tool_use,
                    "supports_json": supports_json,
                    "supports_extended_output": supports_extended_output,
                }
            )

        return models_data

    def _supports_extended_output(self, model_name: str) -> bool:
        """Check if a model likely supports extended output based on its family name."""
        if model_name in ["claude-3-7-sonnet-20250219"]:
            return True
        supported = any(
            model_name.startswith(family)
            for family in self.EXTENDED_OUTPUT_MODEL_FAMILIES
        )
        return supported

    def _supports_thinking(self, model_name: str) -> bool:
        """Check if a model supports extended thinking capabilities."""
        supported = model_name in self.SUPPORTED_THINKING_MODELS or any(
            model_name.startswith(family) for family in ["claude-3-7-sonnet"]
        )
        return supported

    async def pipe(
        self,
        body: Dict,
        __user__: Optional[dict] = None,
        __request__: Optional[Request] = None,
        __event_emitter__=None,
    ) -> Union[str, Generator, Iterator, AsyncIterator, Dict]:
        """Main function entry point called by OpenWebUI."""

        if not self.is_enabled():
            log.warning("Anthropic V3 function called while disabled.")
            await self._emit_status(
                __event_emitter__, "Error: Anthropic V3 function is disabled", True
            )
            return {
                "content": "Error: Anthropic V3 function is disabled",
                "format": "error",
            }

        if not self.valves.ANTHROPIC_API_KEY:
            log.error("ANTHROPIC_API_KEY is not configured.")
            await self._emit_status(
                __event_emitter__, "Error: ANTHROPIC_API_KEY is required", True
            )
            return {
                "content": "Error: ANTHROPIC_API_KEY is required",
                "format": "error",
            }

        try:
            system_message, messages_in = pop_system_message(body["messages"])
            log.info(
                f"Received {len(messages_in)} messages. System prompt: {'Yes' if system_message else 'No'}"
            )

            requested_model_id = body.get(
                "model", f"anthropic/{self.valves.DEFAULT_MODEL}"
            )
            if not requested_model_id.startswith("anthropic/"):
                log.warning(
                    f"Model ID '{requested_model_id}' missing 'anthropic/' prefix, adding it."
                )
                requested_model_id = f"anthropic/{requested_model_id}"

            model_name = requested_model_id.split("/")[-1]
            log.info(f"Processing request for model: {model_name}")

            processed_messages = self._process_messages(messages_in, model_name)
            if not processed_messages:
                log.error(
                    "Message processing resulted in an empty list. Cannot proceed."
                )
                await self._emit_status(
                    __event_emitter__,
                    "Error: No valid messages found after processing.",
                    True,
                )
                return {
                    "content": "Error: No valid messages found after processing.",
                    "format": "error",
                }

            model_supports_extended_output = self._supports_extended_output(model_name)
            model_supports_thinking = self._supports_thinking(model_name)

            use_thinking = self.valves.THINKING_ENABLED and model_supports_thinking
            use_extended_output = (
                self.valves.ENABLE_EXTENDED_OUTPUT
                and model_supports_extended_output
                and not use_thinking
            )

            absolute_max_tokens = 4096

            if use_extended_output:
                absolute_max_tokens = 128000
                log.info(f"Using 128k Extended Output beta limit for {model_name}.")
            elif model_name.startswith("claude-3-7-sonnet"):
                if use_thinking:
                    absolute_max_tokens = 64000
                    log.info(f"Using 64k limit for {model_name} in Thinking mode.")
                else:
                    absolute_max_tokens = 8192
                    log.info(f"Using 8k limit for {model_name} in Normal mode.")
            elif model_name.startswith("claude-3-5-sonnet") or model_name.startswith(
                "claude-3-5-haiku"
            ):
                absolute_max_tokens = 8192
                log.info(f"Using 8k limit for {model_name}.")
            elif model_name.startswith("claude-3-opus") or model_name.startswith(
                "claude-3-haiku"
            ):
                absolute_max_tokens = 4096
                log.info(f"Using 4k limit for {model_name}.")
            else:
                absolute_max_tokens = 4096
                log.warning(
                    f"Model {model_name} not explicitly mapped for max tokens, defaulting to 4096."
                )

            requested_max_tokens_raw = body.get(
                "max_tokens", self.valves.DEFAULT_MAX_OUTPUT_TOKENS
            )
            try:
                requested_max_tokens_val = int(requested_max_tokens_raw)
                if requested_max_tokens_val < 0:
                    raise ValueError("max_tokens cannot be negative")
            except (ValueError, TypeError):
                log.warning(
                    f"Invalid requested max_tokens '{requested_max_tokens_raw}', using valve default: {self.valves.DEFAULT_MAX_OUTPUT_TOKENS}"
                )
                requested_max_tokens_val = self.valves.DEFAULT_MAX_OUTPUT_TOKENS

            if (
                requested_max_tokens_val == 0
                or requested_max_tokens_val > absolute_max_tokens
            ):
                final_max_tokens = absolute_max_tokens
            else:
                final_max_tokens = requested_max_tokens_val

            requested_temp_raw = body.get("temperature")
            final_temperature = self.valves.DEFAULT_TEMPERATURE
            if requested_temp_raw is not None:
                try:
                    requested_temp = float(requested_temp_raw)
                    if 0.0 <= requested_temp <= 1.0:
                        final_temperature = requested_temp
                    else:
                        log.warning(
                            f"Requested temperature {requested_temp} out of range [0.0, 1.0]. Using default: {final_temperature}"
                        )
                except (ValueError, TypeError):
                    log.warning(
                        f"Invalid temperature value '{requested_temp_raw}'. Using default: {final_temperature}"
                    )

            log.info(
                f"Model Capabilities - Ext. Output: {model_supports_extended_output}, Thinking: {model_supports_thinking}"
            )
            log.info(
                f"Valve Settings - Ext. Output: {self.valves.ENABLE_EXTENDED_OUTPUT}, Thinking: {self.valves.THINKING_ENABLED}"
            )
            log.info(
                f"Effective Mode - Use Ext. Output: {use_extended_output}, Use Thinking: {use_thinking}"
            )
            log.info(
                f"Max Tokens - Requested/Valve Default: {requested_max_tokens_raw}, Absolute Limit: {absolute_max_tokens}, Final: {final_max_tokens}"
            )
            log.info(
                f"Temperature - Requested: {requested_temp_raw}, Valve Default: {self.valves.DEFAULT_TEMPERATURE}, Final (pre-thinking): {final_temperature}"
            )

            payload = {
                "model": model_name,
                "messages": processed_messages,
                "max_tokens": final_max_tokens,
                "stream": body.get("stream", True),
            }

            if system_message:
                payload["system"] = system_message

            if use_thinking:
                raw_thinking_budget = self.valves.THINKING_BUDGET
                thinking_budget = max(1024, raw_thinking_budget)

                if thinking_budget >= final_max_tokens:
                    thinking_budget = final_max_tokens - 1
                    thinking_budget = max(1024, thinking_budget)
                    log.warning(
                        f"Thinking budget ({raw_thinking_budget}) capped to {thinking_budget} due to max_tokens ({final_max_tokens})"
                    )

                if thinking_budget >= 1024:
                    payload["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": thinking_budget,
                    }
                    payload["temperature"] = 1.0
                    log.info(
                        f"Thinking ENABLED. Budget: {thinking_budget}, Temp forced to 1.0."
                    )
                    final_temperature = 1.0
                else:
                    log.error(
                        f"Cannot enable thinking: Effective budget ({thinking_budget}) < 1024 after capping by max_tokens ({final_max_tokens}). Thinking disabled."
                    )
                    use_thinking = False
            elif self.valves.THINKING_ENABLED and not model_supports_thinking:
                log.warning(
                    f"Thinking enabled in valves but model {model_name} doesn't support it. Thinking disabled."
                )

            if not use_thinking:
                payload["temperature"] = final_temperature
                log.info(f"Using final temperature: {final_temperature}")

            stop_sequences = self.default_stop_sequences
            body_stop = body.get("stop_sequences", body.get("stop"))
            if body_stop:
                if isinstance(body_stop, str):
                    stop_sequences = [
                        s.strip() for s in body_stop.split(",") if s.strip()
                    ]
                elif isinstance(body_stop, list) and all(
                    isinstance(s, str) for s in body_stop
                ):
                    stop_sequences = body_stop
                else:
                    log.warning(
                        f"Invalid format for stop sequences in body: {body_stop}. Using default."
                    )

            if stop_sequences:
                payload["stop_sequences"] = stop_sequences
                log.info(f"Using stop sequences: {stop_sequences}")

            if self.valves.RESPONSE_FORMAT_JSON:
                payload["response_format"] = {"type": "json"}
                log.info("JSON response format enabled.")

            if (
                self.valves.ENABLE_PROMPT_CACHING
                and self.valves.CACHE_CONTROL != "standard"
            ):
                cache_settings = {}
                if self.valves.CACHE_CONTROL == "aggressive":
                    cache_settings = {"type": "auto"}
                elif self.valves.CACHE_CONTROL == "minimal":
                    cache_settings = {"type": "none"}

                if cache_settings:
                    payload["cache_control"] = cache_settings
                    log.info(f"Cache control set to: {self.valves.CACHE_CONTROL}")

            if self.valves.ENABLE_TOOL_USE:
                tools = body.get("tools")
                functions = body.get("functions")

                if tools and isinstance(tools, list):
                    payload["tools"] = tools
                    log.info(
                        f"Tool use enabled with {len(tools)} tools (Anthropic format)."
                    )
                elif functions and isinstance(functions, list):
                    anthropic_tools = []
                    for func in functions:
                        if isinstance(func, dict) and "name" in func:
                            tool = {
                                "name": func["name"],
                                "description": func.get("description", ""),
                                "input_schema": func.get(
                                    "parameters", {"type": "object", "properties": {}}
                                ),
                            }
                            anthropic_tools.append(tool)
                        else:
                            log.warning(
                                f"Skipping invalid item in OpenAI functions list: {func}"
                            )
                    if anthropic_tools:
                        payload["tools"] = anthropic_tools
                        log.info(
                            f"Tool use enabled by converting {len(anthropic_tools)} OpenAI functions."
                        )
                    tools = anthropic_tools

                if "tools" in payload:
                    tool_choice_strategy = self.valves.TOOL_CHOICE
                    openai_function_call = body.get("function_call")

                    if openai_function_call:
                        if openai_function_call == "none":
                            tool_choice_strategy = "none"
                        elif openai_function_call == "auto":
                            tool_choice_strategy = "auto"
                        elif (
                            isinstance(openai_function_call, dict)
                            and "name" in openai_function_call
                        ):
                            tool_choice_strategy = {
                                "type": "tool",
                                "name": openai_function_call["name"],
                            }
                        else:
                            log.warning(
                                f"Unsupported OpenAI function_call value: {openai_function_call}. Using valve default: {tool_choice_strategy}"
                            )
                    elif self.valves.TOOL_CHOICE not in ["auto", "standard"]:
                        if self.valves.TOOL_CHOICE == "any":
                            tool_choice_strategy = {"type": "any"}
                        elif self.valves.TOOL_CHOICE == "none":
                            tool_choice_strategy = {"type": "none"}
                        elif any(
                            t.get("name") == self.valves.TOOL_CHOICE
                            for t in payload["tools"]
                        ):
                            tool_choice_strategy = {
                                "type": "tool",
                                "name": self.valves.TOOL_CHOICE,
                            }
                        else:
                            log.warning(
                                f"Tool choice valve '{self.valves.TOOL_CHOICE}' is not 'auto', 'any', 'none', or a defined tool name. Defaulting to auto."
                            )
                            tool_choice_strategy = "auto"

                    if isinstance(tool_choice_strategy, dict):
                        payload["tool_choice"] = tool_choice_strategy
                    elif tool_choice_strategy == "auto":
                        payload["tool_choice"] = {"type": "auto"}
                    elif tool_choice_strategy == "any":
                        payload["tool_choice"] = {"type": "any"}
                    elif tool_choice_strategy == "none":
                        payload["tool_choice"] = {"type": "none"}

                    if "tool_choice" in payload:
                        log.info(f"Tool choice set to: {payload['tool_choice']}")

            headers = {
                "x-api-key": self.valves.ANTHROPIC_API_KEY,
                "anthropic-version": self.API_VERSION,
                "content-type": "application/json",
                "accept": "application/json",
            }

            beta_headers_list = []
            has_pdf = any(
                item.get("type") == "document"
                for msg in processed_messages
                for item in msg.get("content", [])
            )
            if has_pdf and model_name in self.SUPPORTED_PDF_MODELS:
                beta_headers_list.append(self.PDF_BETA_HEADER)
                log.info(
                    "PDF content detected and model supports it. Adding PDF beta header."
                )

            if self.valves.ENABLE_PROMPT_CACHING:
                beta_headers_list.append(self.PROMPT_CACHING_BETA_HEADER)
                log.info(
                    "Prompt Caching enabled via valve. Adding caching beta header."
                )

            if use_extended_output:
                beta_headers_list.append(self.EXTENDED_OUTPUT_BETA_HEADER)
                log.info(
                    f"Extended Output (128k) is active. Adding extended output beta header."
                )
            elif self.valves.ENABLE_EXTENDED_OUTPUT and not use_extended_output:
                if not model_supports_extended_output:
                    log.warning(
                        f"Extended output enabled in valves but model {model_name} doesn't support it. Beta NOT added."
                    )
                elif use_thinking:
                    log.info(
                        "Extended output enabled in valves but Thinking is active. Extended output ignored. Beta NOT added."
                    )

            if beta_headers_list:
                headers["anthropic-beta"] = ",".join(beta_headers_list)
                log.info(f"Using Anthropic Beta Headers: {headers['anthropic-beta']}")

            await self._emit_status(
                __event_emitter__, "Sending request to Anthropic API...", False
            )

            if payload.get("stream", True):
                log.info("Initiating streaming request...")
                return self._stream_response(
                    self.MODEL_URL, headers, payload, __event_emitter__
                )
            else:
                log.info("Initiating non-streaming request...")
                return await self._non_stream_response(
                    self.MODEL_URL, headers, payload, __event_emitter__
                )

        except Exception as e:
            error_msg = f"Unhandled error in pipe function: {str(e)}"
            log.error(error_msg, exc_info=True)
            await self._emit_status(__event_emitter__, error_msg, True)
            return {"content": error_msg, "format": "error"}
