"""
title: Anthropic API Integration for OpenWebUI
author: Balaxxe
version: 3.1
license: MIT
requirements: pydantic>=2.0.0, requests>=2.0.0, fastapi>=0.95.0
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

Configurable Valves:
    - ANTHROPIC_API_KEY: Your Anthropic API key (required)
    - DEFAULT_MODEL: Default model to use (e.g., "claude-3-7-sonnet-latest")
    - DEFAULT_MAX_TOKENS: Default token limit (e.g., 4096)
    - DEFAULT_TEMPERATURE: Default temperature between 0.0-1.0 (e.g., 0.7)
    - THINKING_ENABLED: Enable thinking by default for Claude 3.7+ models (true/false)
    - THINKING_BUDGET: Default thinking budget tokens (min 1024, e.g., 2048)
    - DEFAULT_STOP_SEQUENCES: Default stop sequences (comma-separated strings)
    - ENABLE_LARGE_CONTEXT: Enable 128k context for Claude 3.7 models (true/false)

Supports:
# Models
- All Claude 3, 3.5, and 3.7 models (Opus, Sonnet, Haiku variants)
- 128k context window for Claude 3.7 models (available to all users)

# Input Types
- Multi-modal capabilities with image processing (JPEG, PNG, GIF, WebP)
- PDF document processing with selected models
- System prompts and multi-turn conversations

# Advanced Features
- Streaming responses for real-time output
- Function calling / Tool use with appropriate beta headers
- Response format specification (e.g., JSON)
- Extended thinking capability for Claude 3.7 models
- Automatic thinking budget adjustment to prevent errors
- Prompt caching for improved performance
- Cache control for ephemeral content

Usage:
1. Set your Anthropic API key in valves configuration
2. Configure parameters as needed in OpenWebUI:
   - Default Model: Recommended "claude-3-7-sonnet-latest"
   - Max Tokens: Controls maximum response length
   - Thinking Enabled: For deep reasoning (Claude 3.7 only)
   - Thinking Budget: Token allocation for thinking (20,000+ recommended for complex tasks)
   - Enable Large Context: Set to true for 128k context with Claude 3.7

Note: When thinking is enabled, temperature will automatically be set to 1.0 as required by Anthropic's API.
This is enforced by the API and cannot be changed.

Stop sequences are only applied when explicitly requested and help control where the model stops generating.
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
from fastapi import Request


class Pipe:
    API_VERSION = os.getenv("ANTHROPIC_API_VERSION", "2023-06-01")
    MODEL_URL = "https://api.anthropic.com/v1/messages"
    SUPPORTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/gif", "image/webp"]
    SUPPORTED_PDF_MODELS = ["claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20240620", "claude-3-5-opus-20240229", "claude-3-5-opus-20240307", "claude-3-7-sonnet-20240620"]
    MAX_IMAGE_SIZE = 5 * 1024 * 1024
    MAX_PDF_SIZE = 32 * 1024 * 1024
    TOTAL_MAX_IMAGE_SIZE = 100 * 1024 * 1024
    PDF_BETA_HEADER = "pdfs-2024-09-25"
    FUNCTION_BETA_HEADER = "function-calling-2023-09-25"
    BETA_HEADER = "prompt-caching-2024-07-31"
    RETRY_COUNT = int(os.getenv("ANTHROPIC_RETRY_COUNT", "3"))
    TIMEOUT = int(os.getenv("ANTHROPIC_TIMEOUT", "60"))
    REQUEST_TIMEOUT = (3.05, TIMEOUT)
    MODEL_MAX_TOKENS = {
        "claude-3-opus-20240229": 200000,
        "claude-3-sonnet-20240229": 200000,
        "claude-3-haiku-20240307": 200000,
        "claude-3-5-sonnet-20240620": 200000,
        "claude-3-5-sonnet-20241022": 200000,
        "claude-3-5-haiku-20241022": 200000,
        "claude-3-5-opus-20240229": 200000,
        "claude-3-5-opus-20240307": 200000,
        "claude-3-7-sonnet-20240620": 200000,
        "claude-3-opus-latest": 200000,
        "claude-3-5-sonnet-latest": 200000,
        "claude-3-5-haiku-latest": 200000,
        "claude-3-5-opus-latest": 200000,
        "claude-3-7-sonnet-latest": 200000,
    }

    class Valves(BaseModel):
        ANTHROPIC_API_KEY: str = Field(
            default=os.getenv("ANTHROPIC_API_KEY", ""),
            description="Your Anthropic API key",
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
            default=os.getenv("ANTHROPIC_DEFAULT_STOP_SEQUENCES", "\n\nHuman:,<|end-of-output|>"),
            description="Default stop sequences (comma-separated)",
        )
        ENABLE_LARGE_CONTEXT: bool = Field(
            default=os.getenv("ANTHROPIC_ENABLE_LARGE_CONTEXT", "false").lower() == "true",
            description="Enable 128k context for Claude 3.7 models",
        )

    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.type = "manifold"
        self.id = "anthropic"
        self.valves = self.Valves()
        self.request_id = None
        
        self.default_stop_sequences = []
        if self.valves.DEFAULT_STOP_SEQUENCES:
            self.default_stop_sequences = [
                s.strip() for s in self.valves.DEFAULT_STOP_SEQUENCES.split(",") if s.strip()
            ]
            logging.info(f"Initialized with default stop sequences: {self.default_stop_sequences}")
        
        if self.valves.ENABLE_LARGE_CONTEXT:
            logging.info("Large context (128k) is enabled for Claude 3.7 models")
                
        self.LARGE_CONTEXT_BETA_HEADER = None

    def get_anthropic_models(self) -> List[dict]:
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
            "claude-3-opus-latest",
            "claude-3-5-sonnet-latest",
            "claude-3-5-haiku-latest", 
            "claude-3-5-opus-latest",
            "claude-3-7-sonnet-latest",
        ]:
            context_length = 200000
            if name.startswith("claude-3-7") and self.valves.ENABLE_LARGE_CONTEXT:
                context_length = 128000
                
            models.append({
                "id": f"anthropic/{name}",
                "name": name,
                "context_length": context_length,
                "supports_vision": name != "claude-3-5-haiku-20241022",
            })
        return models

    def pipes(self) -> List[dict]:
        return self.get_anthropic_models()

    def process_content(self, content: Union[str, List[dict]]) -> List[dict]:
        """Process content from various formats into Anthropic-compatible format"""
        try:
            if isinstance(content, str):
                return [{"type": "text", "text": content}]

            processed_content = []
            for item in content:
                try:
                    if item["type"] == "text":
                        processed_content.append({"type": "text", "text": item["text"]})
                    elif item["type"] == "image_url":
                        processed_content.append(self.process_image(item))
                    elif item["type"] == "pdf_url":
                        model_name = item.get("model", "").split("/")[-1]
                        if model_name not in self.SUPPORTED_PDF_MODELS:
                            logging.warning(
                                f"PDF support is only available for models: {', '.join(self.SUPPORTED_PDF_MODELS)}"
                            )
                            continue
                        processed_content.append(self.process_pdf(item))
                except Exception as e:
                    logging.error(f"Error processing content item: {str(e)}")
                    continue
            return processed_content
        except Exception as e:
            logging.error(f"Error in process_content: {str(e)}")
            return [{"type": "text", "text": "Error processing content. Please try again."}]

    def process_image(self, image_data):
        """Process image data into Anthropic-compatible format"""
        try:
            if not isinstance(image_data, dict):
                logging.error(f"Invalid image data type: {type(image_data)}")
                return {"type": "text", "text": "[Error: Invalid image data]"}
                
            if "image_url" not in image_data:
                logging.error("Missing image_url in image data")
                return {"type": "text", "text": "[Error: Missing image URL]"}
                
            if not isinstance(image_data["image_url"], dict) or "url" not in image_data["image_url"]:
                logging.error("Invalid image_url format")
                return {"type": "text", "text": "[Error: Invalid image URL format]"}
                
            url = image_data["image_url"]["url"]
            if not url:
                logging.error("Empty image URL")
                return {"type": "text", "text": "[Error: Empty image URL]"}
                
            if isinstance(url, str) and url.startswith("data:image"):
                try:
                    mime_type, base64_data = url.split(",", 1)
                    media_type = mime_type.split(":")[1].split(";")[0]

                    if media_type not in self.SUPPORTED_IMAGE_TYPES:
                        logging.warning(f"Unsupported media type: {media_type}")
                        return {"type": "text", "text": f"[Error: Unsupported image type: {media_type}]"}

                    return {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64_data,
                        },
                    }
                except Exception as e:
                    logging.error(f"Error processing base64 image: {str(e)}")
                    return {"type": "text", "text": "[Error: Invalid image data format]"}
            else:
                if not isinstance(url, str):
                    logging.error(f"Invalid URL type: {type(url)}")
                    return {"type": "text", "text": "[Error: Invalid image URL type]"}
                    
                return {
                    "type": "image",
                    "source": {"type": "url", "url": url},
                }
        except Exception as e:
            logging.error(f"Unexpected error in process_image: {str(e)}")
            return {"type": "text", "text": "[Error processing image]"}

    def process_pdf(self, pdf_data):
        """Process PDF data into Anthropic-compatible format"""
        try:
            if not isinstance(pdf_data, dict):
                logging.error(f"Invalid PDF data type: {type(pdf_data)}")
                return {"type": "text", "text": "[Error: Invalid PDF data]"}
                
            if "pdf_url" not in pdf_data:
                logging.error("Missing pdf_url in PDF data")
                return {"type": "text", "text": "[Error: Missing PDF URL]"}
                
            if not isinstance(pdf_data["pdf_url"], dict) or "url" not in pdf_data["pdf_url"]:
                logging.error("Invalid pdf_url format")
                return {"type": "text", "text": "[Error: Invalid PDF URL format]"}
                
            url = pdf_data["pdf_url"]["url"]
            if not url:
                logging.error("Empty PDF URL")
                return {"type": "text", "text": "[Error: Empty PDF URL]"}
                
            if isinstance(url, str) and url.startswith("data:application/pdf"):
                try:
                    mime_type, base64_data = url.split(",", 1)

                    document = {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": base64_data,
                        },
                    }
                    
                    if pdf_data.get("cache_control") and isinstance(pdf_data["cache_control"], dict):
                        document["cache_control"] = pdf_data["cache_control"]

                    return document
                except Exception as e:
                    logging.error(f"Error processing base64 PDF: {str(e)}")
                    return {"type": "text", "text": "[Error: Invalid PDF data format]"}
            else:
                if not isinstance(url, str):
                    logging.error(f"Invalid PDF URL type: {type(url)}")
                    return {"type": "text", "text": "[Error: Invalid PDF URL type]"}
                    
                document = {
                    "type": "document",
                    "source": {"type": "url", "url": url},
                }

                if pdf_data.get("cache_control") and isinstance(pdf_data["cache_control"], dict):
                    document["cache_control"] = pdf_data["cache_control"]

                return document
        except Exception as e:
            logging.error(f"Unexpected error in process_pdf: {str(e)}")
            return {"type": "text", "text": "[Error processing PDF]"}

    async def pipe(
        self, body: Dict, __user__: dict = None, __request__: Request = None, __event_emitter__=None
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

            if not body.get("model"):
                body["model"] = f"anthropic/{self.valves.DEFAULT_MODEL}"
                logging.info(f"Using default model: {self.valves.DEFAULT_MODEL}")
            
            model_name = body["model"].split("/")[-1]
            max_tokens_limit = self.MODEL_MAX_TOKENS.get(model_name, 4096)

            temperature = self.valves.DEFAULT_TEMPERATURE
            if "temperature" in body:
                try:
                    temp_value = body.get("temperature")
                    if temp_value is not None:
                        if isinstance(temp_value, (int, float, str)):
                            temperature = float(temp_value)
                        else:
                            logging.warning(f"Invalid temperature value: {temp_value}, using default: {temperature}")
                except (ValueError, TypeError) as e:
                    logging.warning(f"Error parsing temperature: {e}, using default: {temperature}")
            
            payload = {
                "model": model_name,
                "messages": self._process_messages(messages),
                "max_tokens": min(
                    body.get("max_tokens", self.valves.DEFAULT_MAX_TOKENS), max_tokens_limit
                ),
                "temperature": temperature,
                "stream": body.get("stream", True),
            }
            
            if body.get("metadata") and isinstance(body.get("metadata"), dict):
                payload["metadata"] = body.get("metadata")
            
            if "stop" in body and body.get("stop") is not None and isinstance(body.get("stop"), list):
                valid_stops = []
                for s in body.get("stop"):
                    if s is not None:
                        try:
                            valid_stops.append(str(s))
                        except:
                            continue
                if valid_stops:
                    payload["stop"] = valid_stops
                    logging.info(f"Using custom stop sequences: {valid_stops}")
            
            if "top_k" in body and body.get("top_k") is not None:
                try:
                    top_k = int(body.get("top_k"))
                    if top_k > 0:
                        payload["top_k"] = top_k
                except (ValueError, TypeError):
                    logging.warning(f"Invalid top_k value: {body.get('top_k')}, omitting parameter")
            
            if "top_p" in body and body.get("top_p") is not None:
                try:
                    top_p = float(body.get("top_p"))
                    if 0 < top_p < 1:
                        payload["top_p"] = top_p
                except (ValueError, TypeError):
                    logging.warning(f"Invalid top_p value: {body.get('top_p')}, omitting parameter")

            if system_message:
                payload["system"] = str(system_message)

            if model_name.startswith("claude-3-7"):
                if body.get("thinking"):
                    thinking_budget = body.get("thinking", {}).get("budget_tokens", 0)
                    if thinking_budget > 0:
                        min_required_max_tokens = thinking_budget + 100
                        if payload["max_tokens"] <= thinking_budget:
                            logging.warning(f"Increasing max_tokens from {payload['max_tokens']} to {min_required_max_tokens} to accommodate thinking budget of {thinking_budget}")
                            payload["max_tokens"] = min_required_max_tokens
                    
                    is_valid, error_msg = self._validate_thinking_budget(
                        body.get("thinking"), 
                        payload["max_tokens"]
                    )
                    if is_valid:
                        payload["thinking"] = body.get("thinking")
                        payload["temperature"] = 1.0
                        logging.info("Setting temperature to 1.0 as required when thinking is enabled")
                    else:
                        logging.warning(f"Invalid thinking configuration: {error_msg}")
                elif self.valves.THINKING_ENABLED:
                    thinking_budget = min(self.valves.THINKING_BUDGET, max_tokens_limit // 2)
                    min_required_max_tokens = thinking_budget + 100
                    if payload["max_tokens"] <= thinking_budget:
                        logging.warning(f"Increasing max_tokens from {payload['max_tokens']} to {min_required_max_tokens} to accommodate default thinking budget of {thinking_budget}")
                        payload["max_tokens"] = min_required_max_tokens
                        
                    payload["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": thinking_budget
                    }
                    payload["temperature"] = 1.0
                    logging.info(f"Using default thinking configuration with budget: {payload['thinking']['budget_tokens']} and setting temperature to 1.0 as required")
            elif body.get("thinking") and not model_name.startswith("claude-3-7"):
                logging.warning(f"Thinking capability is only available for Claude 3.7+ models. Ignoring thinking parameter for {model_name}.")

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
            if any(isinstance(m.get("content"), list) for m in body.get("messages", [])):
                if any(any(content.get("type") == "document" for content in m.get("content", [])) for m in payload["messages"]):
                    beta_headers.append(self.PDF_BETA_HEADER)
                
            if "tools" in body:
                beta_headers.append(self.FUNCTION_BETA_HEADER)
                
            if "cache_control" in body:
                beta_headers.append(self.BETA_HEADER)
                
            if model_name.startswith("claude-3-7") and self.valves.ENABLE_LARGE_CONTEXT:
                logging.info("Using 128k context for Claude 3.7 models")
            
            if beta_headers:
                headers["anthropic-beta"] = ",".join(beta_headers)
                
            debug_payload = payload.copy()
            if "messages" in debug_payload:
                debug_payload["messages"] = f"[{len(debug_payload['messages'])} messages]"
            logging.info(f"Sending request to Anthropic API with payload: {json.dumps(debug_payload)}")
            logging.info(f"Using API version: {self.API_VERSION}")
            logging.info(f"Using beta headers: {beta_headers if beta_headers else 'None'}")

            try:
                if payload["stream"]:
                    return self._stream_with_ui(
                        self.MODEL_URL, headers, payload, body, __event_emitter__
                    )

                response = await self._send_request(self.MODEL_URL, headers, payload)
                if response.status_code != 200:
                    error_text = self._format_error_response(response)
                    if __event_emitter__:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {"description": error_text, "done": True},
                            }
                        )
                    return {"content": error_text, "format": "text"}

                result, _ = self._handle_response(response)
                response_text = result["content"][0]["text"]
                
                thinking_content = ""
                for block in result.get("content", []):
                    if block.get("type") == "thinking" and block.get("text"):
                        thinking_content = block.get("text")
                        break
                
                if thinking_content and __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "thinking",
                            "data": {
                                "thinking": thinking_content
                            },
                        }
                    )

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
    
    def _format_error_response(self, response):
        """Format error response for better readability"""
        try:
            error_json = response.json()
            if "error" in error_json:
                error_data = error_json["error"]
                error_type = error_data.get("type", "unknown_error")
                error_message = error_data.get("message", "Unknown error")
                
                if error_type == "invalid_request_error":
                    return f"Invalid request: {error_message}. Please check your parameters."
                elif error_type == "authentication_error":
                    return "Authentication failed. Please check your API key."
                elif error_type == "rate_limit_error":
                    return f"Rate limit exceeded: {error_message}. Please try again later."
                else:
                    return f"{error_type}: {error_message}"
            return f"Error: HTTP {response.status_code}: {response.text}"
        except ValueError:
            return f"Error: HTTP {response.status_code}: {response.text}"

    async def _stream_with_ui(
        self, url: str, headers: dict, payload: dict, body: dict, __event_emitter__=None
    ) -> Generator:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    self.request_id = response.headers.get("x-request-id")
                    if response.status != 200:
                        response_text = await response.text()
                        error_msg = f"Error: HTTP {response.status}"
                        try:
                            error_json = json.loads(response_text)
                            if "error" in error_json:
                                error_data = error_json["error"]
                                error_msg += f": {error_data.get('type', 'unknown')}: {error_data.get('message', 'Unknown error')}"
                            else:
                                error_msg += f": {response_text}"
                        except:
                            error_msg += f": {response_text}"
                            
                        if self.request_id:
                            error_msg += f" (Request ID: {self.request_id})"
                        
                        logging.error(f"Streaming error: {error_msg}")
                        
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
                    
                    thinking_content = ""

                    async for line in response.content:
                        if line and line.startswith(b"data: "):
                            try:
                                data = json.loads(line[6:])
                                logging.debug(f"Streaming data: {json.dumps(data)}")
                                
                                if (
                                    data["type"] == "content_block_delta"
                                    and "text" in data["delta"]
                                ):
                                    yield data["delta"]["text"]
                                elif data["type"] == "content_block_start" and data.get("content_block", {}).get("type") == "tool_use":
                                    if __event_emitter__:
                                        await __event_emitter__(
                                            {
                                                "type": "tool_calls",
                                                "data": data["content_block"],
                                            }
                                        )
                                elif data["type"] == "content_block_start" and data.get("content_block", {}).get("type") == "thinking":
                                    content_block = data.get("content_block", {})
                                    if isinstance(content_block, dict) and "text" in content_block:
                                        thinking_content = content_block["text"]
                                
                                elif data["type"] == "content_block_delta":
                                    delta = data.get("delta", {})
                                    
                                    if isinstance(delta, dict) and "thinking" in delta:
                                        thinking_delta = delta["thinking"]
                                        if isinstance(thinking_delta, dict) and "text" in thinking_delta:
                                            thinking_content += thinking_delta["text"]
                                
                                elif data["type"] == "message_stop":
                                    if thinking_content and __event_emitter__:
                                        await __event_emitter__(
                                            {
                                                "type": "thinking",
                                                "data": {
                                                    "thinking": thinking_content
                                                },
                                            }
                                        )
                                    
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
                                logging.error(f"Error processing streaming data: {str(e)}")
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
        base_delay = 1
        max_retries = self.RETRY_COUNT

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
                        f"Rate limit hit. Retrying in {retry_after} seconds. Retry count: {retry_count + 1}/{max_retries}"
                    )
                    time.sleep(retry_after)
                    retry_count += 1
                    continue
                
                if response.status_code >= 500:
                    delay = base_delay * (2**retry_count)
                    logging.warning(
                        f"Server error {response.status_code}. Retrying in {delay} seconds. Retry count: {retry_count + 1}/{max_retries}"
                    )
                    time.sleep(delay)
                    retry_count += 1
                    continue
                
                return response
            except requests.exceptions.Timeout:
                delay = base_delay * (2**retry_count)
                logging.warning(
                    f"Request timed out. Retrying in {delay} seconds. Retry count: {retry_count + 1}/{max_retries}"
                )
                time.sleep(delay)
                retry_count += 1
            except requests.exceptions.ConnectionError:
                delay = base_delay * (2**retry_count)
                logging.warning(
                    f"Connection error. Retrying in {delay} seconds. Retry count: {retry_count + 1}/{max_retries}"
                )
                time.sleep(delay)
                retry_count += 1
            except requests.exceptions.RequestException as e:
                logging.error(f"Request failed with error: {str(e)}")
                raise
        
        logging.error(f"Max retries ({max_retries}) exceeded.")
        response = requests.Response()
        response.status_code = 429
        response._content = json.dumps({"error": {"type": "rate_limit_error", "message": "Max retries exceeded"}}).encode()
        return response

    def _handle_response(self, response):
        if response.status_code != 200:
            error_msg = f"Error: HTTP {response.status_code}"
            try:
                error_data = response.json().get("error", {})
                error_msg += f": {error_data.get('message', response.text)}"
                if error_data.get("type") == "authentication_error":
                    error_msg = "Authentication error: Please check your Anthropic API key"
                elif error_data.get("type") == "invalid_request_error":
                    error_msg = f"Invalid request: {error_data.get('message', 'Please check your request parameters')}"
                elif error_data.get("type") == "rate_limit_error":
                    error_msg = "Rate limit exceeded: Please try again later"
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

    def _validate_thinking_budget(self, thinking, max_tokens):
        """Validate that the thinking budget is appropriate"""
        if not isinstance(thinking, dict):
            return False, "Thinking must be an object with 'type' and 'budget_tokens' fields"
        
        if thinking.get("type") != "enabled":
            return False, "Thinking type must be 'enabled'"
        
        budget = thinking.get("budget_tokens")
        if budget is None:
            return False, "budget_tokens is required for thinking"
            
        if not isinstance(budget, int):
            try:
                budget = int(budget)
            except (ValueError, TypeError):
                return False, "budget_tokens must be an integer"
        
        if budget < 1024:
            return False, "Minimum thinking budget is 1,024 tokens"
        
        if budget >= max_tokens:
            return False, f"Thinking budget ({budget}) must be less than max_tokens ({max_tokens})"
            
        if budget > max_tokens * 0.9:
            return False, f"Thinking budget ({budget}) is too large compared to max_tokens ({max_tokens})"
            
        return True, None
