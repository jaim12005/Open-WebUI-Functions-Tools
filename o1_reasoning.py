"""
title: OpenAI o1 Integration for OpenWebUI
author: Balaxxe
version: 1.5
license: MIT
requirements: pydantic>=2.0.0, requests>=2.0.0, aiohttp>=3.0.0
environment_variables: 
    - OPENAI_API_KEY (required)
    - ENABLE_FUNCTION_CALLING (not used; function-calling not supported)

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
            "supports_tools": False,  # o1 models don't support function calling
            "supports_title_generation": True,
            "can_be_task_model": True,
        },
        "o1-mini": {
            "id": "o1-mini",
            "context_window": 128000,
            "max_completion_tokens": 65536,
            "supports_vision": False,
            "supports_reasoning": True,
            "supports_developer_messages": True,
            "supports_tools": False,  # o1 models don't support function calling
            "supports_title_generation": True,
            "can_be_task_model": True,
        }
    }
    REQUEST_TIMEOUT = (3.05, 300)

    class Valves(BaseModel):
        OPENAI_API_KEY: str = Field(
            default=os.getenv("OPENAI_API_KEY", ""),
            description="Your OpenAI API key"
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
            default=False,
            description="Enable markdown formatting in responses"
        )
        ENABLE_DEVELOPER_MESSAGES: bool = Field(
            default=True,
            description="Convert system messages to developer messages for o1 models"
        )

    def __init__(self):
        self.type = "manifold"
        self.id = "openai-o1"
        self.valves = self.Valves()
        self.request_id = None
        # Baked in title generation settings
        self._title_generation_enabled = True
        self._title_generation_model = "o1-mini"
        self._title_generation_max_tokens = 30
        self._title_generation_temperature = 0.7
        self._title_generation_prompt_template = (
            "Here is the query:\n"
            "{chat_text}\n\n"
            "Create a concise, 3-5 word phrase with an emoji as a title for the previous query. "
            "Suitable Emojis for the summary can be used to enhance understanding but avoid quotation marks or special formatting. "
            "RESPOND ONLY WITH THE TITLE TEXT.\n\n"
            "Examples of titles:\n"
            "📉 Stock Market Trends\n"
            "🍪 Perfect Chocolate Chip Recipe\n"
            "Evolution of Music Streaming\n"
            "Remote Work Productivity Tips\n"
            "Artificial Intelligence in Healthcare\n"
            "🎮 Video Game Development Insights"
        )

    def get_o1_models(self) -> List[dict]:
        """Get list of available o1 models with their capabilities."""
        print("Getting o1 models...")  # Debug log
        models = [
            {
                "id": f"openai/{name}",
                "name": self.MODEL_CONFIGS[name]["id"],
                "context_length": self.MODEL_CONFIGS[name]["context_window"],
                "supports_vision": self.MODEL_CONFIGS[name]["supports_vision"],
                "supports_reasoning": True,
                "supports_title_generation": True,
                "can_be_task_model": True,
                "task_model": True,  # Add this flag to indicate it can be used for tasks
            }
            for name in self.MODEL_CONFIGS.keys()
        ]
        print(f"Available models: {[m['id'] for m in models]}")  # Debug log
        return models

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
            processed_content = [{"type": "text", "text": content}] if isinstance(content, str) else []
            if isinstance(content, list):
                for item in content:
                    if item["type"] == "text":
                        processed_content.append({"type": "text", "text": item["text"]})
                    elif item["type"] == "image_url":
                        processed_content.append(self.process_image(item))
            processed_messages.append(
                {
                    "role": role,
                    "content": processed_content if len(processed_content) > 1 else processed_content[0]["text"],
                }
            )
        return processed_messages

    async def _handle_response(self, response: aiohttp.ClientResponse) -> tuple[dict, Optional[dict]]:
        if response.status_code != 200:
            error_msg = f"Error: HTTP {response.status_code}"
            try:
                error_data = (await response.json()).get("error", {})
                error_msg += f": {error_data.get('message', await response.text())}"
            except:
                error_msg += f": {await response.text()}"
            self.request_id = response.headers.get("x-request-id")
            if self.request_id:
                error_msg += f" (Request ID: {self.request_id})"
            return {"content": error_msg, "format": "text"}, None
        
        result = await response.json()
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

    async def _stream_with_ui(self, url: str, headers: dict, payload: dict, body: dict, __event_emitter__=None) -> Generator:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    self.request_id = response.headers.get("x-request-id")
                    if response.status != 200:
                        error_msg = f"Error: HTTP {response.status}: {await response.text()}"
                        if self.request_id:
                            error_msg += f" (Request ID: {self.request_id})"
                        if __event_emitter__:
                            await __event_emitter__({"type": "status", "data": {"description": error_msg, "done": True}})
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
                                    if "delta" in choice and "content" in choice["delta"]:
                                        yield choice["delta"]["content"]
                                    if choice.get("finish_reason") == "stop":
                                        if "usage" in data:
                                            usage = data["usage"]
                                            if "completion_tokens_details" in usage:
                                                details = usage["completion_tokens_details"]
                                                total_reasoning_tokens = details.get("reasoning_tokens", 0)
                                                total_completion_tokens = details.get("accepted_prediction_tokens", 0)
                                        if __event_emitter__:
                                            if total_reasoning_tokens or total_completion_tokens:
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
                await __event_emitter__({"type": "status", "data": {"description": error_msg, "done": True}})
            yield error_msg

    async def _send_request(self, url: str, headers: dict, payload: dict) -> aiohttp.ClientResponse:
        """Send a request to the API endpoint."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    self.request_id = response.headers.get("x-request-id")
                    # Create a Response-like object that matches what the code expects
                    response.status_code = response.status
                    response_text = await response.text()
                    response.text = response_text
                    try:
                        response_json = json.loads(response_text)
                        response.json = lambda: response_json
                    except json.JSONDecodeError:
                        response.json = lambda: {}
                    return response
        except Exception as e:
            raise Exception(f"Request failed: {str(e)}")

    async def generate_title(self, messages: List[dict]) -> str:
        """Generate a title for the chat conversation."""
        if not self._title_generation_enabled or not messages or len(messages) < 2:
            return "New Chat"

        try:
            # Get the last few messages for context
            recent_messages = messages[-3:]
            chat_text = []
            
            # Process each message
            for msg in recent_messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    chat_text.append(content)
                elif isinstance(content, list):
                    text_parts = [item.get("text", "") for item in content if item.get("type") == "text"]
                    if text_parts:
                        chat_text.append(" ".join(text_parts))
            
            # Join all messages into a single text and truncate if needed
            chat_text = "\n".join(chat_text)
            if len(chat_text) > 8000:  # Match OpenWebUI's middletruncate:8000
                half_length = 4000
                chat_text = chat_text[:half_length] + "\n...\n" + chat_text[-half_length:]

            prompt = self._title_generation_prompt_template.format(
                chat_text=chat_text
            )

            payload = {
                "model": self.MODEL_CONFIGS[self._title_generation_model]["id"],
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self._title_generation_max_tokens,
                "temperature": self._title_generation_temperature,
                "stream": False
            }

            headers = {
                "Authorization": f"Bearer {self.valves.OPENAI_API_KEY}",
                "Content-Type": "application/json",
            }

            response = await self._send_request(self.MODEL_URL, headers, payload)
            if response.status_code == 200:
                result = response.json()
                title = result["choices"][0]["message"]["content"].strip()
                # Clean up but preserve emojis
                return title.replace('"', '').replace("'", "").strip()
            return "New Chat"
            
        except Exception as e:
            print(f"Title generation error: {str(e)}")  # Log the error for debugging
            return "New Chat"

    async def pipe(self, body: Dict, __event_emitter__=None) -> Union[str, Generator, Iterator]:
        """Main pipeline method."""
        if not self.valves.OPENAI_API_KEY:
            error_msg = "Error: OPENAI_API_KEY is required"
            if __event_emitter__:
                await __event_emitter__({"type": "status", "data": {"description": error_msg, "done": True}})
            return {"error": error_msg}  # Match OpenWebUI's error format

        # Handle task requests
        if body.get("task"):
            print(f"Task request received: {body.get('task')}")  # Debug log
            if body["task"] == "generate_title":
                try:
                    messages = body.get("messages", [])
                    print(f"Generating title for {len(messages)} messages")  # Debug log
                    title = await self.generate_title(messages)
                    print(f"Title generation completed: {title}")  # Debug log
                    return {"title": title}
                except Exception as e:
                    print(f"Title generation task error: {str(e)}")  # Debug log
                    return {"title": "New Chat"}
            else:
                print(f"Unsupported task: {body.get('task')}")  # Debug log
                return {"error": f"Unsupported task: {body.get('task')}"}

        # Handle regular chat completion requests
        try:
            model_name = body["model"].split("/")[-1]
            if model_name not in self.MODEL_CONFIGS:
                error_msg = f"Model {model_name} not found in available models"
                if __event_emitter__:
                    await __event_emitter__({"type": "status", "data": {"description": error_msg, "done": True}})
                return {"error": error_msg}

            model_config = self.MODEL_CONFIGS[model_name]
            
            # Extract and validate messages
            messages = body.get("messages", [])
            if not messages:
                return {"error": "No messages provided"}

            developer_message, messages = pop_system_message(messages)
            
            # Handle function/tool calling gracefully
            if body.get("tools") or body.get("functions"):
                if __event_emitter__:
                    await __event_emitter__({
                        "type": "status", 
                        "data": {
                            "description": "Note: Function calling is not supported for o1 models",
                            "done": False
                        }
                    })

            # Calculate token allocations
            input_tokens = self._estimate_tokens(messages)
            available_tokens = model_config["context_window"] - input_tokens
            max_completion = min(
                available_tokens,
                self.valves.MAX_COMPLETION_TOKENS or model_config["max_completion_tokens"],
            )
            reasoning_tokens = int(max_completion * self.valves.REASONING_RATIO)
            
            # Validate reasoning tokens
            if reasoning_tokens < self.valves.REASONING_BUFFER:
                error_msg = (
                    f"Insufficient tokens for reasoning. Required: {self.valves.REASONING_BUFFER}, "
                    f"Available: {reasoning_tokens}. Please adjust REASONING_RATIO or reduce input length."
                )
                if __event_emitter__:
                    await __event_emitter__({"type": "status", "data": {"description": error_msg, "done": True}})
                return {"error": error_msg}

            # Prepare the payload
            max_completion = max_completion - max(reasoning_tokens, self.valves.REASONING_BUFFER)
            payload = {
                "model": model_config["id"],
                "messages": self._process_messages(messages),
                "max_completion_tokens": max_completion,
                "stream": body.get("stream", False),
                "response_format": body.get("response_format"),
            }

            # Handle developer messages
            if self.valves.ENABLE_MARKDOWN and developer_message:
                developer_message = f"{developer_message}\nFormatting reenabled"
            if developer_message:
                payload["messages"].insert(
                    0,
                    {
                        "role": "developer" if self.valves.ENABLE_DEVELOPER_MESSAGES and model_config["supports_developer_messages"] else "system",
                        "content": str(developer_message),
                    },
                )

            # Clean up payload
            payload = {k: v for k, v in payload.items() if v is not None}
            
            # Set request headers
            headers = {
                "Authorization": f"Bearer {self.valves.OPENAI_API_KEY}",
                "Content-Type": "application/json",
            }

            # Send thinking status
            if __event_emitter__:
                await __event_emitter__({"type": "status", "data": {"description": "Thinking...", "done": False}})

            # Handle streaming vs non-streaming
            try:
                if payload["stream"]:
                    return self._stream_with_ui(self.MODEL_URL, headers, payload, body, __event_emitter__)
                
                response = await self._send_request(self.MODEL_URL, headers, payload)
                result, reasoning_metrics = await self._handle_response(response)
                
                if "error" in result:
                    return result
                
                response_text = result["choices"][0]["message"]["content"]
                
                # Emit metrics if available
                if __event_emitter__ and reasoning_metrics:
                    await __event_emitter__({
                        "type": "metrics",
                        "data": reasoning_metrics
                    })
                    await __event_emitter__({
                        "type": "status",
                        "data": {"description": "Complete", "done": True}
                    })
                
                return response_text

            except Exception as e:
                error_msg = f"Request failed: {str(e)}"
                if self.request_id:
                    error_msg += f" (Request ID: {self.request_id})"
                if __event_emitter__:
                    await __event_emitter__({"type": "status", "data": {"description": error_msg, "done": True}})
                return {"error": error_msg}

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            if self.request_id:
                error_msg += f" (Request ID: {self.request_id})"
            if __event_emitter__:
                await __event_emitter__({"type": "status", "data": {"description": error_msg, "done": True}})
            return {"error": error_msg}
