"""
title: Imagen 4 Ultra Function for OpenWebUI
author: Balaxxe (adapted from Flux 1.1 Pro Ultra implementation)
author_url: https://github.com/jaim12005/open-webui-imagen-4-ultra
funding_url: https://github.com/open-webui
version: 1.0.0
license: MIT
requirements: pydantic>=1.8.0, aiohttp>=3.8.0
environment_variables:
    - REPLICATE_API_TOKEN (required)
    - IM4_IMAGE_SIZE (optional, default: "1024x1024", options: 512x512|768x768|1024x1024)
    - IM4_ASPECT_RATIO (optional, default: "1:1", options: 21:9|16:9|3:2|4:3|5:4|1:1|4:5|3:4|2:3|9:16|9:21)
    - IM4_OUTPUT_FORMAT (optional, default: "jpg", options: jpg|png)
    - IM4_QUALITY (optional, default: "premium", options: draft|standard|premium)
    - IM4_STYLE (optional, default: "photorealistic", options: photorealistic|illustration|abstract)
    - IM4_SAFETY_FILTER_LEVEL (optional, default: "block_only_high", options: block_low_and_above|block_medium_and_above|block_only_high|none)
    - IM4_SEED (optional)
    - IM4_NEGATIVE_PROMPT (optional)
supported providers: replicate.com

---

**IMPORTANT DISCLAIMER: OpenWebUI Background Tasks**

To avoid unexpected Replicate API usage and costs, it is **STRONGLY RECOMMENDED** to **DISABLE** the following settings in your OpenWebUI instance (usually found under Settings -> General):

* **Title Auto-Generation**
* **Chat Tags Auto-Generation**

If these settings are enabled, OpenWebUI will automatically call this function *again* in the background after your initial image generation request completes (even if it failed). It uses the function to generate titles or tags based on the chat content, leading to multiple Replicate predictions for a single user interaction. Disabling these settings prevents these extra background calls.

---

**Functionality Notes:**

* Uses Replicate's asynchronous API via polling with tuned timeouts based on observed performance.
* Calls the model-specific Replicate endpoint (`google/imagen-4-ultra`).
* Handles API errors, timeouts, and NSFW or other error responses from Replicate.
* Streams status and final results to OpenWebUI via SSE chunks.
* Requires Pydantic V1 (`>=1.8.0`).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import aiohttp
from pydantic import BaseModel, Field, validator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("imagen-4-ultra")


class AspectRatioEnum(str, Enum):
    ratio_21_9 = "21:9"
    ratio_16_9 = "16:9"
    ratio_3_2 = "3:2"
    ratio_4_3 = "4:3"
    ratio_5_4 = "5:4"
    ratio_1_1 = "1:1"
    ratio_4_5 = "4:5"
    ratio_3_4 = "3:4"
    ratio_2_3 = "2:3"
    ratio_9_16 = "9:16"
    ratio_9_21 = "9:21"


class ImageSizeEnum(str, Enum):
    size_512 = "512x512"
    size_768 = "768x768"
    size_1024 = "1024x1024"


class OutputFormatEnum(str, Enum):
    jpg = "jpg"
    png = "png"


class QualityEnum(str, Enum):
    draft = "draft"
    standard = "standard"
    premium = "premium"


class StyleEnum(str, Enum):
    photorealistic = "photorealistic"
    illustration = "illustration"
    abstract = "abstract"


class SafetyFilterEnum(str, Enum):
    block_low_and_above = "block_low_and_above"
    block_medium_and_above = "block_medium_and_above"
    block_only_high = "block_only_high"
    none_ = "none"


class Pipe:
    """A pipe that generates images using Google Imagen-4 Ultra on Replicate."""

    class Valves(BaseModel):
        """User-configurable parameters for Imagen-4 Ultra."""

        REPLICATE_API_TOKEN: str = Field(
            default="", description="Your Replicate API token"
        )
        IM4_IMAGE_SIZE: ImageSizeEnum = Field(
            default=ImageSizeEnum.size_1024,
            description="Output image size (widthxheight)",
            title="Image Size",
        )
        IM4_ASPECT_RATIO: AspectRatioEnum = Field(
            default=AspectRatioEnum.ratio_1_1,
            description="Output image aspect ratio",
            title="Aspect Ratio",
        )
        IM4_OUTPUT_FORMAT: OutputFormatEnum = Field(
            default=OutputFormatEnum.jpg,
            description="Output image format",
            title="Output Format",
        )
        IM4_QUALITY: QualityEnum = Field(
            default=QualityEnum.premium,
            description="Generation quality",
            title="Generation Quality",
        )
        IM4_STYLE: StyleEnum = Field(
            default=StyleEnum.photorealistic,
            description="Requested style",
            title="Style",
        )
        IM4_SEED: Optional[int] = Field(
            default=None, description="Random seed for reproducible generations"
        )
        IM4_SAFETY_FILTER_LEVEL: SafetyFilterEnum = Field(
            default=SafetyFilterEnum.block_only_high,
            description="Safety filter level (per Imagen 4 API)",
            title="Safety Filter Level",
        )
        IM4_NEGATIVE_PROMPT: Optional[str] = Field(
            default=None, description="Negative prompt for image generation"
        )

        # validators
        @validator("IM4_IMAGE_SIZE")
        def validate_image_size(cls, v: str) -> str:  # noqa: N805
            if not isinstance(v, str):
                raise ValueError("Image size must be a string like '1024x1024'.")
            try:
                width, height = map(int, v.lower().split("x"))
                if width <= 0 or height <= 0:
                    raise ValueError
            except Exception:
                raise ValueError(
                    "IM4_IMAGE_SIZE must be in format 'widthxheight', e.g. '1024x1024'"
                )
            return v

        @validator("IM4_SEED", pre=True)
        def validate_seed(cls, v):  # noqa: N805
            if v in (None, ""):
                return None
            try:
                return int(v)
            except Exception:
                logger.warning("Invalid IM4_SEED value '%s', using random seed.", v)
                return None

    def __init__(self) -> None:
        self.type = "pipe"
        self.id = "imagen-4-ultra"
        self.name = "Imagen-4 Ultra"
        self.MODEL_URL = (
            "https://api.replicate.com/v1/models/google/imagen-4-ultra/predictions"
        )
        self.valves = self._load_config_from_env()
        self.session: Optional[aiohttp.ClientSession] = None
        logger.info(
            "Initialized %s pipe with endpoint %s", self.name, self.MODEL_URL
        )

    # ---------------------------------------------------------------------
    # Config helper
    # ---------------------------------------------------------------------

    def _load_config_from_env(self) -> "Pipe.Valves":
        config = {
            "REPLICATE_API_TOKEN": os.getenv("REPLICATE_API_TOKEN", ""),
            "IM4_IMAGE_SIZE": os.getenv("IM4_IMAGE_SIZE", "1024x1024"),
            "IM4_ASPECT_RATIO": os.getenv("IM4_ASPECT_RATIO", "1:1"),
            "IM4_OUTPUT_FORMAT": os.getenv("IM4_OUTPUT_FORMAT", "jpg"),
            "IM4_QUALITY": os.getenv("IM4_QUALITY", "standard"),
            "IM4_STYLE": os.getenv("IM4_STYLE", "photorealistic"),
            "IM4_SEED": os.getenv("IM4_SEED"),
            "IM4_NEGATIVE_PROMPT": os.getenv("IM4_NEGATIVE_PROMPT"),
        }
        try:
            return self.Valves(**config)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Config validation failed: %s", exc, exc_info=True)
            logger.warning("Falling back to default config values.")
            return self.Valves()

    # ---------------------------------------------------------------------
    # HTTP session management
    # ---------------------------------------------------------------------

    @asynccontextmanager
    async def get_session(self):  # noqa: D401
        create_new = self.session is None or self.session.closed
        if create_new:
            conn = aiohttp.TCPConnector(limit=10, ttl_dns_cache=300)
            timeout = aiohttp.ClientTimeout(total=120)
            self.session = aiohttp.ClientSession(connector=conn, timeout=timeout)
        try:
            yield self.session
        finally:
            if create_new and self.session:
                await self.session.close()
                self.session = None

    # ---------------------------------------------------------------------
    # SSE helpers
    # ---------------------------------------------------------------------

    def _create_sse_chunk(
        self,
        content: Union[str, Dict],
        content_type: str = "text/plain",
        finish_reason: Optional[str] = None,
    ) -> str:
        delta_content: Dict[str, Any] = {}
        if not finish_reason:
            if isinstance(content, dict) and "error" in content:
                delta_content = {
                    "role": "assistant",
                    "content": f'<div style="color: red;">Error: {content["error"]}</div>',
                    "content_type": "text/html",
                }
            elif isinstance(content, str):
                delta_content = {
                    "role": "assistant",
                    "content": content,
                    "content_type": content_type,
                }
            else:
                delta_content = {
                    "role": "assistant",
                    "content": "",
                    "content_type": "text/plain",
                }
        chunk = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.id,
            "choices": [
                {
                    "delta": delta_content,
                    "index": 0,
                    "finish_reason": finish_reason,
                }
            ],
        }
        try:
            return f"data: {json.dumps(chunk)}\n\n"
        except TypeError as exc:  # pragma: no cover
            logger.error("Failed to serialize SSE chunk: %s", exc)
            return "data: {}\n\n"

    # ---------------------------------------------------------------------
    # Polling helper
    # ---------------------------------------------------------------------

    async def _wait_for_completion(
        self, prediction_url: str, __event_emitter__=None
    ) -> Dict:
        headers = {
            "Authorization": f"Token {self.valves.REPLICATE_API_TOKEN}",
            "Accept": "application/json",
            "Prefer": "wait=3",
        }
        max_retries = 7
        base_delay = 1
        async with self.get_session() as session:
            for attempt in range(max_retries):
                if attempt:
                    delay = min(base_delay * (1.5 ** attempt), 30)
                    if __event_emitter__:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "description": f"Checking status (attempt {attempt+1}/{max_retries})…",
                                    "done": False,
                                },
                            }
                        )
                    await asyncio.sleep(delay)
                try:
                    async with session.get(prediction_url, headers=headers, timeout=aiohttp.ClientTimeout(total=25)) as resp:
                        resp.raise_for_status()
                        payload = await resp.json()
                        status = payload.get("status")
                        if status == "succeeded":
                            if __event_emitter__:
                                await __event_emitter__(
                                    {
                                        "type": "status",
                                        "data": {
                                            "description": "Generation complete",
                                            "done": True,
                                        },
                                    }
                                )
                            return payload
                        if status in {"failed", "canceled"}:
                            err = payload.get("error", "Unknown error")
                            if __event_emitter__:
                                await __event_emitter__(
                                    {
                                        "type": "status",
                                        "data": {
                                            "description": f"Error: {err}",
                                            "done": True,
                                        },
                                    }
                                )
                            raise RuntimeError(f"Generation {status}: {err}")
                        # else processing or starting → loop again
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning("Status poll failed (attempt %s): %s", attempt + 1, exc)
                    if attempt == max_retries - 1:
                        raise
        raise TimeoutError("Generation timed out.")

    # ---------------------------------------------------------------------
    # Helper utilities
    # ---------------------------------------------------------------------

    @staticmethod
    def _extract_seed_from_logs(logs: str) -> Optional[int]:
        if not isinstance(logs, str):
            return None
        for line in logs.splitlines():
            if "Using seed:" in line:
                try:
                    return int("".join(ch for ch in line.split("Using seed:")[1] if ch.isdigit()))
                except Exception:  # noqa: B902
                    return None
        return None

    @staticmethod
    def _extract_prompt(body: Dict) -> str:
        prompt = ""
        if body.get("messages") and isinstance(body["messages"], list):
            last = body["messages"][-1]
            if isinstance(last, dict):
                prompt = str(last.get("content", "")).strip()
        if not prompt:
            prompt = str(body.get("prompt", "")).strip()
        return prompt

    # ---------------------------------------------------------------------
    # Input preparation
    # ---------------------------------------------------------------------

    def _prepare_input_params(self, prompt: str) -> Dict[str, Any]:
        try:
            width, height = map(int, self.valves.IM4_IMAGE_SIZE.value.split("x"))
            if width <= 0 or height <= 0:
                raise ValueError
        except Exception:
            logger.warning("IM4_IMAGE_SIZE invalid, defaulting to 1024x1024.")
            width, height = 1024, 1024
        params: Dict[str, Any] = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "aspect_ratio": self.valves.IM4_ASPECT_RATIO.value,
            "output_format": self.valves.IM4_OUTPUT_FORMAT.value,
            "quality": self.valves.IM4_QUALITY.value,
        }
        if self.valves.IM4_STYLE:
            params["style"] = self.valves.IM4_STYLE.value
        if self.valves.IM4_SAFETY_FILTER_LEVEL and self.valves.IM4_SAFETY_FILTER_LEVEL != SafetyFilterEnum.none_:
            params["safety_filter_level"] = self.valves.IM4_SAFETY_FILTER_LEVEL.value
        if self.valves.IM4_SEED is not None:
            params["seed"] = self.valves.IM4_SEED
        if self.valves.IM4_NEGATIVE_PROMPT:
            params["negative_prompt"] = self.valves.IM4_NEGATIVE_PROMPT
        return params

    # ---------------------------------------------------------------------
    # Generation start
    # ---------------------------------------------------------------------

    async def _start_generation(self, input_params: Dict[str, Any]) -> Dict[str, Any]:
        payload = {"input": input_params}
        headers = {
            "Authorization": f"Token {self.valves.REPLICATE_API_TOKEN}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Prefer": "respond-async",
        }
        async with self.get_session() as session:
            async with session.post(
                self.MODEL_URL,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    raise RuntimeError(f"Replicate error {resp.status}: {text[:300]}")
                return await resp.json()

    # ---------------------------------------------------------------------
    # Final results
    # ---------------------------------------------------------------------

    async def _yield_final_results(
        self,
        image_url: str,
        prompt: str,
        input_params: Dict[str, Any],
        seed: Any,
        metrics: Dict[str, Any],
    ) -> AsyncIterator[str]:
        img_md = f"![Generated Image]({image_url})"
        yield self._create_sse_chunk(img_md, content_type="text/markdown")

        predict_time = metrics.get("predict_time")
        total_time = metrics.get("total_time")
        details_md = f"""
<details>
<summary>Generation Details</summary>

* **Prompt:** {prompt}
* **Negative Prompt:** {input_params.get('negative_prompt', 'N/A')}
* **Size:** {input_params.get('width')}x{input_params.get('height')}
* **Aspect Ratio:** {input_params.get('aspect_ratio')}
* **Format:** {input_params.get('output_format')}
* **Quality:** {input_params.get('quality')}
* **Style:** {input_params.get('style', 'N/A')}
* **Safety Filter:** {input_params.get('safety_filter_level', 'N/A')}
* **Seed:** {seed if seed is not None else 'Random'}
* **Generation Time:** {predict_time or 'N/A'}
* **Total Time:** {total_time or 'N/A'}
</details>
"""
        yield self._create_sse_chunk(details_md.strip(), content_type="text/markdown")
        yield self._create_sse_chunk({}, finish_reason="stop")
        yield "data: [DONE]\n\n"

    # ---------------------------------------------------------------------
    # Public entry
    # ---------------------------------------------------------------------

    async def pipe(self, body: Dict, __event_emitter__=None) -> AsyncIterator[str]:
        start_time = time.time()
        try:
            if not self.valves.REPLICATE_API_TOKEN:
                raise ValueError("REPLICATE_API_TOKEN not configured.")
            prompt = self._extract_prompt(body)
            if not prompt:
                raise ValueError("No prompt provided.")
            input_params = self._prepare_input_params(prompt)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Input error: %s", exc)
            yield self._create_sse_chunk({"error": str(exc)}, finish_reason="stop")
            yield "data: [DONE]\n\n"
            return

        try:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Starting generation…", "done": False},
                    }
                )
            prediction = await self._start_generation(input_params)
            prediction_id = prediction.get("id", "N/A")
            logger.info("Started prediction %s", prediction_id)
            result = await self._wait_for_completion(prediction["urls"]["get"], __event_emitter__)
            output = result.get("output")
            if isinstance(output, list):
                image_url = output[0]
            else:
                image_url = output  # type: ignore
            seed = result.get("seed") or self._extract_seed_from_logs(result.get("logs", ""))
            metrics = {
                "predict_time": result.get("metrics", {}).get("predict_time"),
                "total_time": time.time() - start_time,
            }
            async for chunk in self._yield_final_results(image_url, prompt, input_params, seed, metrics):
                yield chunk
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Generation failed: %s", exc, exc_info=True)
            yield self._create_sse_chunk({"error": f"Generation failed: {exc}"}, finish_reason="stop")
            yield "data: [DONE]\n\n"
