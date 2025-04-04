"""
title: FLUX Schnell Manifold Function for Black Forest Lab Image Generation Models
author: Balaxxe, credit to mobilestack and bgeneto
author_url: https://github.com/jaim12005/open-webui-flux-1.1-pro-ultra
funding_url: https://github.com/open-webui
version: 2.0.0
license: MIT
requirements: pydantic>=1.8.0, aiohttp>=3.8.0
environment_variables:
    - REPLICATE_API_TOKEN (required)
    - FLUX_GO_FAST (optional, default: true)
    - FLUX_DISABLE_SAFETY (optional, default: false, API only)
    - FLUX_SEED (optional)
    - FLUX_ASPECT_RATIO (optional, default: "1:1")
    - FLUX_OUTPUT_FORMAT (optional, default: "webp")
    - FLUX_OUTPUT_QUALITY (optional, default: 80, range: 1-100)
    - FLUX_NUM_OUTPUTS (optional, default: 1)
    - FLUX_NEGATIVE_PROMPT (optional)
supported providers: replicate.com

---

**IMPORTANT DISCLAIMER: OpenWebUI Background Tasks**

To avoid unexpected Replicate API usage and costs, it is **STRONGLY RECOMMENDED** to **DISABLE** the following settings in your OpenWebUI instance (usually found under Settings -> General):

* **Title Auto-Generation**
* **Chat Tags Auto-Generation**

If these settings are enabled, OpenWebUI will automatically call this function *again* in the background after your initial image generation request completes (even if it failed). It uses the function to generate titles or tags based on the chat content, leading to multiple Replicate predictions for a single user interaction. Disabling these settings prevents these extra background calls.

---

**Functionality Notes:**

* Uses Replicate's asynchronous API via polling with tuned timeouts.
* Calls the model-specific Replicate endpoint (implies latest model version).
* Handles API errors, timeouts, and NSFW detection responses from Replicate.
* Displays the final image(s) using Markdown format in the chat.
* Provides generation details (prompt, seed, timing, etc.) in a collapsible section.
* Requires Pydantic V1 (`>=1.8.0`).

**Known Issues/Behavior:**

* Due to UI timing, the last "Checking status..." message might briefly appear alongside the final generated image. This is visual and does not mean polling continued after success.
* Replicate dashboard will show multiple predictions per user prompt if OpenWebUI background tasks (Title/Tag generation) are enabled (see Disclaimer above).
* Output formats other than webp might be slower or less optimized for this model.
"""

from typing import Dict, Union, Optional, List, AsyncIterator, Any
from pydantic import BaseModel, Field, validator
import os
import base64
import aiohttp
import asyncio
import json
import uuid
import time
import logging
from contextlib import asynccontextmanager
from enum import Enum

try:
    import datetime
except ImportError:
    logging.warning(
        "datetime module not available, total time calculation accuracy reduced."
    )
    datetime = None  # type: ignore


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("flux-schnell")


class AspectRatioEnum(str, Enum):
    ratio_1_1 = "1:1"
    ratio_16_9 = "16:9"
    ratio_21_9 = "21:9"
    ratio_3_2 = "3:2"
    ratio_2_3 = "2:3"
    ratio_4_5 = "4:5"
    ratio_5_4 = "5:4"
    ratio_3_4 = "3:4"
    ratio_4_3 = "4:3"
    ratio_9_16 = "9:16"
    ratio_9_21 = "9:21"


class OutputFormatEnum(str, Enum):
    webp = "webp"
    jpg = "jpg"
    png = "png"


class Pipe:
    """A pipe that generates images using Black Forest Lab's Flux Schnell model."""

    class Valves(BaseModel):
        """Configuration parameters for the Flux Schnell model."""

        REPLICATE_API_TOKEN: str = Field(
            default="", description="Your Replicate API token"
        )
        FLUX_GO_FAST: bool = Field(
            default=True, description="Enable fast mode (uses fewer inference steps)"
        )
        FLUX_DISABLE_SAFETY: bool = Field(
            default=False, description="Disable the built-in safety checker (API only)"
        )
        FLUX_SEED: Optional[int] = Field(
            default=None, description="Random seed for reproducible generations"
        )
        FLUX_ASPECT_RATIO: AspectRatioEnum = Field(
            default=AspectRatioEnum.ratio_1_1, description="Output image aspect ratio"
        )
        FLUX_OUTPUT_FORMAT: OutputFormatEnum = Field(
            default=OutputFormatEnum.webp, description="Output image format"
        )
        FLUX_OUTPUT_QUALITY: int = Field(
            default=80, ge=1, le=100, description="Output image quality (1-100)"
        )
        FLUX_NUM_OUTPUTS: int = Field(
            default=1, ge=1, description="Number of images to generate per prompt"
        )
        FLUX_NEGATIVE_PROMPT: Optional[str] = Field(
            default=None, description="Negative prompt for image generation"
        )

        @validator("FLUX_OUTPUT_QUALITY", pre=True)
        def validate_output_quality(cls, v):
            """Validate and convert output quality."""
            try:
                value = int(v)
                if 1 <= value <= 100:
                    return value
                else:
                    logger.warning(
                        f"Invalid FLUX_OUTPUT_QUALITY value: {v}. Using default: 80"
                    )
                    return 80
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid FLUX_OUTPUT_QUALITY type: {v}. Using default: 80"
                )
                return 80

        @validator("FLUX_NUM_OUTPUTS", pre=True)
        def validate_num_outputs(cls, v):
            """Validate and convert num outputs."""
            try:
                value = int(v)
                if value >= 1:
                    # Replicate might have upper limits, but we'll allow >= 1 here
                    return value
                else:
                    logger.warning(
                        f"Invalid FLUX_NUM_OUTPUTS value: {v}. Using default: 1"
                    )
                    return 1
            except (ValueError, TypeError):
                logger.warning(f"Invalid FLUX_NUM_OUTPUTS type: {v}. Using default: 1")
                return 1

        @validator("FLUX_GO_FAST", "FLUX_DISABLE_SAFETY", pre=True)
        def validate_booleans(cls, v):
            """Validate and convert boolean flags."""
            if isinstance(v, str):
                return v.lower() in ("true", "1", "t", "yes", "y")
            return bool(v)

        @validator("FLUX_SEED", pre=True)
        def validate_seed(cls, v):
            """Validate and convert seed."""
            if v is None or v == "":
                return None
            try:
                return int(v)
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid FLUX_SEED value: {v}. Using random seed (None)."
                )
                return None

    def __init__(self):
        """Initialize the Flux Schnell pipe with configuration."""
        self.type = "pipe"
        self.id = "flux_schnell"
        self.name = "Flux Schnell"
        self.MODEL_URL = "https://api.replicate.com/v1/models/black-forest-labs/flux-schnell/predictions"
        self.valves = self._load_config_from_env()
        self.session = None
        logger.info(
            f"Initialized {self.name} pipe with ID {self.id} using endpoint {self.MODEL_URL}"
        )

    def _load_config_from_env(self) -> Valves:
        """Load configuration from environment variables using Pydantic validation."""
        config_data = {
            "REPLICATE_API_TOKEN": os.getenv("REPLICATE_API_TOKEN", ""),
            "FLUX_GO_FAST": os.getenv("FLUX_GO_FAST", "True"),
            "FLUX_DISABLE_SAFETY": os.getenv("FLUX_DISABLE_SAFETY", "False"),
            "FLUX_SEED": os.getenv("FLUX_SEED"),
            "FLUX_ASPECT_RATIO": os.getenv("FLUX_ASPECT_RATIO", "1:1"),
            "FLUX_OUTPUT_FORMAT": os.getenv("FLUX_OUTPUT_FORMAT", "webp"),
            "FLUX_OUTPUT_QUALITY": os.getenv("FLUX_OUTPUT_QUALITY", "80"),
            "FLUX_NUM_OUTPUTS": os.getenv("FLUX_NUM_OUTPUTS", "1"),
            "FLUX_NEGATIVE_PROMPT": os.getenv("FLUX_NEGATIVE_PROMPT"),
        }
        try:
            return self.Valves(**config_data)
        except Exception as e:
            logger.error(
                f"Error loading or validating configuration: {str(e)}", exc_info=True
            )
            logger.warning("Falling back to default configuration due to error.")
            return self.Valves()

    @asynccontextmanager
    async def get_session(self):
        """Get or create an aiohttp ClientSession using context manager pattern."""
        session_needs_creation = self.session is None or self.session.closed
        if session_needs_creation:
            conn = aiohttp.TCPConnector(
                limit=10,
                ttl_dns_cache=300,
                enable_cleanup_closed=True,
                force_close=False,
            )
            timeout = aiohttp.ClientTimeout(total=120)
            self.session = aiohttp.ClientSession(connector=conn, timeout=timeout)
            logger.debug("Created new aiohttp ClientSession.")
        else:
            logger.debug("Reusing existing aiohttp ClientSession.")

        try:
            yield self.session
        finally:
            if session_needs_creation and self.session:
                await self.session.close()
                self.session = None
                logger.debug("Closed aiohttp ClientSession created by this context.")

    def _create_sse_chunk(
        self,
        content: Union[str, Dict],
        content_type: str = "text/plain",
        finish_reason: Optional[str] = None,
    ) -> str:
        """
        Creates a Server-Sent Events (SSE) chunk in the format expected by OpenWebUI,
        mimicking the OpenAI streaming format.
        """
        delta_content = {}
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
                logger.warning(
                    f"Unexpected content type for SSE delta: {type(content)}"
                )
                delta_content = {
                    "role": "assistant",
                    "content": "",
                    "content_type": "text/plain",
                }

        chunk_data = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.id,  # Use pipe ID
            "choices": [
                {
                    "delta": delta_content,
                    "index": 0,
                    "finish_reason": finish_reason,
                }
            ],
        }

        try:
            json_payload = json.dumps(chunk_data)
            return f"data: {json_payload}\n\n"
        except TypeError as e:
            logger.error(
                f"Failed to serialize SSE chunk data to JSON: {e} - Data: {chunk_data}"
            )
            error_payload = json.dumps(
                {
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": self.id,
                    "choices": [
                        {
                            "delta": {
                                "role": "assistant",
                                "content": "<div style='color: red;'>Error: Failed to serialize response chunk.</div>",
                                "content_type": "text/html",
                            },
                            "index": 0,
                            "finish_reason": "stop",
                        }
                    ],
                }
            )
            return f"data: {error_payload}\n\n"

    async def _wait_for_completion(
        self, prediction_url: str, __event_emitter__=None
    ) -> Dict:
        """
        Polls the Replicate prediction URL with exponential backoff until the
        generation is complete ('succeeded'), fails ('failed', 'canceled'),
        or times out. Emits status updates via __event_emitter__.
        Uses tuned parameters based on typical performance.
        """
        headers = {
            "Authorization": f"Token {self.valves.REPLICATE_API_TOKEN}",
            "Accept": "application/json",
            "Prefer": "wait=5",  # Schnell is fast, wait less per poll
        }

        max_retries = 7  # Should be enough for ~30-60s total window
        base_delay = 0.5  # Start with shorter delay for Schnell

        async with self.get_session() as session:
            for attempt in range(max_retries):
                if attempt > 0:
                    # Slightly faster backoff for Schnell
                    delay = base_delay * (1.8**attempt) + (
                        0.5 * base_delay * (asyncio.get_event_loop().time() % 1)
                    )
                    delay = min(delay, 15)  # Cap delay lower for Schnell

                    status_message = f"Checking generation status (attempt {attempt+1}/{max_retries}, waiting {delay:.1f}s)..."
                    logger.info(status_message)
                    if __event_emitter__:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {"description": status_message, "done": False},
                            }
                        )
                    await asyncio.sleep(delay)
                else:
                    logger.info(
                        f"Checking prediction status (attempt {attempt+1}/{max_retries})..."
                    )

                try:
                    # Shorter timeout aligned with lower 'Prefer: wait'
                    poll_timeout = aiohttp.ClientTimeout(
                        total=15, connect=5, sock_read=10
                    )
                    async with session.get(
                        prediction_url, headers=headers, timeout=poll_timeout
                    ) as response:
                        response.raise_for_status()
                        result = await response.json()
                        status = result.get("status")
                        prediction_id = result.get("id", "N/A")

                        logger.debug(
                            f"Polling attempt {attempt+1}: Status={status}, ID={prediction_id}"
                        )

                        if status == "succeeded":
                            logger.info(f"Generation succeeded (ID: {prediction_id})")
                            return result
                        elif status in ["failed", "canceled"]:
                            error_msg = result.get("error", "Unknown error")
                            logger.error(
                                f"Generation {status} (ID: {prediction_id}): {error_msg}"
                            )
                            raise Exception(f"Generation {status}: {error_msg}")
                        elif status == "processing":
                            logs = result.get("logs", "")
                            status_desc = "Processing..."
                            if logs and isinstance(logs, str):
                                try:
                                    last_log_line = logs.strip().split("\n")[-1]
                                    status_desc = f"Processing... ({last_log_line[:100]}{'...' if len(last_log_line) > 100 else ''})"
                                except IndexError:
                                    pass
                            logger.info(
                                f"Generation still processing (ID: {prediction_id})..."
                            )
                            if __event_emitter__:
                                await __event_emitter__(
                                    {
                                        "type": "status",
                                        "data": {
                                            "description": status_desc,
                                            "done": False,
                                        },
                                    }
                                )
                            continue
                        elif status == "starting":
                            logger.info(
                                f"Generation status is 'starting' (ID: {prediction_id}). Waiting..."
                            )
                            if __event_emitter__:
                                await __event_emitter__(
                                    {
                                        "type": "status",
                                        "data": {
                                            "description": "Generation starting...",
                                            "done": False,
                                        },
                                    }
                                )
                            continue
                        else:
                            logger.warning(
                                f"Unknown status received: '{status}' (ID: {prediction_id}). Continuing poll."
                            )
                            if __event_emitter__:
                                await __event_emitter__(
                                    {
                                        "type": "status",
                                        "data": {
                                            "description": f"Status: {status}...",
                                            "done": False,
                                        },
                                    }
                                )
                            continue

                except aiohttp.ClientResponseError as e:
                    logger.error(
                        f"API response error on status check attempt {attempt+1}: {e.status} {e.message}"
                    )
                    if e.status in [401, 403, 404]:
                        raise Exception(
                            f"Fatal client error checking status ({e.status}): {e.message}"
                        )
                    if attempt == max_retries - 1:
                        raise Exception(
                            f"API status check failed after {max_retries} attempts: {str(e)}"
                        )
                except aiohttp.ClientError as e:
                    logger.error(
                        f"API connection/client error on status check attempt {attempt+1}: {str(e)}"
                    )
                    if attempt == max_retries - 1:
                        raise Exception(
                            f"API status check failed after {max_retries} attempts: {str(e)}"
                        )
                except asyncio.TimeoutError:
                    logger.error(f"API status check timed out on attempt {attempt+1}")
                    if attempt == max_retries - 1:
                        raise Exception(
                            f"API status check timed out after {max_retries} attempts"
                        )
                except Exception as e:
                    logger.error(
                        f"Unexpected error during status check attempt {attempt+1}: {str(e)}",
                        exc_info=True,
                    )
                    if attempt == max_retries - 1:
                        raise Exception(
                            f"Polling failed after {max_retries} attempts due to unexpected error: {str(e)}"
                        )

        timeout_message = f"Generation timed out after {max_retries} polling attempts."
        logger.error(timeout_message)
        raise Exception(timeout_message)

    def _extract_seed_from_logs(self, logs: str) -> Optional[int]:
        """
        Extracts the seed value from Replicate's generation logs as a fallback method.
        Looks for a line like "Using seed: 12345".
        Returns the integer seed or None if not found or parsing fails.
        """
        if not isinstance(logs, str) or not logs:
            return None

        try:
            for line in logs.splitlines():
                if "Using seed:" in line:
                    seed_str = line.split("Using seed:")[1].split()[0]
                    seed_str_cleaned = "".join(filter(str.isdigit, seed_str))
                    if seed_str_cleaned:
                        logger.debug(f"Extracted seed '{seed_str_cleaned}' from logs.")
                        return int(seed_str_cleaned)
            return None
        except (IndexError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse seed from logs: {e}")
            return None

    def _extract_prompt(self, body: Dict) -> str:
        """
        Extracts the user's prompt from the request body.
        Prioritizes the OpenAI message format ('messages' list) but falls back
        to a direct 'prompt' key if necessary.
        Returns an empty string if no prompt is found.
        """
        prompt = ""
        if (
            "messages" in body
            and isinstance(body["messages"], list)
            and body["messages"]
        ):
            last_message = body["messages"][-1]
            if isinstance(last_message, dict) and "content" in last_message:
                prompt = str(last_message.get("content", "")).strip()

        if not prompt and "prompt" in body:
            prompt = str(body.get("prompt", "")).strip()

        if not prompt:
            logger.warning("Could not extract prompt from request body.")

        return prompt

    def _prepare_input_params(self, prompt: str) -> Dict:
        """
        Prepares the 'input' dictionary for the Replicate API request,
        mapping configured Valves to the expected API parameter names for Flux Schnell.
        """
        input_params = {
            "prompt": prompt,
            "go_fast": self.valves.FLUX_GO_FAST,
            "num_outputs": self.valves.FLUX_NUM_OUTPUTS,
            "aspect_ratio": self.valves.FLUX_ASPECT_RATIO.value,
            "output_format": self.valves.FLUX_OUTPUT_FORMAT.value,
            "output_quality": self.valves.FLUX_OUTPUT_QUALITY,
            "disable_safety_checker": self.valves.FLUX_DISABLE_SAFETY,
        }

        if self.valves.FLUX_SEED is not None:
            try:
                input_params["seed"] = int(self.valves.FLUX_SEED)
            except (ValueError, TypeError):
                logger.warning(
                    f"Could not convert FLUX_SEED '{self.valves.FLUX_SEED}' to int. Sending without seed."
                )

        if self.valves.FLUX_NEGATIVE_PROMPT:
            input_params["negative_prompt"] = self.valves.FLUX_NEGATIVE_PROMPT

        # Add other specific Schnell parameters here if needed, checking API docs/examples
        # e.g., megapixels (though it wasn't confirmed in browse)
        # if self.valves.FLUX_MEGAPIXELS:
        #    input_params["megapixels"] = self.valves.FLUX_MEGAPIXELS.value

        logger.debug(f"Prepared input parameters for Schnell: {input_params}")
        return input_params

    async def _start_generation(self, input_params: Dict) -> Dict:
        """
        Sends the initial POST request to the Replicate API (using model-specific URL)
        to start the prediction. Handles retries. Returns initial prediction object.
        """
        max_retries = 3
        base_delay = 1
        last_error = None

        payload = {"input": input_params}

        api_url = self.MODEL_URL

        for attempt in range(max_retries):
            try:
                start_timeout = aiohttp.ClientTimeout(
                    total=60, connect=20, sock_connect=20, sock_read=40
                )

                async with self.get_session() as session:
                    logger.info(
                        f"Starting generation request (attempt {attempt+1}/{max_retries}) to {api_url}"
                    )
                    logger.debug(f"Replicate Payload: {json.dumps(payload)}")

                    async with session.post(
                        api_url,
                        headers={
                            "Authorization": f"Token {self.valves.REPLICATE_API_TOKEN}",
                            "Content-Type": "application/json",
                            "Accept": "application/json",
                            "Prefer": "respond-async",
                        },
                        json=payload,
                        timeout=start_timeout,
                    ) as response:
                        if response.status >= 400:
                            error_text = await response.text()
                            logger.error(
                                f"API error response on start (status {response.status}): {error_text[:500]}"
                            )
                            try:
                                error_json = json.loads(error_text)
                                error_detail = error_json.get("detail", error_text)
                            except json.JSONDecodeError:
                                error_detail = error_text
                            raise aiohttp.ClientResponseError(
                                request_info=response.request_info,
                                history=response.history,
                                status=response.status,
                                message=f"Error from Replicate API: {error_detail}",
                                headers=response.headers,
                            )

                        result = await response.json()
                        prediction_id = result.get("id", "N/A")
                        status_code = response.status
                        logger.info(
                            f"Successfully initiated generation (Status: {status_code}, ID: {prediction_id})"
                        )

                        if not isinstance(result.get("urls"), dict) or not result[
                            "urls"
                        ].get("get"):
                            logger.error(
                                f"Invalid response structure (ID: {prediction_id}): Missing 'urls.get'. Response: {result}"
                            )
                            raise Exception(
                                "Invalid response from Replicate: Missing 'urls.get' for polling."
                            )

                        return result

            except aiohttp.ClientResponseError as e:
                logger.error(
                    f"API response error on start attempt {attempt+1}: {e.status} {e.message}"
                )
                last_error = e
                if e.status in (400, 401, 403, 404, 422):
                    logger.error(
                        f"Unrecoverable API error ({e.status}). Aborting generation start."
                    )
                    raise e
                if attempt == max_retries - 1:
                    raise Exception(
                        f"API request failed after {max_retries} attempts with status {e.status}: {e.message}"
                    )

            except aiohttp.ClientError as e:
                logger.error(
                    f"API connection/client error on start attempt {attempt+1}: {str(e)}"
                )
                last_error = e
                if attempt == max_retries - 1:
                    raise Exception(
                        f"API connection failed after {max_retries} attempts: {str(e)}"
                    )

            except asyncio.TimeoutError:
                logger.error(f"API start request timed out on attempt {attempt+1}")
                last_error = asyncio.TimeoutError("Start generation timed out")
                if attempt == max_retries - 1:
                    raise last_error

            except Exception as e:
                logger.error(
                    f"Unexpected error on start attempt {attempt+1}: {str(e)}",
                    exc_info=True,
                )
                last_error = e
                raise Exception(f"Unexpected error during generation start: {str(e)}")

            delay = (
                base_delay
                * (2**attempt)
                * (0.5 + 0.5 * (asyncio.get_event_loop().time() % 1))
            )
            delay = min(delay, 10)
            logger.info(f"Retrying generation start in {delay:.2f} seconds...")
            await asyncio.sleep(delay)

        final_error_msg = f"Failed to start generation after {max_retries} attempts."
        if last_error:
            final_error_msg += f" Last error: {str(last_error)}"
        logger.error(final_error_msg)
        raise Exception(final_error_msg)

    async def _yield_final_results(
        self,
        image_urls: List[str],  # Expect a list now for num_outputs
        prompt: str,
        input_params: Dict,
        seed: Any,
        metrics: Dict,
    ) -> AsyncIterator[str]:
        """
        Yields the final image(s) and metadata as SSE chunks using Markdown format.
        Handles multiple output images.
        """
        try:
            # Yield Image Chunks (Markdown)
            for i, image_url in enumerate(image_urls):
                image_md = f"![Generated Image {i+1}/{len(image_urls)}]({image_url})"
                yield self._create_sse_chunk(image_md, content_type="text/markdown")
                # Optional: Add a small delay or separator between multiple images
                if len(image_urls) > 1:
                    yield self._create_sse_chunk("---", content_type="text/markdown")

            # Yield Metadata Chunk (Markdown with HTML Details)
            predict_time_str = "N/A"
            if metrics.get("predict_time") is not None:
                try:
                    predict_time_str = f"{float(metrics['predict_time']):.2f}s"
                except (ValueError, TypeError):
                    pass

            total_time_str = "N/A"
            if metrics.get("total_time") is not None:
                try:
                    total_time_str = f"{float(metrics['total_time']):.2f}s"
                except (ValueError, TypeError):
                    pass

            metadata_md = f"""
<details>
<summary>Generation Details</summary>

* **Prompt:** {prompt}
* **Negative Prompt:** {input_params.get("negative_prompt", "N/A")}
* **Go Fast:** {input_params.get("go_fast", "N/A")}
* **Num Outputs:** {input_params.get("num_outputs", "N/A")}
* **Aspect Ratio:** {input_params.get("aspect_ratio", "N/A")}
* **Format:** {input_params.get("output_format", "N/A")}
* **Quality:** {input_params.get("output_quality", "N/A")}%
* **Safety Checker:** {"Disabled" if input_params.get("disable_safety_checker") else "Enabled"}
* **Seed:** {seed if seed is not None else "Random"}
* **Generation Time:** {predict_time_str}
* **Total Time:** {total_time_str}
</details>
"""
            yield self._create_sse_chunk(
                metadata_md.strip(), content_type="text/markdown"
            )

            yield self._create_sse_chunk({}, finish_reason="stop")
            yield "data: [DONE]\n\n"

            logger.debug("Yielded final results chunks.")

        except Exception as e:
            logger.error(f"Error yielding final results: {str(e)}", exc_info=True)
            yield self._create_sse_chunk(
                {"error": f"Error formatting final result: {str(e)}"},
                finish_reason="stop",
            )
            yield "data: [DONE]\n\n"

    async def pipe(self, body: Dict, __event_emitter__=None) -> AsyncIterator[str]:
        """
        Main entry point for the OpenWebUI Pipe for Flux Schnell.
        Handles validation, starting prediction, polling, processing, and yielding results.
        """
        start_time = time.time()
        prediction_id = "N/A"

        try:
            if not self.valves.REPLICATE_API_TOKEN:
                raise ValueError("REPLICATE_API_TOKEN is not configured.")

            prompt = self._extract_prompt(body)
            if not prompt:
                raise ValueError("No prompt provided in the request.")

            input_params = self._prepare_input_params(prompt)

        except ValueError as e:
            error_msg = f"Configuration/Input Error: {str(e)}"
            logger.error(error_msg)
            yield self._create_sse_chunk({"error": error_msg}, finish_reason="stop")
            yield "data: [DONE]\n\n"
            return

        except Exception as e:
            error_msg = f"Error preparing request: {str(e)}"
            logger.error(error_msg, exc_info=True)
            yield self._create_sse_chunk({"error": error_msg}, finish_reason="stop")
            yield "data: [DONE]\n\n"
            return

        prediction = None
        prediction_start_time = time.time()
        try:
            logger.info(
                f"Attempting to start Schnell generation for prompt: {prompt[:50]}..."
            )
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Starting Flux Schnell generation...",
                            "done": False,
                        },
                    }
                )

            prediction = await self._start_generation(input_params)
            prediction_id = prediction.get("id", "N/A")
            prediction_url = prediction["urls"]["get"]
            logger.info(
                f"Schnell generation started successfully (ID: {prediction_id}). Polling URL: {prediction_url}"
            )

        except Exception as e:
            error_msg = (
                f"Failed to start Schnell generation (ID: {prediction_id}): {str(e)}"
            )
            logger.error(error_msg, exc_info=True)
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": error_msg, "done": True}}
                )
            yield self._create_sse_chunk({"error": error_msg}, finish_reason="stop")
            yield "data: [DONE]\n\n"
            return

        result = None
        try:
            logger.info(f"Polling for Schnell completion (ID: {prediction_id})...")
            result = await self._wait_for_completion(prediction_url, __event_emitter__)
            logger.info(
                f"Schnell generation completed successfully (ID: {prediction_id})"
            )

        except Exception as e:
            error_msg = f"Error during Schnell generation/polling (ID: {prediction_id}): {str(e)}"
            logger.error(error_msg, exc_info=True)
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": error_msg, "done": True}}
                )
            yield self._create_sse_chunk({"error": error_msg}, finish_reason="stop")
            yield "data: [DONE]\n\n"
            return

        try:
            logger.info(f"Processing final Schnell result (ID: {prediction_id})...")
            metrics = result.get("metrics", {})

            if prediction and prediction.get("created_at") and datetime:
                try:
                    created_at_str = prediction["created_at"]
                    if created_at_str.endswith("Z"):
                        created_at_str = created_at_str[:-1] + "+00:00"
                    created_at = datetime.datetime.fromisoformat(created_at_str)
                    now_utc = datetime.datetime.now(datetime.timezone.utc)
                    if created_at.tzinfo is None:
                        created_at = created_at.replace(tzinfo=datetime.timezone.utc)
                    metrics["total_time"] = (now_utc - created_at).total_seconds()
                    logger.debug(
                        f"Calculated total_time based on created_at: {metrics['total_time']:.2f}s"
                    )
                except (ValueError, TypeError) as dt_error:
                    logger.warning(
                        f"Could not parse prediction created_at timestamp '{prediction.get('created_at')}': {dt_error}. Falling back to less accurate total time."
                    )
                    metrics["total_time"] = time.time() - prediction_start_time
            else:
                metrics["total_time"] = time.time() - prediction_start_time
                logger.debug(
                    f"Using fallback total_time calculation: {metrics['total_time']:.2f}s"
                )

            logs = result.get("logs", "")
            output_data = result.get("output")

            final_seed = None
            if isinstance(result.get("input"), dict):
                final_seed = result["input"].get("seed")
            if final_seed is None:
                final_seed = self._extract_seed_from_logs(logs)
            if final_seed is None:
                final_seed = input_params.get("seed")
            logger.debug(f"Final seed determined: {final_seed}")

            # --- Output Validation and URL Extraction ---
            # Schnell output is documented as List[str]
            image_urls_to_process = []
            if isinstance(output_data, list) and output_data:
                valid_urls = [
                    url
                    for url in output_data
                    if isinstance(url, str) and url.startswith("http")
                ]
                if valid_urls:
                    image_urls_to_process = valid_urls
                else:
                    logger.warning(
                        f"Output list did not contain valid URLs: {output_data}"
                    )

            if not image_urls_to_process:
                error_msg = "Error: Could not extract any valid image URLs from the prediction output."
                logger.error(
                    f"{error_msg} (ID: {prediction_id}). Result output: {output_data}"
                )
                raise ValueError(error_msg)

            async for chunk in self._yield_final_results(
                image_urls_to_process, prompt, input_params, final_seed, metrics
            ):
                yield chunk

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Image(s) generated successfully!",
                            "done": True,
                        },
                    }
                )
            total_pipe_time = time.time() - start_time
            logger.info(
                f"Successfully processed and yielded result for ID: {prediction_id}. Total pipe time: {total_pipe_time:.2f}s"
            )

        except Exception as e:
            error_msg = f"Error processing final result (ID: {prediction_id}): {str(e)}"
            logger.error(error_msg, exc_info=True)
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": error_msg, "done": True}}
                )
            yield self._create_sse_chunk({"error": error_msg}, finish_reason="stop")
            yield "data: [DONE]\n\n"