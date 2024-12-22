"""
title: FLUX Schnell Manifold Function for Black Forest Lab Image Generation Models
author: Balaxxe, credit to mobilestack and bgeneto
author_url: https://github.com/jaim12005/open-webui-flux-1.1-pro-ultra
funding_url: https://github.com/open-webui
version: 1.5
license: MIT
requirements: pydantic>=2.0.0, aiohttp>=3.8.1
environment_variables: 
    - REPLICATE_API_TOKEN (required)
    - FLUX_GO_FAST (optional, default: true)
    - FLUX_DISABLE_SAFETY (optional, default: false)
    - FLUX_SEED (optional)
    - FLUX_ASPECT_RATIO (optional, default: "1:1")
    - FLUX_OUTPUT_FORMAT (optional, default: "webp")
    - FLUX_OUTPUT_QUALITY (optional, default: 80)
    - FLUX_NUM_OUTPUTS (optional, default: 1)
supported providers: replicate.com

NOTE: Due to the asynchronous nature of the Replicate API, each image generation will make 2-3 (rare occasion 4) API requests:
1. Initial request to start generation
2. Follow-up request(s) to check completion status
This is normal behavior and required by the API design. You will typically see only 2 requests after the first generation.

NOTE: Your thread will appear blank on the left hand side. Also PNG files slow down the interface 
for some reason. Generate and save your images, then delete the thread.
"""

from typing import Dict, AsyncIterator, Optional, Literal, cast, Union, List
from pydantic import BaseModel, Field
import os
import base64
import aiohttp
import asyncio
from datetime import datetime
import json
import uuid
import time

AspectRatioType = Literal["1:1", "16:9", "21:9", "3:2", "2:3", "4:5", "5:4", "3:4", "4:3", "9:16", "9:21"]
OutputFormatType = Literal["webp", "jpg", "png"]
MegapixelsType = Literal["1", "0.25"]

class Pipe:
    class Valves(BaseModel):
        REPLICATE_API_TOKEN: str = Field(
            default="", description="Your Replicate API token"
        )
        FLUX_GO_FAST: bool = Field(
            default=True, description="Enable fast mode"
        )
        FLUX_DISABLE_SAFETY: bool = Field(
            default=False, description="Disable the built-in safety checker (API only)"
        )
        FLUX_SEED: Optional[int] = Field(
            default=None, description="Random seed for reproducible generations"
        )
        FLUX_ASPECT_RATIO: AspectRatioType = Field(
            default="1:1", description="Output image aspect ratio"
        )
        FLUX_OUTPUT_FORMAT: OutputFormatType = Field(
            default="webp", description="Output image format"
        )
        FLUX_OUTPUT_QUALITY: int = Field(
            default=80, description="Output image quality (1-100)"
        )
        FLUX_NUM_OUTPUTS: int = Field(
            default=1, description="Number of images to generate"
        )

    def __init__(self):
        self.type = "pipe"
        self.id = "flux_schnell"
        self.name = "Flux Schnell"
        self.MODEL_URL = "https://api.replicate.com/v1/models/black-forest-labs/flux-schnell/predictions"
        self.valves = self.Valves(**{
            k: v for k, v in {
                "REPLICATE_API_TOKEN": os.getenv("REPLICATE_API_TOKEN", ""),
                "FLUX_SEED": int(os.getenv("FLUX_SEED")) if os.getenv("FLUX_SEED") else None,
                "FLUX_ASPECT_RATIO": os.getenv("FLUX_ASPECT_RATIO", "1:1"),
                "FLUX_OUTPUT_FORMAT": os.getenv("FLUX_OUTPUT_FORMAT", "webp"),
                "FLUX_GO_FAST": bool(os.getenv("FLUX_GO_FAST", True)),
                "FLUX_NUM_OUTPUTS": int(os.getenv("FLUX_NUM_OUTPUTS", "1")),
                "FLUX_OUTPUT_QUALITY": int(os.getenv("FLUX_OUTPUT_QUALITY", "80")),
            }.items() if v is not None
        })

    def _get_status(self, message: str) -> str:
        """Format a status message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        return f"[{timestamp}] {message}"

    async def _process_image(self, url_or_data: str, prompt: str, params: Dict, stream: bool = True) -> Union[str, List[str]]:
        """Process image data and return it in SSE format."""
        if url_or_data.startswith("http"):
            async with aiohttp.ClientSession() as session:
                async with session.get(url_or_data, timeout=30) as response:
                    response.raise_for_status()
                    image_data = base64.b64encode(await response.read()).decode("utf-8")
                    content_type = response.headers.get("Content-Type", f"image/{self.valves.FLUX_OUTPUT_FORMAT}")
                    image_url = f"data:{content_type};base64,{image_data}"
        else:
            image_url = url_or_data

        if not stream:
            return f"""<div class="generated-image">
                <img src="{image_url}" alt="Generated Image" />
                <details>
                    <summary>Generation Details</summary>
                    <ul>
                        <li>Prompt: {prompt}</li>
                        <li>Aspect Ratio: {params["aspect_ratio"]}</li>
                        <li>Format: {params["output_format"]}</li>
                        <li>Quality: {params["output_quality"]}%</li>
                        <li>Seed: {params.get("seed", "Random")}</li>
                    </ul>
                </details>
            </div>"""

        # For streaming mode, return chunks
        responses = []
        responses.append(self._create_sse_chunk(
            f'<div class="generated-image-container">'
            f'<img src="{image_url}" alt="Generated Image" style="max-width: 100%; height: auto;" />'
        ))
        
        metadata = (
            f'<div class="image-metadata">'
            f'<details><summary>Generation Details</summary>'
            f'<p><strong>Prompt:</strong> {prompt}</p>'
            f'<ul>'
            f'<li>Aspect Ratio: {params["aspect_ratio"]}</li>'
            f'<li>Format: {params["output_format"]}</li>'
            f'<li>Quality: {params["output_quality"]}%</li>'
            f'<li>Seed: {params.get("seed", "Random")}</li>'
            f'</ul></details></div></div>'
        )
        responses.append(self._create_sse_chunk(metadata))
        responses.append(self._create_sse_chunk({}, finish_reason="stop"))
        responses.append("data: [DONE]\n\n")
        
        return responses

    def _create_sse_chunk(self, content: Union[str, Dict], content_type: str = "text/html", finish_reason: Optional[str] = None) -> str:
        """Create a Server-Sent Events chunk."""
        chunk_data = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "flux-schnell",
            "choices": [{
                "delta": {} if finish_reason else {
                    "role": "assistant",
                    "content": content,
                    "content_type": content_type
                },
                "index": 0,
                "finish_reason": finish_reason
            }]
        }
        return f"data: {json.dumps(chunk_data)}\n\n"

    async def _wait_for_completion(self, prediction_url: str, __event_emitter__=None) -> Dict:
        headers = {
            "Authorization": f"Token {self.valves.REPLICATE_API_TOKEN}",
            "Accept": "application/json",
            "Prefer": "wait=30"
        }
        
        async with aiohttp.ClientSession() as session:
            await asyncio.sleep(0.5)
            
            async with session.get(prediction_url, headers=headers, timeout=35) as response:
                response.raise_for_status()
                result = await response.json()
                if result.get("status") in ["succeeded", "failed", "canceled"]:
                    return result
                
            await asyncio.sleep(0.3)
            
            async with session.get(prediction_url, headers=headers, timeout=35) as response:
                response.raise_for_status()
                result = await response.json()
                if result.get("status") in ["succeeded", "failed", "canceled"]:
                    return result
                
            await asyncio.sleep(0.3)
            async with session.get(prediction_url, headers=headers, timeout=35) as response:
                response.raise_for_status()
                final_result = await response.json()
                if final_result.get("status") in ["succeeded", "failed", "canceled"]:
                    return final_result
                raise Exception(f"Generation incomplete after {final_result.get('status')} status")

    async def pipe(self, body: Dict, __event_emitter__=None) -> AsyncIterator[str]:
        if not self.valves.REPLICATE_API_TOKEN:
            yield "Error: REPLICATE_API_TOKEN is required"
            return

        try:
            prompt = (body.get("messages", [{}])[-1].get("content", "") or "").strip()
            if not prompt:
                yield "Error: No prompt provided"
                return

            input_params = {
                "prompt": prompt,
                "go_fast": self.valves.FLUX_GO_FAST,
                "num_outputs": self.valves.FLUX_NUM_OUTPUTS,
                "aspect_ratio": self.valves.FLUX_ASPECT_RATIO,
                "output_format": self.valves.FLUX_OUTPUT_FORMAT,
                "output_quality": self.valves.FLUX_OUTPUT_QUALITY,
                "disable_safety_checker": self.valves.FLUX_DISABLE_SAFETY
            }
            if self.valves.FLUX_SEED is not None:
                input_params["seed"] = self.valves.FLUX_SEED

            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Starting Flux Schnell generation...", "done": False}
                })

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.MODEL_URL,
                    headers={
                        "Authorization": f"Token {self.valves.REPLICATE_API_TOKEN}",
                        "Content-Type": "application/json",
                        "Prefer": "wait=30"
                    },
                    json={"input": input_params},
                    timeout=35
                ) as response:
                    response.raise_for_status()
                    prediction = await response.json()

                result = await self._wait_for_completion(prediction["urls"]["get"], __event_emitter__)
                
                if result.get("status") != "succeeded":
                    raise Exception(f"Generation failed: {result.get('error', 'Unknown error')}")

                metrics = result.get("metrics", {})
                logs = result.get("logs", "")
                seed = logs.split("Using seed:")[1].split()[0].strip() if "Using seed:" in logs else None
                image_url = result["output"][0] if isinstance(result.get("output"), list) else result.get("output")
                
                if not image_url:
                    raise Exception("No valid output URL in prediction result")

                if __event_emitter__:
                    await __event_emitter__({
                        "type": "message",
                        "data": {
                            "content": f"![Generated Image]({image_url})",
                            "content_type": "text/markdown"
                        }
                    })

                    await __event_emitter__({
                        "type": "message",
                        "data": {
                            "content": f"""<details>
<summary>Generation Details</summary>

- **Prompt:** {prompt}
- **Aspect Ratio:** {input_params["aspect_ratio"]}
- **Format:** {input_params["output_format"]}
- **Quality:** {input_params["output_quality"]}%
- **Seed:** {seed or input_params.get("seed", "Random")}
- **Generation Time:** {metrics.get("predict_time", "N/A")}s
- **Total Time:** {metrics.get("total_time", "N/A")}s
- **Safe Images:** {logs.split("Total safe images: ")[1].split(" ")[0] if "Total safe images:" in logs and not self.valves.FLUX_DISABLE_SAFETY else "Disabled"}
</details>""",
                            "content_type": "text/markdown"
                        }
                    })

                    await __event_emitter__({
                        "type": "status",
                        "data": {"description": "Image generated successfully!", "done": True}
                    })

                yield ""

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": error_msg, "done": True}
                })
            yield error_msg
