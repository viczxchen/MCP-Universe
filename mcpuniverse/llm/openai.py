"""
OpenAI LLMs
"""
# pylint: disable=broad-exception-caught
import os
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union, Optional, Type, List
from urllib.parse import urlparse

from openai import OpenAI
from openai import RateLimitError, APIError, APITimeoutError
from dotenv import load_dotenv
from pydantic import BaseModel as PydanticBaseModel
import mimetypes

from mcpuniverse.common.config import BaseConfig
from mcpuniverse.common.context import Context
from .base import BaseLLM

load_dotenv()


@dataclass
class OpenAIConfig(BaseConfig):
    """
    Configuration for OpenAI language models.

    Attributes:
        model_name (str): The name of the OpenAI model to use (default: "gpt-4o").
        api_key (str): The OpenAI API key (default: environment variable OPENAI_API_KEY).
        temperature (float): Controls randomness in output (default: 1.0).
        top_p (float): Controls diversity of output (default: 1.0).
        frequency_penalty (float): Penalizes frequent token use (default: 0.0).
        presence_penalty (float): Penalizes repeated topics (default: 0.0).
        max_completion_tokens (int): Maximum number of tokens in the completion (default: 2048).
        reasoning_effort (str): The reasoning effort to use (default: "medium").
        seed (int): Random seed for reproducibility (default: 12345).
    """
    model_name: str = "gpt-4.1"
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    temperature: float = 1.0
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_completion_tokens: int = 10000
    reasoning_effort: str = "medium"
    seed: int = 12345


class OpenAIModel(BaseLLM):
    """
    OpenAI language models.

    This class provides methods to interact with OpenAI's language models,
    including generating responses based on input messages.

    Attributes:
        config_class (Type[OpenAIConfig]): Configuration class for the model.
        alias (str): Alias for the model, used for identification.
    """
    config_class = OpenAIConfig
    alias = "openai"
    env_vars = ["OPENAI_API_KEY"]

    def __init__(self, config: Optional[Union[Dict, str]] = None):
        super().__init__()
        self.config = OpenAIModel.config_class.load(config)

    def _generate(
            self,
            messages: List[Dict[str, Any]],
            response_format: Type[PydanticBaseModel] = None,
            **kwargs
    ):
        """
        Generates content using the OpenAI model.

        Args:
            messages (List[dict[str, str]]): List of message dictionaries,
                each containing 'role' and 'content' keys.
            response_format (Type[PydanticBaseModel], optional): Pydantic model
                defining the structure of the desired output. If None, generates
                free-form text.
            **kwargs: Additional keyword arguments including:
                - max_retries (int): Maximum number of retry attempts (default: 5)
                - base_delay (float): Base delay in seconds for exponential backoff (default: 10.0)
                - timeout (int): Request timeout in seconds (default: 60)

        Returns:
            Union[str, PydanticBaseModel, None]: Generated content as a string
                if no response_format is provided, a Pydantic model instance if
                response_format is provided, or None if parsing structured output fails.
                Returns None if all retry attempts fail or non-retryable errors occur.
        """
        max_retries = kwargs.get("max_retries", 5)
        base_delay = kwargs.get("base_delay", 10.0)

        kwargs.pop("tracer", None)
        kwargs.pop("callbacks", None)

        # Before calling OpenAI, normalize messages so that any local file URLs
        # (e.g. file://...) are converted into data URIs that the remote model
        # can actually see. This allows agents to use tools like media_tools.read_image
        # to validate local paths, while still giving the vision model access to
        # the underlying image content.
        # Get container path mapping from context if available
        container_path_mapping = None
        if self._context:
            container_path_mapping = self._context.metadata.get("container_path_mapping")
        
        # Debug: log multimodal message details
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "image_url":
                        url = block.get("image_url", {}).get("url", "")
                        logging.info(f"[OpenAIModel] Found image_url block: {url[:80]}...")
                        logging.info(f"[OpenAIModel] container_path_mapping: {container_path_mapping}")
        
        normalized_messages = self._normalize_messages_with_local_images(
            messages, container_path_mapping=container_path_mapping
        )

        for attempt in range(max_retries + 1):
            try:
                client = OpenAI(api_key=self.config.api_key,base_url="https://api.agicto.cn/v1")
                # Models support the 'reasoning_effort' parameter.
                # This set can be extended as new models are introduced.
                _models_with_reasoning_effort_support = {"gpt-5", "o3", "o4-mini", "gpt-5-high"}
                if any(prefix in self.config.model_name
                       for prefix in _models_with_reasoning_effort_support):
                    kwargs["reasoning_effort"] = self.config.reasoning_effort

                if "high" in self.config.model_name:
                    kwargs["reasoning_effort"] = "high"
                    self.config.model_name = "gpt-5"

                if response_format is None:
                    chat = client.chat.completions.create(
                        messages=normalized_messages,
                        model=self.config.model_name,
                        temperature=self.config.temperature,
                        timeout=int(kwargs.get("timeout", 60)),
                        top_p=self.config.top_p,
                        frequency_penalty=self.config.frequency_penalty,
                        presence_penalty=self.config.presence_penalty,
                        max_completion_tokens=self.config.max_completion_tokens,
                        seed=self.config.seed,
                        **kwargs
                    )
                    # If tools are provided, return the entire response object
                    # so the caller can handle both content and tool_calls
                    if 'tools' in kwargs:
                        return chat
                    # For backward compatibility, return just content when no tools
                    return chat.choices[0].message.content

                chat = client.beta.chat.completions.parse(
                    messages=normalized_messages,
                    model=self.config.model_name,
                    temperature=self.config.temperature,
                    timeout=int(kwargs.get("timeout", 60)),
                    top_p=self.config.top_p,
                    frequency_penalty=self.config.frequency_penalty,
                    presence_penalty=self.config.presence_penalty,
                    max_completion_tokens=self.config.max_completion_tokens,
                    seed=self.config.seed,
                    response_format=response_format,
                    **kwargs
                )
                # If tools are provided, return the entire response object
                # so the caller can handle both content and tool_calls
                if 'tools' in kwargs:
                    return chat
                # For backward compatibility, return just parsed content when no tools
                return chat.choices[0].message.parsed

            except (RateLimitError, APIError, APITimeoutError) as e:
                if attempt == max_retries:
                    # Last attempt failed, return None instead of raising
                    logging.warning("All %d attempts failed. Last error: %s", max_retries + 1, e)
                    return None

                # Calculate delay with exponential backoff
                delay = base_delay * (2 ** attempt)
                logging.info("Attempt %d failed with error: %s. Retrying in %.1f seconds...",
                           attempt + 1, e, delay)
                time.sleep(delay)

            except Exception as e:
                # For non-retryable errors, return None instead of raising
                logging.error("Non-retryable error occurred: %s", e)
                return None

    def _normalize_messages_with_local_images(
            self,
            messages: List[Dict[str, Any]],
            container_path_mapping: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Convert any file:// image URLs in messages into data URIs.

        This bridges the gap between local tooling (e.g. media_tools.read_image
        returning file:// URLs) and remote OpenAI vision models that require
        either http(s) URLs or data:image/...;base64,... URIs.

        Args:
            messages: List of message dictionaries.
            container_path_mapping: Optional mapping from container paths to host paths.
                e.g., {"/workspace": "/Users/.../task_dir"}
                If provided, container paths (like file:///workspace/xxx.png) will be
                mapped to host paths before reading.
        """
        normalized: List[Dict[str, Any]] = []

        # Get path mapping from context if available and not explicitly provided
        if container_path_mapping is None and self._context:
            # Try to get from context metadata
            container_path_mapping = self._context.metadata.get("container_path_mapping")

        for msg in messages:
            msg_copy = dict(msg)
            content = msg_copy.get("content")

            # Only need to touch messages whose content is a list of blocks
            if isinstance(content, list):
                new_blocks: List[Dict[str, Any]] = []
                for block in content:
                    block_copy = dict(block)
                    if block_copy.get("type") in ("image_url", "input_image"):
                        image_url = block_copy.get("image_url", {})
                        url = image_url.get("url") if isinstance(image_url, dict) else None
                        if isinstance(url, str) and url.startswith("file://"):
                            # Convert file:// URL to a data URI
                            try:
                                parsed = urlparse(url)
                                container_path = Path(parsed.path)
                                
                                # Map container path to host path if mapping is provided
                                host_path = container_path
                                if container_path_mapping:
                                    container_path_str = str(container_path)
                                    # Find matching mapping (e.g., /workspace -> /Users/.../task_dir)
                                    for container_prefix, host_prefix in container_path_mapping.items():
                                        if container_path_str.startswith(container_prefix):
                                            # Replace container prefix with host prefix
                                            relative_path = container_path_str[len(container_prefix):].lstrip("/")
                                            host_path = Path(host_prefix) / relative_path
                                            break
                                
                                logging.info(f"[OpenAIModel] Checking host_path: {host_path}, exists: {host_path.exists()}")
                                if host_path.exists():
                                    mime, _ = mimetypes.guess_type(str(host_path))
                                    if not mime:
                                        mime = "application/octet-stream"
                                    with host_path.open("rb") as f:
                                        import base64
                                        b64 = base64.b64encode(f.read()).decode("ascii")
                                    data_uri = f"data:{mime};base64,{b64}"
                                    logging.info(f"[OpenAIModel] Converted {url} to base64 data URI (len={len(data_uri)})")
                                    # Update URL to data URI so that OpenAI can see the image
                                    if isinstance(image_url, dict):
                                        image_url = dict(image_url)
                                        image_url["url"] = data_uri
                                        block_copy["image_url"] = image_url
                                else:
                                    logging.warning(f"[OpenAIModel] Host path does not exist: {host_path}")
                            except Exception as exc:  # pragma: no cover - defensive
                                logging.warning("Failed to convert local image %s to data URI: %s", url, exc)
                    new_blocks.append(block_copy)

                # Optional reordering: put image blocks in front of text blocks.
                image_first: List[Dict[str, Any]] = []
                others: List[Dict[str, Any]] = []
                for b in new_blocks:
                    if b.get("type") in ("image_url", "input_image"):
                        image_first.append(b)
                    else:
                        others.append(b)
                msg_copy["content"] = image_first + others

            normalized.append(msg_copy)

        return normalized

    def set_context(self, context: Context):
        """
        Set context, e.g., environment variables (API keys).
        """
        super().set_context(context)
        self.config.api_key = context.env.get("OPENAI_API_KEY", self.config.api_key)
