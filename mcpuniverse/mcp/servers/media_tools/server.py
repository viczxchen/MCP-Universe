"""
An MCP server providing basic multimodal helper tools.

Tools:
- read_image:  Validate an image URL or local path and return a usable URL/data URI.
- watch_video: Validate a video URL or local path and return a usable URL/path and
               basic metadata when available.

These tools are intended to be used by agents that want to construct multimodal
messages (e.g. OpenAI / Claude style blocks) without having to implement file/URL
handling and validation logic themselves.
"""

from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Union
from urllib.parse import urlparse

import click
import httpx
from mcp.server.fastmcp import FastMCP

from mcpuniverse.common.logger import get_logger


logger = get_logger("media_tools")


def _is_url(path_or_url: str) -> bool:
    """Return True if the input looks like an http(s) URL."""
    try:
        parsed = urlparse(path_or_url)
        return parsed.scheme in ("http", "https")
    except Exception:  # pragma: no cover - very defensive
        return False


async def _validate_url(url: str, timeout: float = 10.0) -> Dict[str, Any]:
    """
    Validate that a URL is reachable (best-effort).

    Returns a dict with:
        { "ok": bool, "status_code": int | None, "error": str | None }
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.head(url, follow_redirects=True)
        return {"ok": resp.status_code < 400, "status_code": resp.status_code, "error": None}
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning("URL validation failed for %s: %s", url, exc)
        return {"ok": False, "status_code": None, "error": str(exc)}


def _path_to_file_url(path: Path) -> str:
    """
    Convert a local file path to a file:// URL.

    This is lighter-weight than embedding the entire file as a base64 data URI
    in JSON, which can easily blow up the payload size for large images.
    """
    return path.as_uri()


def build_server(port: int) -> FastMCP:
    """
    Initializes the MCP server for media helper tools.

    :param port: Port for SSE.
    :return: The MCP server.
    """
    mcp = FastMCP("media_tools", port=port)

    @mcp.tool()
    async def read_image(path_or_url: Union[str, List[str]]) -> str:
        """
        Read/validate one or more images from local paths or URLs and return usable URLs.

        Args:
            path_or_url:
                - Either a single local filesystem path or http(s) URL (str)
                - Or a list of such paths/URLs (List[str]) for batch processing.

        Returns:
            JSON string.

            - If input is a single string, returns a single dict:
                  {
                    "ok": bool,
                    "source": "local" | "url",
                    "url": "<http or data URI>",
                    "path": "<absolute local path or null>",
                    "mime_type": "<best-effort mime type or null>",
                    "error": "<error message or null>"
                  }
            - If input is a list of strings, returns a list[dict]，其中每个元素的结构
              与上述单张图片的返回结构一致。
        """
        async def _handle_single(item: str) -> Dict[str, Any]:
            """Internal helper to process a single path/URL and return a result dict."""
            try:
                # URL case
                if _is_url(item):
                    validation = await _validate_url(item)
                    if not validation["ok"]:
                        return {
                            "ok": False,
                            "source": "url",
                            "url": item,
                            "path": None,
                            "mime_type": None,
                            "error": validation["error"]
                            or f"URL not reachable (status={validation['status_code']})",
                        }

                    mime, _ = mimetypes.guess_type(item)
                    return {
                        "ok": True,
                        "source": "url",
                        "url": item,
                        "path": None,
                        "mime_type": mime,
                        "error": None,
                    }

                # Local file case
                path = Path(item).expanduser().resolve()
                if not path.exists():
                    return {
                        "ok": False,
                        "source": "local",
                        "url": None,
                        "path": str(path),
                        "mime_type": None,
                        "error": f"File not found: {path}",
                    }

                # Represent local image as a file:// URL to avoid embedding large
                # base64 payloads in JSON. The vision-capable LLM backend can then
                # decide how to fetch/use this URL.
                file_url = _path_to_file_url(path)
                mime, _ = mimetypes.guess_type(str(path))
                return {
                    "ok": True,
                    "source": "local",
                    "url": file_url,
                    "path": str(path),
                    "mime_type": mime,
                    "error": None,
                }

            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.error("read_image failed for %s: %s", item, exc, exc_info=True)
                return {
                    "ok": False,
                    "source": "url" if _is_url(item) else "local",
                    "url": item if _is_url(item) else None,
                    "path": item if not _is_url(item) else None,
                    "mime_type": None,
                    "error": str(exc),
                }

        # Batch mode: list of paths/URLs
        if isinstance(path_or_url, (list, tuple)):
            results: List[Dict[str, Any]] = []
            for item in path_or_url:
                # Skip non-str items defensively
                if not isinstance(item, str):
                    results.append(
                        {
                            "ok": False,
                            "source": "local",
                            "url": None,
                            "path": str(item),
                            "mime_type": None,
                            "error": "Invalid path_or_url item type; expected string",
                        }
                    )
                    continue
                results.append(await _handle_single(item))
            return json.dumps(results, ensure_ascii=False, indent=2)

        # Single string case (backwards compatible behavior)
        if not isinstance(path_or_url, str):
            # 明确错误，避免类型误用
            return json.dumps(
                {
                    "ok": False,
                    "source": "local",
                    "url": None,
                    "path": str(path_or_url),
                    "mime_type": None,
                    "error": "Invalid type for path_or_url; expected string or list of strings",
                },
                ensure_ascii=False,
                indent=2,
            )

        result = await _handle_single(path_or_url)
        return json.dumps(result, ensure_ascii=False, indent=2)

    @mcp.tool()
    async def read_images(paths_or_urls: List[str]) -> str:
        """
        Batch version of `read_image` that processes multiple paths/URLs at once.

        Args:
            paths_or_urls: List of local filesystem paths or http(s) URLs.

        Returns:
            JSON string representing a list of results, where each element has the
            same schema as `read_image`:

                {
                  "ok": bool,
                  "source": "local" | "url",
                  "url": "<http or data URI>",
                  "path": "<absolute local path or null>",
                  "mime_type": "<best-effort mime type or null>",
                  "error": "<error message or null>"
                }
        """
        results: List[Dict[str, Any]] = []
        for item in paths_or_urls:
            try:
                # 复用单张图片的逻辑，保持行为一致
                single = await read_image(item)
                parsed = json.loads(single)
                if isinstance(parsed, dict):
                    results.append(parsed)
                else:
                    results.append(
                        {
                            "ok": False,
                            "source": "url" if _is_url(item) else "local",
                            "url": item if _is_url(item) else None,
                            "path": item if not _is_url(item) else None,
                            "mime_type": None,
                            "error": "read_image returned non-dict result",
                        }
                    )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.error("read_images failed for %s: %s", item, exc, exc_info=True)
                results.append(
                    {
                        "ok": False,
                        "source": "url" if _is_url(item) else "local",
                        "url": item if _is_url(item) else None,
                        "path": item if not _is_url(item) else None,
                        "mime_type": None,
                        "error": str(exc),
                    }
                )

        return json.dumps(results, ensure_ascii=False, indent=2)

    @mcp.tool()
    async def watch_video(path_or_url: str) -> str:
        """
        Validate a video URL or local path and return basic information.

        NOTE: This tool does *not* stream video data. It is intended to be used
        as a helper for agents to confirm that a link/path is valid, and to get
        a URL/path that can be referenced in multimodal prompts.

        Args:
            path_or_url: Either a local filesystem path or an http(s) URL.

        Returns:
            JSON string with fields:
                {
                  "ok": bool,
                  "source": "local" | "url",
                  "url": "<http url or file://...>",
                  "path": "<absolute local path or null>",
                  "error": "<error message or null>"
                }
        """
        try:
            # URL case
            if _is_url(path_or_url):
                validation = await _validate_url(path_or_url)
                if not validation["ok"]:
                    return json.dumps(
                        {
                            "ok": False,
                            "source": "url",
                            "url": path_or_url,
                            "path": None,
                            "error": validation["error"]
                            or f"URL not reachable (status={validation['status_code']})",
                        },
                        ensure_ascii=False,
                        indent=2,
                    )

                result = {
                    "ok": True,
                    "source": "url",
                    "url": path_or_url,
                    "path": None,
                    "error": None,
                }
                return json.dumps(result, ensure_ascii=False, indent=2)

            # Local file case
            path = Path(path_or_url).expanduser().resolve()
            if not path.exists():
                return json.dumps(
                    {
                        "ok": False,
                        "source": "local",
                        "url": None,
                        "path": str(path),
                        "error": f"File not found: {path}",
                    },
                    ensure_ascii=False,
                    indent=2,
                )

            # Represent local video as a file:// URL so that agents can reference it
            file_url = path.as_uri()
            result = {
                "ok": True,
                "source": "local",
                "url": file_url,
                "path": str(path),
                "error": None,
            }
            return json.dumps(result, ensure_ascii=False, indent=2)

        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("watch_video failed for %s: %s", path_or_url, exc, exc_info=True)
            return json.dumps(
                {
                    "ok": False,
                    "source": "url" if _is_url(path_or_url) else "local",
                    "url": path_or_url if _is_url(path_or_url) else None,
                    "path": path_or_url if not _is_url(path_or_url) else None,
                    "error": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            )

    return mcp


@click.command()
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    help="Transport type",
)
@click.option("--port", default="8000", help="Port to listen on for SSE")
def main(transport: str, port: str):
    """
    Starts the media_tools MCP server.
    """
    print(f"Starting media_tools MCP server on port {port} with transport {transport}")
    assert transport.lower() in ["stdio", "sse"], "Transport should be `stdio` or `sse`"
    logger.info("Starting the media_tools MCP server")
    mcp = build_server(int(port))
    mcp.run(transport=transport.lower())


