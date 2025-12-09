"""
An MCP server for Google search
"""
# pylint: disable=broad-exception-caught
import os
import json
import math
from typing import List, Dict, Any

import httpx
import click
from mcp.server.fastmcp import FastMCP
from mcpuniverse.common.logger import get_logger

SERP_API_BASE = "https://serpapi.com/search.json"
API_KEY = os.environ.get("SERP_API_KEY", "")


async def _search(
        query: str,
        location: str = "",
        engine: str = "google",
        num_items: int = 20,
        timeout: float = 30
) -> List[Dict[str, Any]]:
    """
    Make a request to the Serp API.

    :param query: The search query string.
    :param location: The location for the search query.
    :param engine: The search engine to use (default is "google").
    :param num_items: The maximum number of results to return.
    :param timeout: The timeout.
    """
    all_items = []
    num_pages = int(math.ceil(num_items / 10))
    num_items_per_page = 10

    for page in range(num_pages):
        offset = page * num_items_per_page
        params = {
            "api_key": API_KEY,
            "q": query,
            "location": location,
            "engine": engine,
            "num": num_items_per_page,
            "start": offset
        }
        async with httpx.AsyncClient() as client:
            response = await client.get(SERP_API_BASE, params=params, timeout=timeout)
            response.raise_for_status()
            results = response.json()
            all_items.extend([{
                "position": result.get("position") + offset,
                "title": result.get("title"),
                "snippet": result.get("snippet"),
                "link": result.get("link"),
            } for result in results.get("organic_results", [])])
    return all_items[:num_items]


async def _google_lens_search(
        image_url: str,
        query: str = "",
        search_type: str = "all",
        timeout: float = 30,
) -> Dict[str, Any]:
    """
    Make a request to the Serp API Google Lens engine.

    See SerpAPI Google Lens docs for response structure, including
    fields like `ai_overview`, `visual_matches`, and `related_content`.

    :param image_url: Publicly accessible image URL to analyze.
    :param query: Optional text query to refine search results. Only applicable when
                  search_type is 'all', 'visual_matches', or 'products'.
    :param search_type: Type of search to perform. Options: 'all', 'products',
                        'exact_matches', 'visual_matches'. Default is 'all'.
    :param timeout: Request timeout in seconds.
    :return: Parsed JSON response from SerpAPI.
    """
    params = {
        "api_key": API_KEY,
        "engine": "google_lens",
        "url": image_url,
        "type": search_type,
    }

    # Add query parameter if provided and type supports it
    if query and search_type in ["all", "visual_matches", "products"]:
        params["q"] = query

    async with httpx.AsyncClient() as client:
        response = await client.get(SERP_API_BASE, params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()


def build_server(port: int) -> FastMCP:
    """
    Initializes the MCP server.

    :param port: Port for SSE.
    :return: The MCP server.
    """
    mcp = FastMCP("google_search", port=port)

    @mcp.tool()
    async def search(query: str) -> str:
        """
        A tool to execute the Google search and return the top results.

        Args:
            query: The search query string.
        """
        if not API_KEY:
            return json.dumps(
                {"error": "SERP_API_KEY is not set in environment"},
                ensure_ascii=False,
            )
        try:
            items = await _search(query=query)
            return "\n".join([json.dumps(item, ensure_ascii=False, indent=2) for item in items])
        except Exception as e:
            return json.dumps({"error": f"Search failed: {str(e)}"}, ensure_ascii=False)

    @mcp.tool()
    async def google_lens(
        image_url: str,
        query: str = "",
        search_type: str = "all",
    ) -> str:
        """
        Use Google Lens via SerpAPI to find visually similar images and related content.

        This wraps SerpAPI's Google Lens API
        (see `https://serpapi.com/google-lens-api` for full response schema),
        and returns key fields such as `ai_overview`, `visual_matches`, and
        `related_content`.

        Args:
            image_url: Publicly accessible image URL to analyze.
            query: Optional text query to refine search results. Only applicable when
                   search_type is 'all', 'visual_matches', or 'products'.
            search_type: Type of search to perform. Options: 'all', 'products',
                        'exact_matches', 'visual_matches'. Default is 'all'.

        Returns:
            JSON string containing the structured Google Lens response.
        """
        if not API_KEY:
            return json.dumps(
                {"error": "SERP_API_KEY is not set in environment"},
                ensure_ascii=False,
            )

        # Validate search_type
        valid_types = ["all", "products", "exact_matches", "visual_matches"]
        if search_type not in valid_types:
            return json.dumps(
                {
                    "error": f"Invalid search_type '{search_type}'. Must be one of: {', '.join(valid_types)}"
                },
                ensure_ascii=False,
            )

        try:
            raw = await _google_lens_search(
                image_url=image_url, query=query, search_type=search_type
            )

            # Extract key sections commonly used from the Google Lens API
            ai_overview = raw.get("ai_overview")
            visual_matches = raw.get("visual_matches", [])
            related_content = raw.get("related_content", [])
            exact_matches = raw.get("exact_matches", [])
            products = raw.get("products", [])

            result = {
                "image_url": image_url,
                "query": query if query else None,
                "search_type": search_type,
                "ai_overview": ai_overview,
                "visual_matches_count": len(visual_matches),
                "visual_matches": visual_matches,
                "exact_matches_count": len(exact_matches) if exact_matches else 0,
                "exact_matches": exact_matches if exact_matches else None,
                "products_count": len(products) if products else 0,
                "products": products if products else None,
                "related_content": related_content,
                "raw": raw,
            }

            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as e:
            return json.dumps(
                {"error": f"Google Lens search failed: {str(e)}"}, ensure_ascii=False
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
    Starts the initialized MCP server.

    :param port: Port for SSE.
    :param transport: The transport type, e.g., `stdio` or `sse`.
    """
    print(f"Starting the MCP server on port {port} with transport {transport}")
    assert transport.lower() in ["stdio", "sse"], \
        "Transport should be `stdio` or `sse`"
    logger = get_logger("Service:google_search")
    logger.info("Starting the MCP server")
    mcp = build_server(int(port))
    mcp.run(transport=transport.lower())
