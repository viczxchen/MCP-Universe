"""
An MCP server for Bilibili video operations
"""
import os
import json
import re
import click
import httpx
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from mcp.server.fastmcp import FastMCP
from mcpuniverse.common.logger import get_logger


# Bilibili API endpoints
BILIBILI_VIEW_API = "https://api.bilibili.com/x/web-interface/view"
BILIBILI_PLAYURL_API = "https://api.bilibili.com/x/player/wbi/playurl"
BILIBILI_SEARCH_API = "https://api.bilibili.com/x/web-interface/search/all/v2"  # Use old API without WBI signature
BILIBILI_SEARCH_WBI_API = "https://api.bilibili.com/x/web-interface/wbi/search/all/v2"  # New API with WBI signature
BILIBILI_COMMENTS_API = "https://api.bilibili.com/x/v2/reply"
BILIBILI_SUBTITLES_API = "https://api.bilibili.com/x/player/v2"
BILIBILI_HOME_URL = "https://www.bilibili.com"

# Default headers for Bilibili API requests
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.bilibili.com"
}

# Quality mapping
QUALITY_MAP = {
    "240p": 6,
    "360p": 16,
    "480p": 32,
    "720p": 64,
    "720p60": 74,
    "1080p": 80,
    "1080p+": 112,
    "1080p60": 116,
    "4k": 120,
    "best": 80  # Default to 1080p for "best"
}


def parse_video_id(video_id: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Parse video ID to extract bvid and aid.

    Args:
        video_id: Video ID in format BVxxxxxx or AVxxxxxx or just the number

    Returns:
        Tuple of (bvid, aid)
    """
    video_id = video_id.strip().upper()

    # Check if it's a BV number
    if video_id.startswith("BV"):
        return video_id, None

    # Check if it's an AV number
    if video_id.startswith("AV"):
        aid = int(video_id[2:])
        return None, aid

    # Try to parse as number (assume it's an aid)
    try:
        aid = int(video_id)
        return None, aid
    except ValueError:
        # If it looks like a BV number without prefix
        if re.match(r'^[A-Z0-9]{10}$', video_id):
            return f"BV{video_id}", None

    return None, None


async def get_video_info(bvid: Optional[str] = None, aid: Optional[int] = None, sessdata: Optional[str] = None) -> Dict[str, Any]:
    """
    Get video information including cid.

    Args:
        bvid: Bilibili video ID (BV number)
        aid: Bilibili article ID (AV number)
        sessdata: Optional SESSDATA cookie for authentication

    Returns:
        Video information dictionary
    """
    params = {}
    if bvid:
        params["bvid"] = bvid
    elif aid:
        params["aid"] = aid
    else:
        raise ValueError("Either bvid or aid must be provided")

    # Get base cookies first to avoid -412 error
    base_cookies = await get_bilibili_cookies()

    # Merge with SESSDATA if provided
    cookies = base_cookies.copy()
    if sessdata:
        cookies["SESSDATA"] = sessdata

    # Build cookie string
    cookie_str = "; ".join([f"{k}={v}" for k, v in cookies.items()])

    headers = DEFAULT_HEADERS.copy()
    headers["Cookie"] = cookie_str

    async with httpx.AsyncClient(headers=headers, cookies=cookies, timeout=30.0) as client:
        response = await client.get(BILIBILI_VIEW_API, params=params)
        response.raise_for_status()
        data = response.json()

        if data.get("code") != 0:
            raise ValueError(f"Failed to get video info: {data.get('message', 'Unknown error')}")

        return data.get("data", {})


async def get_video_stream_url(
    bvid: str,
    cid: int,
    qn: int = 32,
    fnval: int = 1,
    sessdata: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get video stream URL.

    Args:
        bvid: Bilibili video ID
        cid: Content ID
        qn: Quality code (32 for 480P)
        fnval: Format code (1 for MP4)
        sessdata: Optional SESSDATA cookie for authentication

    Returns:
        Video stream information
    """
    params = {
        "bvid": bvid,
        "cid": cid,
        "qn": qn,
        "fnval": fnval,
        "fnver": 0,
        "otype": "json",
        "platform": "html5"  # Use html5 platform to avoid anti-leech verification
    }

    headers = DEFAULT_HEADERS.copy()
    if sessdata:
        headers["Cookie"] = f"SESSDATA={sessdata}"

    async with httpx.AsyncClient(headers=headers, timeout=30.0) as client:
        response = await client.get(BILIBILI_PLAYURL_API, params=params)
        response.raise_for_status()
        data = response.json()

        if data.get("code") != 0:
            raise ValueError(f"Failed to get video stream: {data.get('message', 'Unknown error')}")

        return data.get("data", {})


async def download_file(url: str, output_path: str, headers: Optional[Dict[str, str]] = None) -> str:
    """
    Download a file from URL.

    Args:
        url: File URL
        output_path: Output file path
        headers: Optional headers for the request

    Returns:
        Path to downloaded file
    """
    download_headers = DEFAULT_HEADERS.copy()
    if headers:
        download_headers.update(headers)

    async with httpx.AsyncClient(headers=download_headers, timeout=300.0) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()

            # Ensure output directory exists
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Download file
            with open(output_path, "wb") as f:
                async for chunk in response.aiter_bytes():
                    f.write(chunk)

    return output_path


async def get_video_comments(
    aid: int,
    page: int = 1,
    page_size: int = 20,
    sort: str = "hot",
    sessdata: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get comments for a Bilibili video.

    Args:
        aid: Article ID (AV number)
        page: Page number
        page_size: Number of comments per page
        sort: Sort order - "hot" for hot comments, "time" for time-based
        sessdata: Optional SESSDATA cookie for authentication

    Returns:
        Comments data dictionary
    """
    params = {
        "type": 1,  # 1 for video comments
        "oid": aid,  # oid is the aid for videos
        "pn": page,
        "ps": page_size,
        "sort": 0 if sort == "hot" else 1  # 0 for hot, 1 for time
    }

    headers = DEFAULT_HEADERS.copy()
    if sessdata:
        headers["Cookie"] = f"SESSDATA={sessdata}"

    async with httpx.AsyncClient(headers=headers, timeout=30.0) as client:
        response = await client.get(BILIBILI_COMMENTS_API, params=params)
        response.raise_for_status()
        data = response.json()

        if data.get("code") != 0:
            raise ValueError(f"Failed to get comments: {data.get('message', 'Unknown error')}")

        return data.get("data", {})


async def get_video_subtitles(
    bvid: str,
    cid: int,
    sessdata: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get subtitles for a Bilibili video.

    Args:
        bvid: Bilibili video ID
        cid: Content ID
        sessdata: Optional SESSDATA cookie for authentication

    Returns:
        Subtitles data dictionary
    """
    params = {
        "bvid": bvid,
        "cid": cid
    }

    headers = DEFAULT_HEADERS.copy()
    if sessdata:
        headers["Cookie"] = f"SESSDATA={sessdata}"

    async with httpx.AsyncClient(headers=headers, timeout=30.0) as client:
        response = await client.get(BILIBILI_SUBTITLES_API, params=params)
        response.raise_for_status()
        data = response.json()

        if data.get("code") != 0:
            raise ValueError(f"Failed to get subtitles: {data.get('message', 'Unknown error')}")

        return data.get("data", {})


async def download_subtitle_file(subtitle_url: str) -> Dict[str, Any]:
    """
    Download subtitle file from URL.

    Args:
        subtitle_url: Subtitle file URL

    Returns:
        Parsed subtitle data
    """
    headers = DEFAULT_HEADERS.copy()

    async with httpx.AsyncClient(headers=headers, timeout=30.0) as client:
        response = await client.get(subtitle_url)
        response.raise_for_status()
        return response.json()


def convert_subtitle_to_srt(subtitle_data: Dict[str, Any]) -> str:
    """
    Convert subtitle JSON to SRT format.

    Args:
        subtitle_data: Subtitle data from API

    Returns:
        SRT formatted string
    """
    body = subtitle_data.get("body", [])
    srt_lines = []

    for idx, item in enumerate(body, 1):
        from_time = item.get("from", 0)
        to_time = item.get("to", 0)
        content = item.get("content", "")

        # Convert seconds to SRT time format (HH:MM:SS,mmm)
        def format_time(seconds: float) -> str:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds % 1) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

        srt_lines.append(f"{idx}")
        srt_lines.append(f"{format_time(from_time)} --> {format_time(to_time)}")
        srt_lines.append(content)
        srt_lines.append("")

    return "\n".join(srt_lines)


def convert_subtitle_to_vtt(subtitle_data: Dict[str, Any]) -> str:
    """
    Convert subtitle JSON to VTT format.

    Args:
        subtitle_data: Subtitle data from API

    Returns:
        VTT formatted string
    """
    body = subtitle_data.get("body", [])
    vtt_lines = ["WEBVTT", ""]

    for item in body:
        from_time = item.get("from", 0)
        to_time = item.get("to", 0)
        content = item.get("content", "")

        # Convert seconds to VTT time format (HH:MM:SS.mmm)
        def format_time(seconds: float) -> str:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds % 1) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

        vtt_lines.append(f"{format_time(from_time)} --> {format_time(to_time)}")
        vtt_lines.append(content)
        vtt_lines.append("")

    return "\n".join(vtt_lines)


async def get_bilibili_cookies() -> Dict[str, str]:
    """
    Get cookies from Bilibili homepage to avoid -412 error.

    Returns:
        Dictionary of cookies
    """
    async with httpx.AsyncClient(headers=DEFAULT_HEADERS, timeout=30.0, follow_redirects=True) as client:
        # Visit homepage to get cookies
        response = await client.get(BILIBILI_HOME_URL)
        response.raise_for_status()

        # Extract cookies from response
        cookies = {}
        for cookie in client.cookies.jar:
            cookies[cookie.name] = cookie.value

        return cookies


async def search_bilibili(keyword: str) -> Dict[str, Any]:
    """
    Search Bilibili for videos and other content.

    Args:
        keyword: Search keyword

    Returns:
        Search results dictionary
    """
    logger = get_logger("bilibili-video-tool")
    # Get cookies first to avoid -412 error
    cookies = await get_bilibili_cookies()

    # Prepare headers with cookies
    headers = DEFAULT_HEADERS.copy()
    if cookies:
        cookie_str = "; ".join([f"{k}={v}" for k, v in cookies.items()])
        headers["Cookie"] = cookie_str

    params = {
        "keyword": keyword
    }

    # Try old API first (doesn't require WBI signature)
    async with httpx.AsyncClient(headers=headers, timeout=30.0, cookies=cookies) as client:
        try:
            response = await client.get(BILIBILI_SEARCH_API, params=params)
            response.raise_for_status()
            data = response.json()

            if data.get("code") == -412:
                raise ValueError("Search request was blocked. May need additional cookies or WBI signature.")

            if data.get("code") != 0:
                raise ValueError(f"Search failed: {data.get('message', 'Unknown error')}")

            search_data = data.get("data", {})

            # Check if result is valid
            result = search_data.get("result", [])
            if isinstance(result, str):
                # If result is a string, the API might have returned an error message
                raise ValueError(f"Search API returned invalid result: {result}")

            return search_data

        except (httpx.HTTPStatusError, ValueError) as e:
            # If old API fails, log and re-raise
            logger.warning("Old search API failed, error: %s", str(e))
            raise


def format_search_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format a single search result item.

    Args:
        result: Raw result item from API

    Returns:
        Formatted result dictionary
    """
    result_type = result.get("type", "")

    if result_type == "video":
        return {
            "type": "video",
            "bvid": result.get("bvid", ""),
            "aid": result.get("aid", ""),
            "title": result.get("title", "").replace("<em class=\"keyword\">", "").replace("</em>", ""),
            "description": result.get("description", ""),
            "author": result.get("author", ""),
            "mid": result.get("mid", ""),
            "duration": result.get("duration", ""),
            "play": result.get("play", 0),  # 播放数
            "video_review": result.get("video_review", 0),  # 弹幕数
            "favorites": result.get("favorites", 0),  # 收藏数
            "review": result.get("review", 0),  # 评论数
            "pic": result.get("pic", ""),
            "arcurl": result.get("arcurl", ""),
            "pubdate": result.get("pubdate", 0),
            "tag": result.get("tag", "")
        }
    else:
        # For other types, return basic info
        return {
            "type": result_type,
            "id": result.get("id", ""),
            "title": result.get("title", "").replace("<em class=\"keyword\">", "").replace("</em>", ""),
            "description": result.get("description", ""),
            **{k: v for k, v in result.items() if k not in ["title", "description"]}
        }


def build_server(port: int) -> FastMCP:
    """
    Initializes the MCP server for Bilibili video operations.

    :param port: Port for SSE.
    :return: The MCP server.
    """
    mcp = FastMCP("bilibili-video-tool", port=port)
    logger = get_logger("bilibili-video-tool")

    @mcp.tool()
    async def search_video(keyword: str) -> str:
        """
        Search for videos and other content on Bilibili using comprehensive search.

        Args:
            keyword: The search keyword string.

        Returns:
            JSON string containing search results. Returns up to 20 results including videos,
            users, topics, and other content types.
        """
        try:
            logger.info("Searching Bilibili with keyword: %s", keyword)

            # Perform search
            search_data = await search_bilibili(keyword)

            # Debug: log the structure
            logger.debug("Search data keys: %s", list(search_data.keys()) if isinstance(search_data, dict) else "Not a dict")

            # Extract results - check the actual structure
            results = search_data.get("result", [])

            # Log for debugging
            logger.debug("Result type: %s, Result length: %s", type(results), len(results) if isinstance(results, list) else "N/A")

            # If result is not a list, handle different cases
            if not isinstance(results, list):
                if isinstance(results, str):
                    # If result is a string, it's likely an error or empty
                    logger.warning("Result is a string, might need WBI signature: %s", results[:100])
                    results = []
                elif isinstance(results, dict):
                    # Maybe result contains nested data
                    results = results.get("data", results.get("items", results.get("list", [])))
                    if not isinstance(results, list):
                        results = []
                else:
                    results = []

            # Format results
            # Note: Old API returns results in format: [{"result_type": "video", "data": {...}}, ...]
            formatted_results = []
            for result_item in results:
                if isinstance(result_item, dict):
                    # Check if it's the old API format with result_type and data
                    if "result_type" in result_item and "data" in result_item:
                        result_type = result_item.get("result_type", "")
                        result_data = result_item.get("data", {})

                        # Extract items from data if it's a list
                        if isinstance(result_data, list):
                            for item in result_data:
                                if isinstance(item, dict):
                                    # Add result_type to item for formatting
                                    item["type"] = result_type
                                    formatted_result = format_search_result(item)
                                    formatted_results.append(formatted_result)
                        elif isinstance(result_data, dict):
                            # Single item
                            result_data["type"] = result_type
                            formatted_result = format_search_result(result_data)
                            formatted_results.append(formatted_result)
                    else:
                        # Direct format (new API or different structure)
                        formatted_result = format_search_result(result_item)
                        formatted_results.append(formatted_result)

            # Extract summary information
            num_results = search_data.get("numResults", 0)
            top_tlist = search_data.get("top_tlist", {})

            # Log for debugging
            logger.info("Found %d formatted results, numResults: %d", len(formatted_results), num_results)

            # Build response
            response = {
                "status": "success",
                "keyword": keyword,
                "total_results": num_results,
                "result_count": len(formatted_results),
                "results": formatted_results,
                "summary": {
                    "video_count": top_tlist.get("video", 0),
                    "user_count": top_tlist.get("bili_user", 0),
                    "topic_count": top_tlist.get("topic", 0),
                    "live_count": top_tlist.get("live", 0),
                    "article_count": top_tlist.get("article", 0),
                    "bangumi_count": top_tlist.get("media_bangumi", 0),
                    "movie_count": top_tlist.get("media_ft", 0)
                }
            }

            return json.dumps(response, ensure_ascii=False, indent=2)

        except httpx.HTTPStatusError as e:
            logger.error("HTTP error searching Bilibili: %s", str(e))
            return json.dumps({
                "status": "error",
                "message": f"HTTP error: {e.response.status_code} - {e.response.text[:200]}"
            }, ensure_ascii=False)
        except (ValueError, KeyError) as e:
            logger.error("Error searching Bilibili: %s", str(e))
            return json.dumps({
                "status": "error",
                "message": f"Search failed: {str(e)}"
            }, ensure_ascii=False)
        except Exception as e:
            logger.error("Unexpected error searching Bilibili: %s", str(e), exc_info=True)
            return json.dumps({
                "status": "error",
                "message": f"Unexpected error: {str(e)}"
            }, ensure_ascii=False)

    @mcp.tool()
    async def download_video(video_id: str) -> str:
        """
        Download a video from Bilibili in 480P MP4 format.

        Args:
            video_id: The Bilibili video ID (BV number like BV1xx411c7mN or AV number like AV123456).

        Returns:
            JSON string containing download status and file path.

        Note:
            Output path is read from BILIBILI_FILE_PATH environment variable.
            If not set, saves to current directory with video title.
        """
        try:
            logger.info("Downloading video %s", video_id)

            # Parse video ID
            bvid, aid = parse_video_id(video_id)
            if not bvid and not aid:
                return json.dumps({
                    "status": "error",
                    "message": f"Invalid video ID format: {video_id}. Expected BVxxxxxx or AVxxxxxx"
                }, ensure_ascii=False)

            # Get SESSDATA from environment if available
            sessdata = os.environ.get("BILIBILI_SESSDATA")

            # Get video information
            video_info = await get_video_info(bvid=bvid, aid=aid, sessdata=sessdata)

            # Extract necessary information
            title = video_info.get("title", "video")
            # Get first page's cid (for multi-part videos, use first part)
            pages = video_info.get("pages", [])
            if not pages:
                return json.dumps({
                    "status": "error",
                    "message": "No video pages found"
                }, ensure_ascii=False)

            cid = pages[0].get("cid")
            if not cid:
                return json.dumps({
                    "status": "error",
                    "message": "Failed to get video cid"
                }, ensure_ascii=False)

            # Use bvid if available, otherwise construct from aid
            if not bvid:
                bvid = video_info.get("bvid")

            # Get video stream URL (480P MP4 format)
            # qn=32 for 480P, fnval=1 for MP4 format
            stream_info = await get_video_stream_url(
                bvid=bvid,
                cid=cid,
                qn=32,  # 480P
                fnval=1,  # MP4 format
                sessdata=sessdata
            )

            # Extract video URL from response
            durl = stream_info.get("durl", [])
            if not durl:
                return json.dumps({
                    "status": "error",
                    "message": "No video stream URL found in response"
                }, ensure_ascii=False)

            video_url = durl[0].get("url")
            if not video_url:
                return json.dumps({
                    "status": "error",
                    "message": "Video URL not found in stream response"
                }, ensure_ascii=False)

            # Get output path from environment variable
            output_path = os.environ.get("BILIBILI_FILE_PATH", "")

            # If not set, use default path with video title
            if not output_path:
                # Sanitize title for filename
                safe_title = re.sub(r'[^\w\s-]', '', title).strip()
                safe_title = re.sub(r'[-\s]+', '-', safe_title)
                output_path = f"{safe_title}_{bvid or aid}.mp4"
            else:
                # If BILIBILI_FILE_PATH is a directory, append filename
                output_dir = Path(output_path)
                if output_dir.is_dir() or (not output_path.endswith('.mp4') and not output_path.endswith('/') and not output_path.endswith('\\')):
                    # It's a directory or doesn't have extension, append filename
                    safe_title = re.sub(r'[^\w\s-]', '', title).strip()
                    safe_title = re.sub(r'[-\s]+', '-', safe_title)
                    output_path = str(Path(output_path) / f"{safe_title}_{bvid or aid}.mp4")

            # Ensure .mp4 extension
            if not output_path.endswith('.mp4'):
                output_path = f"{output_path}.mp4"

            # Download video
            logger.info("Downloading video from URL: %s", video_url)
            downloaded_path = await download_file(video_url, output_path)

            # Get file size
            file_size = Path(downloaded_path).stat().st_size

            return json.dumps({
                "status": "success",
                "video_id": video_id,
                "bvid": bvid,
                "title": title,
                "quality": "480p",
                "format": "mp4",
                "file_path": downloaded_path,
                "file_size": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2)
            }, ensure_ascii=False)

        except httpx.HTTPStatusError as e:
            logger.error("HTTP error downloading video: %s", str(e))
            return json.dumps({
                "status": "error",
                "message": f"HTTP error: {e.response.status_code} - {e.response.text[:200]}"
            }, ensure_ascii=False)
        except (ValueError, KeyError) as e:
            logger.error("Error downloading video: %s", str(e))
            return json.dumps({
                "status": "error",
                "message": f"Failed to download video: {str(e)}"
            }, ensure_ascii=False)
        except Exception as e:
            logger.error("Unexpected error downloading video: %s", str(e), exc_info=True)
            return json.dumps({
                "status": "error",
                "message": f"Unexpected error: {str(e)}"
            }, ensure_ascii=False)

    @mcp.tool()
    async def get_comments(video_id: str, page: int = 1, page_size: int = 20, sort: str = "hot") -> str:
        """
        Get comments for a Bilibili video.

        Args:
            video_id: The Bilibili video ID (BV number or AV number).
            page: Page number for pagination (default: 1).
            page_size: Number of comments per page (default: 20).
            sort: Sort order - "hot" for hot comments, "time" for time-based (default: "hot").

        Returns:
            JSON string containing comments data.
        """
        try:
            logger.info("Getting comments for video %s, page: %s, page_size: %s, sort: %s", video_id, page, page_size, sort)

            # Parse video ID to get aid
            bvid, aid = parse_video_id(video_id)

            # If we have bvid but not aid, get video info to get aid
            if bvid and not aid:
                video_info = await get_video_info(bvid=bvid)
                aid = video_info.get("aid")

            if not aid:
                return json.dumps({
                    "status": "error",
                    "message": f"Failed to get aid for video: {video_id}"
                }, ensure_ascii=False)

            # Get SESSDATA from environment if available
            sessdata = os.environ.get("BILIBILI_SESSDATA")

            # Get comments
            comments_data = await get_video_comments(
                aid=aid,
                page=page,
                page_size=page_size,
                sort=sort,
                sessdata=sessdata
            )

            # Format comments
            replies = comments_data.get("replies", [])
            formatted_comments = []

            for reply in replies:
                comment = {
                    "rpid": reply.get("rpid", 0),  # Comment ID
                    "mid": reply.get("mid", 0),  # User ID
                    "uname": reply.get("member", {}).get("uname", ""),  # Username
                    "message": reply.get("content", {}).get("message", ""),  # Comment content
                    "ctime": reply.get("ctime", 0),  # Comment time (timestamp)
                    "like": reply.get("like", 0),  # Like count
                    "rcount": reply.get("rcount", 0),  # Reply count
                    "level": reply.get("member", {}).get("level_info", {}).get("current_level", 0)  # User level
                }
                formatted_comments.append(comment)

            # Build response
            response = {
                "status": "success",
                "video_id": video_id,
                "aid": aid,
                "page": page,
                "page_size": page_size,
                "sort": sort,
                "total": comments_data.get("page", {}).get("count", 0),
                "total_replies": comments_data.get("page", {}).get("acount", 0),
                "comments": formatted_comments
            }

            return json.dumps(response, ensure_ascii=False, indent=2)

        except httpx.HTTPStatusError as e:
            logger.error("HTTP error getting comments: %s", str(e))
            return json.dumps({
                "status": "error",
                "message": f"HTTP error: {e.response.status_code} - {e.response.text[:200]}"
            }, ensure_ascii=False)
        except (ValueError, KeyError) as e:
            logger.error("Error getting comments: %s", str(e))
            return json.dumps({
                "status": "error",
                "message": f"Failed to get comments: {str(e)}"
            }, ensure_ascii=False)
        except Exception as e:
            logger.error("Unexpected error getting comments: %s", str(e), exc_info=True)
            return json.dumps({
                "status": "error",
                "message": f"Unexpected error: {str(e)}"
            }, ensure_ascii=False)

    @mcp.tool()
    async def get_subtitles(video_id: str, language: str = "zh-CN", subtitle_format: str = "srt") -> str:
        """
        Get subtitles/captions for a Bilibili video.

        Args:
            video_id: The Bilibili video ID (BV number or AV number).
            language: Subtitle language code (default: "zh-CN" for Chinese).
            subtitle_format: Subtitle format - "srt", "vtt", or "json" (default: "srt").

        Returns:
            JSON string containing subtitle data or subtitle file content.
        """
        try:
            logger.info("Getting subtitles for video %s, language: %s, format: %s", video_id, language, subtitle_format)

            # Parse video ID
            bvid, aid = parse_video_id(video_id)

            # Get video information to get cid
            video_info = await get_video_info(bvid=bvid, aid=aid)

            # Use bvid if available, otherwise get from video_info
            if not bvid:
                bvid = video_info.get("bvid")

            # Get first page's cid (for multi-part videos, use first part)
            pages = video_info.get("pages", [])
            if not pages:
                return json.dumps({
                    "status": "error",
                    "message": "No video pages found"
                }, ensure_ascii=False)

            cid = pages[0].get("cid")
            if not cid:
                return json.dumps({
                    "status": "error",
                    "message": "Failed to get video cid"
                }, ensure_ascii=False)

            # Get SESSDATA from environment if available
            sessdata = os.environ.get("BILIBILI_SESSDATA")

            # Get subtitles
            subtitles_data = await get_video_subtitles(
                bvid=bvid,
                cid=cid,
                sessdata=sessdata
            )

            # Extract subtitle information
            subtitle_info = subtitles_data.get("subtitle", {})
            subtitles_list = subtitle_info.get("subtitles", [])

            if not subtitles_list:
                return json.dumps({
                    "status": "success",
                    "video_id": video_id,
                    "bvid": bvid,
                    "cid": cid,
                    "message": "No subtitles available for this video",
                    "subtitles": []
                }, ensure_ascii=False)

            # Find subtitle by language (default to first one if not found)
            selected_subtitle = None
            for subtitle in subtitles_list:
                if subtitle.get("lan") == language or subtitle.get("lan_doc") == language:
                    selected_subtitle = subtitle
                    break

            if not selected_subtitle:
                # Use first subtitle if language not found
                selected_subtitle = subtitles_list[0]
                logger.warning("Language %s not found, using %s", language, selected_subtitle.get("lan_doc", "default"))

            # Download subtitle file
            subtitle_url = selected_subtitle.get("subtitle_url", "")
            if not subtitle_url:
                return json.dumps({
                    "status": "error",
                    "message": "Subtitle URL not found"
                }, ensure_ascii=False)

            # Download and parse subtitle
            subtitle_file_data = await download_subtitle_file(subtitle_url)

            # Convert to requested format
            if subtitle_format.lower() == "srt":
                subtitle_content = convert_subtitle_to_srt(subtitle_file_data)
            elif subtitle_format.lower() == "vtt":
                subtitle_content = convert_subtitle_to_vtt(subtitle_file_data)
            elif subtitle_format.lower() == "json":
                subtitle_content = json.dumps(subtitle_file_data, ensure_ascii=False, indent=2)
            else:
                # Default to JSON if format not recognized
                subtitle_content = json.dumps(subtitle_file_data, ensure_ascii=False, indent=2)

            # Build response
            response = {
                "status": "success",
                "video_id": video_id,
                "bvid": bvid,
                "cid": cid,
                "language": selected_subtitle.get("lan_doc", ""),
                "language_code": selected_subtitle.get("lan", ""),
                "format": subtitle_format.lower(),
                "available_languages": [
                    {
                        "code": sub.get("lan", ""),
                        "name": sub.get("lan_doc", "")
                    }
                    for sub in subtitles_list
                ],
                "subtitle_content": subtitle_content
            }

            return json.dumps(response, ensure_ascii=False, indent=2)

        except httpx.HTTPStatusError as e:
            logger.error("HTTP error getting subtitles: %s", str(e))
            return json.dumps({
                "status": "error",
                "message": f"HTTP error: {e.response.status_code} - {e.response.text[:200]}"
            }, ensure_ascii=False)
        except (ValueError, KeyError) as e:
            logger.error("Error getting subtitles: %s", str(e))
            return json.dumps({
                "status": "error",
                "message": f"Failed to get subtitles: {str(e)}"
            }, ensure_ascii=False)
        except Exception as e:
            logger.error("Unexpected error getting subtitles: %s", str(e), exc_info=True)
            return json.dumps({
                "status": "error",
                "message": f"Unexpected error: {str(e)}"
            }, ensure_ascii=False)

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
    logger = get_logger("Service:bilibili-video-tool")
    logger.info("Starting the MCP server")
    mcp = build_server(int(port))
    mcp.run(transport=transport.lower())


if __name__ == "__main__":
    main()
