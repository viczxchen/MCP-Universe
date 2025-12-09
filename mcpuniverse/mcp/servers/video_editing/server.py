"""
An MCP server for basic video editing operations using OpenCV.

Tools:
- get_info: Get video duration and frame count.
- crop_video: Crop a video segment (video only, no audio).
- get_frame: Extract one or multiple frames as image files.
"""
import os
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import click
import cv2  # type: ignore
from mcp.server.fastmcp import FastMCP
from mcpuniverse.common.logger import get_logger


def _get_output_dir() -> Path:
    """
    Get the base output directory for generated video/image files.

    Uses VIDEO_EDITING_OUTPUT_DIR env var if set, otherwise current working directory.
    """
    base = os.environ.get("VIDEO_EDITING_OUTPUT_DIR", os.getcwd())
    path = Path(base).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_video_capture(path: str) -> cv2.VideoCapture:
    """Helper to create VideoCapture and validate path."""
    cap = cv2.VideoCapture(path)
    return cap


def _get_video_properties(cap: cv2.VideoCapture) -> Dict[str, Any]:
    """Extract common video properties from an open VideoCapture."""
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0

    duration = 0.0
    if fps > 0:
        duration = frame_count / fps

    return {
        "frame_count": frame_count,
        "fps": fps,
        "duration_seconds": duration,
        "width": width,
        "height": height,
    }


def build_server(port: int) -> FastMCP:
    """
    Initializes the MCP server for video editing operations.

    :param port: Port for SSE.
    :return: The MCP server.
    """
    mcp = FastMCP("video-editing", port=port)
    logger = get_logger("video-editing")

    @mcp.tool()
    async def get_info(video_path: str) -> str:
        """
        Get basic information about a video file.

        Args:
            video_path: Absolute or relative path to the video file.

        Returns:
            JSON string containing duration (seconds), frame count, fps, width, height.
        """
        try:
            path = Path(video_path).expanduser()
            if not path.exists():
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"Video file not found: {path}",
                    },
                    ensure_ascii=False,
                )

            cap = _get_video_capture(str(path))
            if not cap.isOpened():
                cap.release()
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"Failed to open video file: {path}",
                    },
                    ensure_ascii=False,
                )

            props = _get_video_properties(cap)
            cap.release()

            result = {
                "status": "success",
                "video_path": str(path),
                **props,
            }
            return json.dumps(result, ensure_ascii=False, indent=2)

        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("Error getting video info: %s", str(exc), exc_info=True)
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Failed to get video info: {str(exc)}",
                },
                ensure_ascii=False,
            )

    @mcp.tool()
    async def crop_video(
        video_path: str,
        start_time: float,
        end_time: float,
    ) -> str:
        """
        Crop a segment from a video and save it as a new file (video only, no audio).

        Args:
            video_path: Path to the source video file.
            start_time: Start time in seconds.
            end_time: End time in seconds (must be greater than start_time).

        Returns:
            JSON string containing status and new file path.
        """
        try:
            src_path = Path(video_path).expanduser()
            if not src_path.exists():
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"Video file not found: {src_path}",
                    },
                    ensure_ascii=False,
                )

            if end_time <= start_time:
                return json.dumps(
                    {
                        "status": "error",
                        "message": "end_time must be greater than start_time",
                    },
                    ensure_ascii=False,
                )

            cap = _get_video_capture(str(src_path))
            if not cap.isOpened():
                cap.release()
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"Failed to open video file: {src_path}",
                    },
                    ensure_ascii=False,
                )

            props = _get_video_properties(cap)
            fps = props["fps"]
            frame_count = props["frame_count"]
            width = props["width"]
            height = props["height"]

            if fps <= 0 or frame_count <= 0:
                cap.release()
                return json.dumps(
                    {
                        "status": "error",
                        "message": "Invalid video FPS or frame count",
                    },
                    ensure_ascii=False,
                )

            total_duration = props["duration_seconds"]
            # Clamp times
            start = max(0.0, start_time)
            end = min(total_duration, end_time) if total_duration > 0 else end_time

            if end <= start:
                cap.release()
                return json.dumps(
                    {
                        "status": "error",
                        "message": "Invalid crop range after clamping to video duration",
                    },
                    ensure_ascii=False,
                )

            start_frame = int(math.floor(start * fps))
            end_frame = int(math.floor(end * fps))
            start_frame = max(0, min(start_frame, frame_count - 1))
            end_frame = max(start_frame + 1, min(end_frame, frame_count))

            # Prepare output file
            output_dir = _get_output_dir()
            safe_stem = src_path.stem.replace(" ", "_")
            output_filename = f"{safe_stem}_crop_{start_frame}_{end_frame}.mp4"
            output_path = output_dir / output_filename

            # Video writer (mp4v codec)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            current_frame = start_frame
            while current_frame < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                current_frame += 1

            cap.release()
            out.release()

            return json.dumps(
                {
                    "status": "success",
                    "video_path": str(src_path),
                    "output_path": str(output_path),
                    "start_time": start,
                    "end_time": end,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "fps": fps,
                },
                ensure_ascii=False,
                indent=2,
            )

        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("Error cropping video: %s", str(exc), exc_info=True)
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Failed to crop video: {str(exc)}",
                },
                ensure_ascii=False,
            )

    @mcp.tool()
    async def get_frame(
        video_path: str,
        times: str = "",
        time: float = 0.0,
    ) -> str:
        """
        Extract one or multiple frames from a video as image files.

        Args:
            video_path: Path to the source video file.
            times: Optional comma-separated list of timestamps in seconds,
                   e.g. "0, 1.5, 3". If provided, `time` is ignored.
            time: Single timestamp in seconds (used when `times` is empty).

        Returns:
            JSON string containing status and list of extracted frame paths.
        """
        try:
            src_path = Path(video_path).expanduser()
            if not src_path.exists():
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"Video file not found: {src_path}",
                    },
                    ensure_ascii=False,
                )

            cap = _get_video_capture(str(src_path))
            if not cap.isOpened():
                cap.release()
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"Failed to open video file: {src_path}",
                    },
                    ensure_ascii=False,
                )

            props = _get_video_properties(cap)
            fps = props["fps"]
            frame_count = props["frame_count"]

            if fps <= 0 or frame_count <= 0:
                cap.release()
                return json.dumps(
                    {
                        "status": "error",
                        "message": "Invalid video FPS or frame count",
                    },
                    ensure_ascii=False,
                )

            # Parse times
            timestamps: List[float] = []
            if times.strip():
                for part in times.split(","):
                    try:
                        ts = float(part.strip())
                        if ts >= 0:
                            timestamps.append(ts)
                    except ValueError:
                        continue
            else:
                if time >= 0:
                    timestamps.append(time)

            if not timestamps:
                cap.release()
                return json.dumps(
                    {
                        "status": "error",
                        "message": "No valid timestamps provided",
                    },
                    ensure_ascii=False,
                )

            output_dir = _get_output_dir()
            safe_stem = src_path.stem.replace(" ", "_")

            frame_paths: List[str] = []

            for idx, ts in enumerate(timestamps):
                frame_index = int(math.floor(ts * fps))
                frame_index = max(0, min(frame_index, frame_count - 1))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                if not ret:
                    continue

                # Build filename; include index to avoid collisions
                if len(timestamps) == 1:
                    suffix = f"_frame_{frame_index}"
                else:
                    suffix = f"_frame_{idx}_{frame_index}"
                output_path = output_dir / f"{safe_stem}{suffix}.png"

                cv2.imwrite(str(output_path), frame)
                frame_paths.append(str(output_path))

            cap.release()

            if not frame_paths:
                return json.dumps(
                    {
                        "status": "error",
                        "message": "Failed to extract any frames",
                    },
                    ensure_ascii=False,
                )

            return json.dumps(
                {
                    "status": "success",
                    "video_path": str(src_path),
                    "fps": fps,
                    "frame_count": frame_count,
                    "timestamps": timestamps,
                    "frames": frame_paths,
                },
                ensure_ascii=False,
                indent=2,
            )

        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("Error extracting frames: %s", str(exc), exc_info=True)
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Failed to extract frames: {str(exc)}",
                },
                ensure_ascii=False,
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
    assert transport.lower() in ["stdio", "sse"], "Transport should be `stdio` or `sse`"
    logger = get_logger("Service:video-editing")
    logger.info("Starting the MCP server")
    mcp = build_server(int(port))
    mcp.run(transport=transport.lower())


if __name__ == "__main__":
    main()
