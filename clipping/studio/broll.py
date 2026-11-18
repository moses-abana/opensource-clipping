import html
import importlib.util
import json
import math
import os
import random
import re
import shutil
import string
import subprocess
import textwrap
import time
import urllib.parse
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
import requests
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image, ImageDraw, ImageFont
from yt_dlp import YoutubeDL

def _load_studio_internal_module(file_name: str, module_alias: str):
    module_path = os.path.join(os.path.dirname(__file__), file_name)
    spec = importlib.util.spec_from_file_location(module_alias, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

utils = _load_studio_internal_module("utils.py", "clipping_studio_utils")
_resize_frame = utils._resize_frame
_is_vertical_ratio = utils._is_vertical_ratio

FIREFOX_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:148.0) Gecko/20100101 Firefox/148.0"


_helpers = _load_studio_internal_module("helpers.py", "clipping_studio_helpers")
_ffmpeg_utils = _load_studio_internal_module("ffmpeg_utils.py", "clipping_studio_ffmpeg_utils")
format_seconds = _helpers.format_seconds
escape_ffmpeg_filter_value = _helpers.escape_ffmpeg_filter_value
detect_video_encoder = _ffmpeg_utils.detect_video_encoder
get_ts_encode_args = _ffmpeg_utils.get_ts_encode_args
get_mp4_encode_args = _ffmpeg_utils.get_mp4_encode_args
open_ffmpeg_video_writer = _ffmpeg_utils.open_ffmpeg_video_writer
build_ffmpeg_progress_cmd = _ffmpeg_utils.build_ffmpeg_progress_cmd
run_ffmpeg_with_progress = _ffmpeg_utils.run_ffmpeg_with_progress

USED_PIXABAY_IDS = set()
PIXABAY_SEARCH_CACHE = {}


def download_pixabay_broll(query, rasio, output_filename, pixabay_api_key):
    """
    Search and download one Pixabay B-roll video clip matching the query and aspect ratio.

    Args:
        query (str): Search query term (e.g., 'nature', 'technology').
        rasio (str): Target aspect ratio string (`9:16` for portrait or `16:9` for landscape).
        output_filename (str): Local file path where the downloaded MP4 will be saved.
        pixabay_api_key (str): Valid Pixabay API key.

    Returns:
        bool: True if the video was successfully downloaded and saved, False otherwise.

    Side Effects:
        Makes HTTP GET requests to the Pixabay API and video CDN.
        Mutates the global `USED_PIXABAY_IDS` set to prevent duplicate downloads.
        Writes a temporary file (`.part`) and renames it upon successful download.
        Prints status and error messages to stdout.

    Raises:
        None explicitly. Exceptions during download or API calls are caught and return False.
    """
    global USED_PIXABAY_IDS

    if not pixabay_api_key:
        print("   ⚠️ PIXABAY_API_KEY was not found. Skipping B-roll.")
        return False

    params = urllib.parse.urlencode(
        {
            "key": pixabay_api_key,
            "q": query,
            "per_page": 30,
            "safesearch": "true",
        }
    )
    search_url = f"https://pixabay.com/api/videos/?{params}"
    req = urllib.request.Request(search_url, headers={"User-Agent": "Mozilla/5.0"})

    cached = PIXABAY_SEARCH_CACHE.get(query)
    if cached and time.monotonic() - cached[0] < 24 * 60 * 60:
        data = cached[1]
    else:
        try:
            with urllib.request.urlopen(req) as response:
                data = json.load(response)
            PIXABAY_SEARCH_CACHE[query] = (time.monotonic(), data)
        except Exception as e:
            # The API key is a query parameter; avoid logging exceptions that may echo the URL.
            print(f"   ⚠️ Error API Pixabay saat mencari '{query}': {type(e).__name__}")
            return False

    videos = data.get("hits", [])
    if not videos:
        print(f"   ⚠️ Pixabay found no videos for '{query}'.")
        return False

    vertical = _is_vertical_ratio(rasio)
    available_videos = [
        video for video in videos
        if video.get("id") not in USED_PIXABAY_IDS
        and any(
            isinstance(variant, dict)
            and variant.get("url")
            and bool(variant.get("width", 0) > variant.get("height", 0)) != vertical
            for variant in video.get("videos", {}).values()
        )
    ]
    if not available_videos:
        available_videos = [
            video for video in videos
            if any(
                isinstance(variant, dict) and variant.get("url")
                for variant in video.get("videos", {}).values()
            )
        ]
    if not available_videos:
        print(f"   ⚠️ Pixabay returned no downloadable video files for '{query}'.")
        return False

    video_data = random.choice(available_videos)
    USED_PIXABAY_IDS.add(video_data.get("id"))
    video_variants = video_data.get("videos", {})
    matching_variants = [
        video_variants.get(size, {})
        for size in ("large", "medium", "small", "tiny")
        if video_variants.get(size, {}).get("url")
        and bool(video_variants[size].get("width", 0) > video_variants[size].get("height", 0)) != vertical
    ]
    all_variants = [
        video_variants.get(size, {})
        for size in ("large", "medium", "small", "tiny")
        if video_variants.get(size, {}).get("url")
    ]
    download_url = (matching_variants or all_variants or [{}])[0].get("url")
    if not download_url:
        return False
    download_req = urllib.request.Request(
        download_url, headers={"User-Agent": "Mozilla/5.0"}
    )

    try:
        temp_path = output_filename + ".part"
        with (
            urllib.request.urlopen(download_req) as response,
            open(temp_path, "wb") as f,
        ):
            shutil.copyfileobj(response, f)
        os.replace(temp_path, output_filename)
        return True
    except Exception as e:
        print(f"   ⚠️ Error downloading B-roll for '{query}': {e}")
        return False


# Kept as a compatibility alias for scripts importing the old public helper.
USED_PEXELS_IDS = USED_PIXABAY_IDS
download_pexels_broll = download_pixabay_broll


def crop_center_broll(img, target_w, target_h):
    """
    Center-crop an image frame to the exact target aspect ratio, then resize it.

    Args:
        img (np.ndarray): Input image frame array (from OpenCV).
        target_w (int): Desired output width in pixels.
        target_h (int): Desired output height in pixels.

    Returns:
        np.ndarray: The cropped and resized frame.

    Side Effects:
        None.

    Raises:
        cv2.error: If the input image format is invalid or resizing fails.
    """
    h, w = img.shape[:2]
    target_ratio = target_w / target_h
    img_ratio = w / h

    if img_ratio > target_ratio:
        new_w = int(h * target_ratio)
        x = (w - new_w) // 2
        img = img[:, x : x + new_w]
    elif img_ratio < target_ratio:
        new_h = int(w / target_ratio)
        y = (h - new_h) // 2
        img = img[y : y + new_h, :]

    return _resize_frame(img, (target_w, target_h))


