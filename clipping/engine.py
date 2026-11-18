"""
clipping.engine — Download, Transcription & Gemini AI Analysis

Maps to Cell 2 (The Engine) of the notebook.
"""

import json
import os
import re
import time

from yt_dlp import YoutubeDL
from faster_whisper import WhisperModel


# ==============================================================================
# TAHAP 1: DOWNLOAD VIDEO
# ==============================================================================

def _build_ydl_format_selector(download_source_height: str | int) -> str:
    """
    Build a yt-dlp format selector string for source-quality preference.
    """
    # Skip AV1 codec as it lacks HW acceleration on many platforms (e.g., Colab T4)
    # and causes decoding failures in OpenCV/FFmpeg software fallbacks.
    # Note: Using [vcodec!*=av01] to safely ensure it does not contain 'av01' anywhere.
    codec_filter = "[vcodec!*=av01]"

    if download_source_height == "max":
        return f"bestvideo{codec_filter}+bestaudio/best{codec_filter}"

    try:
        h_val = int(download_source_height)
    except (ValueError, TypeError):
        h_val = 0

    if 0 < h_val <= 1080:
        # For standard resolutions, strictly prefer native MP4 (H.264/AAC), ensuring no AV1 in mp4
        return (
            f"bestvideo[height<=?{h_val}][ext=mp4]{codec_filter}+bestaudio[ext=m4a]/"
            f"bestvideo[height<=?{h_val}]{codec_filter}+bestaudio/"
            f"best[height<=?{h_val}][ext=mp4]{codec_filter}/"
            f"best[height<=?{h_val}]{codec_filter}"
        )

    return (
        f"bestvideo[height<=?{download_source_height}]{codec_filter}+bestaudio/"
        f"best[height<=?{download_source_height}]{codec_filter}"
    )


_PLATFORM_LABELS = {
    "youtube": "YouTube",
    "tiktok": "TikTok",
    "instagram": "Instagram",
    "gdrive": "Google Drive",
}


def _extract_gdrive_file_id(url: str) -> str | None:
    """Extract the Google Drive file ID from various URL formats."""
    import re as _re
    m = _re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    m = _re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    return None


def _download_gdrive(url: str, output_path: str) -> None:
    """Download a video from Google Drive using gdown (more reliable than yt-dlp)."""
    import gdown

    file_id = _extract_gdrive_file_id(url)
    if not file_id:
        raise RuntimeError(
            f"Could not extract a file ID from the Google Drive URL: {url}\n"
            "      Supported formats:\n"
            "        • https://drive.google.com/file/d/FILE_ID/view\n"
            "        • https://drive.google.com/open?id=FILE_ID"
        )

    download_url = f"https://drive.google.com/uc?id={file_id}"
    print(f"      📥 File ID: {file_id}")
    gdown.download(download_url, output_path, quiet=False)


def _ydl_progress_hook(d: dict) -> None:
    """Render one download progress bar line from yt-dlp hook data.

    yt-dlp downloads video and audio streams separately, so this hook is called
    for each one. The newline on "finished" keeps each bar on its own line.
    """
    status = d.get("status")
    if status == "downloading":
        total = d.get("total_bytes") or d.get("total_bytes_estimate")
        downloaded = d.get("downloaded_bytes", 0)
        speed = d.get("speed")
        eta = d.get("eta")
        spd = f"{speed / 1024 / 1024:4.1f}MB/s" if speed else "  --MB/s"
        eta_s = f"{eta:>3}s" if eta is not None else " --s"
        if total:
            pct = downloaded / total * 100
            filled = int(20 * downloaded / total)
            bar = "█" * filled + " " * (20 - filled)
            print(
                f"\r      Download: {pct:3.0f}%|{bar}| "
                f"{downloaded / 1048576:.0f}/{total / 1048576:.0f}MB {spd} ETA {eta_s}   ",
                end="", flush=True,
            )
        else:
            # If the size is unknown (live/streamed manifest), show bytes and speed only.
            print(
                f"\r      Download: {downloaded / 1048576:.0f}MB {spd}   ",
                end="", flush=True,
            )
    elif status == "finished":
        print(flush=True)  # Close this stream's progress line.


def _apply_cookies(opts: dict, cookies: str | None) -> None:
    """Apply cookies to yt-dlp options dictionary."""
    if not cookies:
        return
    if cookies.endswith(".txt") or os.path.isfile(cookies):
        opts["cookiefile"] = cookies
    else:
        # Browser format: (browser, profile, keyring, container)
        opts["cookiesfrombrowser"] = (cookies, None, None, None)


def download_video(
    url: str,
    output_path: str,
    use_dlp_subs: bool = False,
    download_source_height: str | int = "max",
    source_platform: str = "youtube",
    yt_cookies: str | None = None,
) -> None:
    """
    Download a video to *output_path* with configurable source height.

    Parameters
    ----------
    source_platform : str
        One of ``"youtube"`` (default), ``"tiktok"``, ``"instagram"``,
        or ``"gdrive"``.
    """
    platform_label = _PLATFORM_LABELS.get(source_platform, source_platform)
    uses_youtube_format = source_platform == "youtube"

    print(f"[1/3] Downloading video from {platform_label}...")
    if download_source_height == "max":
        print("      🎯 Source quality: highest available", flush=True)
    else:
        print(f"      🎯 Source quality: up to {download_source_height}p", flush=True)

    # --- Google Drive: use gdown instead of yt-dlp ---
    if source_platform == "gdrive":
        _download_gdrive(url, output_path)
        if not os.path.exists(output_path):
            raise RuntimeError(
                f"❌ Google Drive download failed — file not found at {output_path}"
            )
        print("      ✅ Video downloaded from Google Drive.", flush=True)
        return

    # --- Build yt-dlp options per platform ---
    if uses_youtube_format:
        # YouTube: complex format selector + AV1 filter + remote components
        ydl_opts = {
            "format": _build_ydl_format_selector(download_source_height),
            "outtmpl": output_path,
            "quiet": True,
            "merge_output_format": "mp4",
            "remote_components": ["ejs:github"],
            "progress_hooks": [_ydl_progress_hook],
            "extractor_args": {"youtube": ["player_client=android,web"]},
        }
    else:
        # TikTok / Instagram: ensure video and audio are merged
        # We explicitly prefer H.264 over H.265 (TikTok's bytevc1) to prevent 
        # PyAV/faster-whisper from crashing with IndexError on Kaggle/Colab.
        ydl_opts = {
            "format": "bestvideo[vcodec^=h264]+bestaudio/best[vcodec^=h264]/best",
            "outtmpl": output_path,
            "quiet": True,
            "merge_output_format": "mp4",
            "progress_hooks": [_ydl_progress_hook],
        }

    _apply_cookies(ydl_opts, yt_cookies)

    # --- Subtitle download — only supported for YouTube ---
    if use_dlp_subs and uses_youtube_format:
        print("      Searching for automatic subtitles (en / id)...")
        import glob

        for lang in ["en", "id"]:
            ydl_opts_subs = ydl_opts.copy()
            ydl_opts_subs.update({
                "writesubtitles": True,
                "writeautomaticsub": True,
                "subtitleslangs": [lang],
                "subtitlesformat": "json3",
                "skip_download": True,  # Only download subtitles.
            })

            try:
                with YoutubeDL(ydl_opts_subs) as ydl:
                    ydl.download([url])

                # Check whether the JSON3 file for this language was actually downloaded.
                if glob.glob(output_path.replace(".mp4", f".*.json3")):
                    print(f"      ✅ Found '{lang}' subtitles. Continuing to video download...")
                    break
            except Exception as e:
                print(f"      ⚠️ Failed to download '{lang}' subtitles ({e}). Trying the next option...")
    elif use_dlp_subs and not uses_youtube_format:
        print(f"      ℹ️ {platform_label} does not provide automatic subtitles. Whisper will be used.")

    # Download the video separately from subtitle handling.
    with YoutubeDL(ydl_opts) as ydl:
        # Extra step to verify resolution before downloading
        try:
            info = ydl.extract_info(url, download=False)
            best_h = info.get("height", "unknown")
            v_codec = info.get("vcodec", "unknown")
            print(f"      ✅ Downloading: {best_h}p (Codec: {v_codec})", flush=True)
        except Exception as e:
            print(f"      ⚠️ Failed to check video details: {e}", flush=True)

        ydl.download([url])

    # --- Post-download verification ---
    if not os.path.exists(output_path):
        raise RuntimeError(
            f"❌ Download from {platform_label} failed — video file not found at {output_path}.\n"
            "      Make sure the URL is valid and publicly accessible."
        )


# ==============================================================================
# TAHAP 2: TRANSKRIPSI WHISPER & JSON3 FALLBACK
# ==============================================================================

def parse_youtube_json3_subs(json_path: str, max_words_per_subtitle: int = 5) -> tuple[str, list[dict]]:
    """
    Parse downloaded YouTube JSON3 subtitles into transkrip_lengkap and data_segmen.
    Returns empty string/list if parsing fails.
    """
    import json

    print("[2/3] Processing YouTube JSON3 subtitles...")
    transkrip_lengkap = ""
    data_segmen = []

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            subs_data = json.load(f)

        events = subs_data.get("events", [])

        flat_words = []
        for event in events:
            # YouTube timestamps are in ms
            t_start = event.get("tStartMs", 0) / 1000.0
            d_duration = event.get("dDurationMs", 0) / 1000.0
            event_end = t_start + d_duration

            segs = event.get("segs", [])
            for i, seg in enumerate(segs):
                text = seg.get("utf8", "")
                if not text.strip() or text == "\n":
                    continue

                # tOffsetMs is offset from t_start
                offset = seg.get("tOffsetMs", 0) / 1000.0
                seg_start = t_start + offset

                # Determine end of this segment
                if i < len(segs) - 1:
                    next_offset = segs[i + 1].get("tOffsetMs", 0) / 1000.0
                    seg_end = t_start + next_offset
                else:
                    seg_end = event_end

                if seg_end <= seg_start:
                    seg_end = seg_start + 1.0  # Fallback duration

                # Clean up YouTube subtitle artifacts
                clean_text = text.replace("\n", " ").replace("\u200b", "").strip()
                # Remove HTML tags (e.g., <i>, </i>, <b>, </b>, <font color="...">)
                clean_text = re.sub(r"<[^>]+>", "", clean_text)
                # Remove YouTube annotation brackets: [Music], [Applause], [Laughter], etc.
                clean_text = re.sub(r"\[[\w\s]+\]", "", clean_text)
                # Remove speaker change markers: >> 
                clean_text = re.sub(r">>\s*", "", clean_text)
                # Remove music symbols: ♪, ♫, etc.
                clean_text = re.sub(r"[♪♫♬♩]", "", clean_text)
                # Remove leading dashes often used for speaker identification
                clean_text = re.sub(r"^\s*-\s+", "", clean_text)
                # Collapse multiple spaces into one
                clean_text = re.sub(r"\s{2,}", " ", clean_text).strip()

                if clean_text:
                    # Split the text into individual words so word-by-word karaoke matches Whisper.
                    words_in_seg = clean_text.split()
                    if not words_in_seg:
                        continue

                    duration_per_word = (seg_end - seg_start) / len(words_in_seg)

                    for w_idx, w_text in enumerate(words_in_seg):
                        w_start = seg_start + (w_idx * duration_per_word)
                        w_end = w_start + duration_per_word

                        flat_words.append({
                            "word": w_text,
                            "start": w_start,
                            "end": w_end,
                        })

        # Adjust end times based on the start time of the next word to prevent overlaps
        for i in range(len(flat_words) - 1):
            if flat_words[i]["end"] > flat_words[i + 1]["start"]:
                flat_words[i]["end"] = max(flat_words[i]["start"] + 0.1, flat_words[i + 1]["start"])

        # Group them into segments
        chunk_words = []
        chunk_start = 0.0

        for i, w in enumerate(flat_words):
            if len(chunk_words) == 0:
                chunk_start = w["start"]

            chunk_words.append(w)

            if len(chunk_words) == max_words_per_subtitle or i == len(flat_words) - 1:
                chunk_text = " ".join([cw["word"] for cw in chunk_words])
                chunk_end = w["end"]
                transkrip_lengkap += f"[{chunk_start:.1f} - {chunk_end:.1f}] {chunk_text}\n"

                data_segmen.append({
                    "start": chunk_start,
                    "end": chunk_end,
                    "words": chunk_words,
                })
                chunk_words = []

        return transkrip_lengkap, data_segmen

    except Exception as e:
        print(f"⚠️ Failed to parse JSON3: {e}")
        return "", []


def transcribe_video(
    video_path: str,
    max_words_per_subtitle: int = 5,
    model_size: str = "large-v3",
    device: str = "cuda",
    compute_type: str = "float16",
    whisper_model_instance=None,
) -> tuple[str, list[dict]]:
    """
    Transcribe *video_path* using Faster-Whisper.

    Parameters
    ----------
    whisper_model_instance : WhisperModel or None
        Optional pre-loaded WhisperModel instance.  When provided the
        *model_size*, *device* and *compute_type* arguments are ignored
        and the given model is reused directly.  This avoids expensive
        re-loading when processing multiple videos in batch mode.

    Returns
    -------
    transkrip_lengkap : str
        Human-readable transcript with timestamps.
    data_segmen : list[dict]
        Word-level segments grouped by *max_words_per_subtitle*.
    """
    print("[2/3] Starting transcription with Faster-Whisper (word-level)...")

    # Faster-Whisper produces no output before the first segment, so announce each
    # phase; otherwise, the first CPU run (model download and full audio decode)
    # can look like it has hung.
    if whisper_model_instance is not None:
        model = whisper_model_instance
        print(
            "      ♻️ Reusing the loaded Whisper model (batch mode)",
            flush=True,
        )
    else:
        print(
            f"      ⏳ Memuat model Whisper '{model_size}' ({device})"
            " — the first download may take a while...",
            flush=True,
        )
        model = WhisperModel(model_size, device=device, compute_type=compute_type)

    print("      ⏳ Decoding audio and extracting features (no output yet)...", flush=True)
    segments, info = model.transcribe(video_path, beam_size=5, word_timestamps=True)

    transkrip_lengkap = ""
    data_segmen: list[dict] = []

    # Progress bar berdasarkan timestamp audio. faster-whisper men-stream segmen
    # Streams lazily, so advance the progress bar to each segment's end time as it arrives.
    from tqdm import tqdm

    total_dur = round(info.duration, 2)
    progress = tqdm(
        total=total_dur,
        unit="s",
        desc="      Transcription",
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:.0f}/{total:.0f}s [{elapsed}<{remaining}]",
    )

    for segment in segments:
        # Clamp floating-point drift so it cannot overshoot the duration.
        progress.update(min(segment.end, total_dur) - progress.n)
        transkrip_lengkap += f"[{segment.start:.1f} - {segment.end:.1f}] {segment.text}\n"

        if segment.words:
            chunk_words: list[dict] = []
            chunk_start = 0.0

            for i, w in enumerate(segment.words):
                if len(chunk_words) == 0:
                    chunk_start = w.start

                chunk_words.append({
                    "word": w.word.strip(),
                    "start": w.start,
                    "end": w.end,
                })

                if len(chunk_words) == max_words_per_subtitle or i == len(segment.words) - 1:
                    data_segmen.append({
                        "start": chunk_start,
                        "end": w.end,
                        "words": chunk_words,
                    })
                    chunk_words = []

    progress.update(total_dur - progress.n)  # Snap to 100% when complete.
    progress.close()
    return transkrip_lengkap, data_segmen


# ==============================================================================
# TAHAP 3: ANALISIS GEMINI AI
# ==============================================================================

TARGET_ACCOUNTS = {
    "Business": {
        "akun_tujuan": "Business.Mereska",
        "angle_desc": "Use this account for business, brands, revenue, sales, founders, marketing, or small businesses.",
        "bio": "Business insights, founder stories, and local brands. Business | Founder | Finance | Beauty | Marketing"
    },
    "Life": {
        "akun_tujuan": "Life.Mereska",
        "angle_desc": "Use this account for personal life, lifestyle, skincare, careers, mindset, relationships, personal finance, or self-growth.",
        "bio": "Insights for a better life and mindset. Podcast | Career | Finance | Beauty | Self Growth"
    },
    "Creator": {
        "akun_tujuan": "Creator.Mereska",
        "angle_desc": "Use this account for digital content, AI, affiliate marketing, tools, clipping, monetization, or earning from content.",
        "bio": "Turning digital content into income. AI | Affiliate | Clips | Tools | Monetization"
    },
    "Muslim": {
        "akun_tujuan": "Muslim.Mereska",
        "angle_desc": "Use this account for faith, provision, prayer, worship, work in service of Allah, Muslim family life, or business guided by Islamic values.",
        "bio": "Reminders about work, provision, and an Islamic way of life. Islam | Family | Work | Business"
    }
}

def _build_account_classification_prompt() -> str:
    lines = []
    for tipe, data in TARGET_ACCOUNTS.items():
        lines.append(f"- {tipe}: {data['angle_desc']}")
        lines.append(f"  (akun_tujuan: \"{data['akun_tujuan']}\", bio: \"{data['bio']}\")")
    return "\n".join(lines)


# ---- Retry Config ----
MAX_ATTEMPTS = 10
INITIAL_WAIT_SECONDS = 60
WAIT_INCREMENT_SECONDS = 30
REQUEST_TIMEOUT_MS = 15 * 60 * 1000  # 15 minutes
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def _extract_status_code(exc: Exception):
    for attr in ("status_code", "code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)

    match = re.search(r"\b(408|429|500|502|503|504)\b", str(exc))
    return int(match.group(1)) if match else None


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, json.JSONDecodeError):
        return True

    code = _extract_status_code(exc)
    if code in RETRYABLE_STATUS_CODES:
        return True

    msg = str(exc).lower()
    keywords = (
        "timeout", "temporarily unavailable", "deadline",
        "connection reset", "connection aborted", "service unavailable",
    )
    return any(k in msg for k in keywords)


def _generate_json_with_retry(client, model, fallback_model, contents, config):
    last_exc = None
    status_code = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            print(f"[Gemini] Attempt {attempt}/{MAX_ATTEMPTS}...")

            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )

            text = getattr(response, "text", None)
            if not text or not text.strip():
                raise ValueError("Gemini returned an empty response.text.")

            return json.loads(text)

        except Exception as exc:
            last_exc = exc
            status_code = _extract_status_code(exc)
            retryable = _is_retryable(exc)

            print(
                f"[Gemini] Attempt {attempt}/{MAX_ATTEMPTS} failed | "
                f"status={status_code} | error={exc}"
            )

            if (not retryable) or attempt == MAX_ATTEMPTS:
                break

            wait_seconds = INITIAL_WAIT_SECONDS + ((attempt - 1) * WAIT_INCREMENT_SECONDS)
            print(f"[Gemini] Retrying in {wait_seconds} seconds...")
            time.sleep(wait_seconds)

    print(f"[Gemini] The primary model attempt ({model}) failed.")
    if fallback_model:
        print(f"[Gemini] Retrying once with fallback model ({fallback_model})...")
        try:
            response = client.models.generate_content(
                model=fallback_model,
                contents=contents,
                config=config,
            )
            text = getattr(response, "text", None)
            if not text or not text.strip():
                raise ValueError("The Gemini fallback returned an empty response.text.")

            return json.loads(text)
        except Exception as exc_fallback:
            print(f"[Gemini] Fallback model failed | error={exc_fallback}")
            raise RuntimeError(
                f"Both the primary and fallback Gemini calls failed. "
                f"Primary status={status_code}, error={last_exc} | "
                f"Laporan Fallback error={exc_fallback}"
            ) from exc_fallback

    raise RuntimeError(
        f"Gemini call failed after {MAX_ATTEMPTS} attempts. Last error: {last_exc}"
    ) from last_exc


# ==== KONFIGURASI DURASI KLIP ====
# Change these values to adjust the clip duration limits (in seconds).
MIN_CLIP_DURATION = 20
MAX_CLIP_DURATION = 179

def get_analysis_prompt(transkrip_lengkap: str, jumlah_clip: int, durasi_hook: int, cfg=None) -> str:
    """Centralized prompt for both Gemini and NVIDIA providers."""
    # Build optional Hook V2 prompt section
    _hook_v2_prompt = ""
    if cfg and getattr(cfg, "hook_v2", False):
        _hook_v2_items = getattr(cfg, "hook_v2_items", 3)
        _hook_v2_style = getattr(cfg, "hook_v2_style", "controversial_fast_glitch")
        _hook_v2_prompt = f"""

HOOK V2 (MULTI-HOOK INTRO — REQUIRED):
- In addition to the standard hook, create "hook_v2" with {_hook_v2_items} short (0.5–2 second) segments taken from the clip's most striking, controversial, or emotional moments.
- Style: {_hook_v2_style}
- Each item must include start_time, end_time, and text (2–5 short on-screen words).
- Order items from strongest to weakest. The system adds transitions (white flash or glitch) automatically.
- Set "hook_v2" as an object with "enabled": true, an "items" array of (start_time, end_time, text) objects, and a "transition" object with "type" set to "white_flash" or "glitch".
"""

    # Build optional Segment Trimming prompt section
    _segment_prompt = ""
    if cfg and not getattr(cfg, "no_segment_trim", False):
        _silence_hint = ""
        if cfg and getattr(cfg, "silence_trim", False):
            _silence_hint = "\n- Aggressively remove silence, dead air, and pauses. Do not include pauses longer than 0.5 seconds."
        _segment_prompt = f"""

SEGMENT-BASED TRIMMING (KEEP SEGMENTS — REQUIRED):
- For each clip, identify sections that are dull, too quiet, rambling, or filler.
- If needed, split the clip into "keep_segments" and retain only the best parts.
- Each segment contains start_time and end_time. Keep segments chronological and non-overlapping.
- If the full clip is already engaging, use one segment covering its full duration.{_silence_hint}
- Set "keep_segments" to an array of (start_time, end_time) objects.
"""
    return f"""
You are an art director, video editor, and short-form content metadata strategist for TikTok, Reels, and YouTube Shorts.

Analyze the transcript below. Its format is [start_seconds - end_seconds] text.

PRIMARY TASK:
- Find the {jumlah_clip} strongest, most engaging, shareable, and potentially viral moments for short clips.
- Sort clips by viral_score from highest to lowest. Use rank only as a sequence number (1, 2, 3, ...).
- For each clip, provide timing, a hook, a typography plan, a B-roll plan, selection rationale, cross-platform metadata, and target-account classification.
- Keep every choice relevant to the clip itself, not just the full video topic.

CLIP SELECTION AND RETENTION:
- Each clip must be {MIN_CLIP_DURATION}–{MAX_CLIP_DURATION} seconds long.
- Choose moments with emotion, conflict, surprise, insight, a strong opinion, a practical lesson, or a clear punchline.
- Score virality from 1 to 100: 90–100 exceptionally strong; 80–89 engaging; 70–79 informative but less compelling. Avoid scores below 70 unless the transcript has very few good moments.
- Prefer clips that make sense without the full-video context. Avoid repetitive, flat, rambling clips without a clear payoff.
- Make the first three seconds compelling. Aim for: hook → brief context → tension or insight → payoff.
- Start as close as possible to the first strong moment. End after the payoff, conclusion, punchline, or emotional beat. Do not cut off an unfinished sentence or extend the clip after its point is made.
- Avoid unnecessary intros, small talk, long pauses, and transitions. Combine nearby related moments only when the duration remains within limits; separate moments with different angles.

INTERNAL VIRAL SCORE (do not add fields for these components):
- Hook strength: 1–20; emotional intensity: 1–20; shareability/comment potential: 1–20; standalone clarity: 1–20; payoff/retention: 1–20.
- Judge the hook by whether the first three seconds stop a viewer from scrolling, and judge payoff by whether there is a reason to watch through the end.

TARGET ACCOUNT:
Classify each clip by its angle, not just its topic:
{_build_account_classification_prompt()}
- Beauty can fit Business, Life, or Creator depending on the angle. Finance can fit Business, Life, or Creator. Classify owner stories as Business, Life, or Muslim according to their focus.

HOOK (REQUIRED):
- Select one punchy sentence spoken in the clip and record hook_start_time and hook_end_time.
- Keep it natural, faithful to the transcript, compelling within about {durasi_hook} seconds, and suitable as opening on-screen text. Do not use false clickbait.

KINETIC TYPOGRAPHY:
- Select 3–6 individual high-impact words from the transcript; preserve their exact spelling in kata_utama.
- Set scale_level to 1 (small), 2 (large), or 3 (very large); style to utama or khusus; and animasi to bounce_pop or stagger_up.
- Prefer words with strong emotional, semantic, or visual-retention value. Do not select phrases.

B-ROLL (when relevant):
- Suggest at most 1–3 moments, each 3–7 seconds, with start_time, end_time, and a short, clear English search_query.
- Do not overlap the hook. Include B-roll only when it helps visualize the spoken content; otherwise return an empty broll_list.
- Also suggest 2–5 contrasting, funny, dramatic, or curiosity-provoking opening B-roll ideas in recommended_visual_broll_hook, with YouTube/TikTok search keywords and a gesture reference when useful. These are manual-editing suggestions only.

BACKGROUND MUSIC:
Choose one mood from [chill, epic, sad, upbeat, suspense] that fits the clip's emotion and topic.

ENDING AND RATIONALE:
- Extend end_time by 0.10–0.85 seconds after the last word for a natural ending.
- Explain briefly in alasan why the clip is worth selecting: its emotional value, hook, retention, shareability, payoff, main viral trigger, and standalone appeal.

METADATA LANGUAGE RULES:
- Keep title_indonesia in natural Indonesian, at most 100 characters, for internal compatibility/fallback.
- Write title_inggris, hastag, description_hook, description_context, keyword_tags, and tiktok_caption in natural, concise English.
- Write tiktok_title_id and tiktok_caption_id in natural Indonesian for Indonesian TikTok audiences. The title should describe the clip more fully than title_indonesia; the caption may be 1–2 informative, conversational sentences.
- Do not mix languages within a field. Avoid stiff literal translations, excessive punctuation, generic or spammy tags, unsupported claims, and false hyperbole.
- title_inggris: at most 100 characters, one main idea, relevant to the clip, not generic or clickbait.
- hastag: 2–3 relevant English hashtags in one space-separated string; avoid duplicates and generic tags unless clearly relevant.
- description_hook and description_context: exactly one English sentence each; the first should create curiosity, the second should briefly explain the clip.
- keyword_tags: 5–8 short English search phrases, not hashtags or spam; prioritize terms people may search for.
- tiktok_caption: 1–2 short, natural English sentences, complementary to the title and preferably under 140 characters.
- tiktok_title_id: natural Indonesian, clear and more descriptive than title_indonesia. tiktok_caption_id: 1–2 natural, informative Indonesian sentences, not a copy of the title.
- Keep all platform-facing metadata accurate to the clip. Make titles, descriptions, and captions complement one another.

OUTPUT:
Return only a valid JSON array, with every required field filled. Prioritize accuracy over creativity.
{_hook_v2_prompt}{_segment_prompt}

REQUIRED JSON STRUCTURE (use these exact field names):
[
  {{
    "rank": 1,
    "viral_score": 95,
    "start_time": 30.5,
    "end_time": 90.0,
    "hook_start_time": 30.5,
    "hook_end_time": 35.0,
    "bgm_mood": "mood_here",
    "typography_plan": [{{ "kata_utama": "...", "scale_level": 2, "style": "utama", "animasi": "bounce_pop" }}],
    "broll_list": [{{ "start_time": 40.0, "end_time": 45.0, "search_query": "..." }}],
    "recommended_visual_broll_hook": [
      {{ "broll_idea": "...", "search_keyword": "...", "why_it_works": "..." }}
    ],
    "hook_v2": {{
      "enabled": true,
      "items": [{{ "start_time": 31.0, "end_time": 32.5, "text": "KATA KUNCI" }}],
      "transition": {{ "type": "white_flash" }}
    }},
    "keep_segments": [
      {{ "start_time": 30.5, "end_time": 55.0 }},
      {{ "start_time": 58.0, "end_time": 90.0 }}
    ],
    "title_indonesia": "...",
    "title_inggris": "...",
    "hastag": "#hastag1 #hastag2",
    "description_hook": "...",
    "description_context": "...",
    "keyword_tags": ["tag1", "tag2"],
    "tiktok_title_id": "...",
    "tiktok_caption_id": "...",
    "tiktok_caption": "...",
    "alasan": "...",
    "klasifikasi_akun": {{
      "tipe_akun": "Creator",
      "akun_tujuan": "Creator.Mereska",
      "confidence": 87,
      "angle_utama": "Monetizing digital content in the beauty niche",
      "alasan": "...",
      "kata_kunci_pendukung": ["affiliate", "monetisasi"],
      "bio_akun": "...",
      "alternatif_akun": {{
        "tipe_akun": "Life",
        "akun_tujuan": "Life.Mereska",
        "alasan": "..."
      }}
    }}
  }}
]

Transcript:
{transkrip_lengkap}
"""


def analyze_with_nvidia(transkrip_lengkap: str, cfg) -> list[dict]:
    """Analyze transcript using NVIDIA NIM API (OpenAI compatible)."""
    from openai import OpenAI
    
    print(f"[3/3] Analyzing the top {cfg.jumlah_clip} moments with NVIDIA ({cfg.nvidia_model})...")
    
    if not cfg.api_key_nvidia:
        raise ValueError("NVIDIA_API_KEY was not found in the environment.")

    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=cfg.api_key_nvidia
    )
    
    prompt = get_analysis_prompt(transkrip_lengkap, cfg.jumlah_clip, cfg.durasi_hook, cfg=cfg)
    
    # Define the strict schema for Guided JSON (NVIDIA NIM specific)
    clips_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "rank": {"type": "integer"},
                "viral_score": {"type": "integer"},
                "start_time": {"type": "number"},
                "end_time": {"type": "number"},
                "hook_start_time": {"type": "number"},
                "hook_end_time": {"type": "number"},
                "bgm_mood": {
                    "type": "string",
                    "enum": ["chill", "epic", "sad", "upbeat", "suspense"]
                },
                "typography_plan": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "kata_utama": {"type": "string"},
                            "scale_level": {"type": "integer", "enum": [1, 2, 3]},
                            "style": {"type": "string", "enum": ["utama", "khusus"]},
                            "animasi": {"type": "string", "enum": ["bounce_pop", "stagger_up"]}
                        },
                        "required": ["kata_utama", "scale_level", "style", "animasi"]
                    }
                },
                "broll_list": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "start_time": {"type": "number"},
                            "end_time": {"type": "number"},
                            "search_query": {"type": "string"}
                        },
                        "required": ["start_time", "end_time", "search_query"]
                    }
                },
                "recommended_visual_broll_hook": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "broll_idea": {"type": "string"},
                            "search_keyword": {"type": "string"},
                            "why_it_works": {"type": "string"}
                        },
                        "required": ["broll_idea", "search_keyword", "why_it_works"]
                    }
                },
                "title_indonesia": {"type": "string"},
                "title_inggris": {"type": "string"},
                "hastag": {"type": "string"},
                "description_hook": {"type": "string"},
                "description_context": {"type": "string"},
                "keyword_tags": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "tiktok_title_id": {"type": "string"},
                "tiktok_caption_id": {"type": "string"},
                "tiktok_caption": {"type": "string"},
                "alasan": {"type": "string"},
                "klasifikasi_akun": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "tipe_akun": {"type": "string", "enum": list(TARGET_ACCOUNTS.keys())},
                        "akun_tujuan": {"type": "string"},
                        "confidence": {"type": "integer"},
                        "angle_utama": {"type": "string"},
                        "alasan": {"type": "string"},
                        "kata_kunci_pendukung": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "bio_akun": {"type": "string"},
                        "alternatif_akun": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "tipe_akun": {"type": "string", "enum": list(TARGET_ACCOUNTS.keys())},
                                "akun_tujuan": {"type": "string"},
                                "alasan": {"type": "string"}
                            },
                            "required": ["tipe_akun", "akun_tujuan", "alasan"]
                        }
                    },
                    "required": ["tipe_akun", "akun_tujuan", "confidence", "angle_utama", "alasan", "kata_kunci_pendukung", "bio_akun", "alternatif_akun"]
                },
                "hook_v2": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "start_time": {"type": "number"},
                                    "end_time": {"type": "number"},
                                    "text": {"type": "string"},
                                },
                                "required": ["start_time", "end_time", "text"],
                            },
                        },
                        "transition": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "type": {"type": "string", "enum": ["white_flash", "glitch"]},
                            },
                            "required": ["type"],
                        },
                    },
                    "required": ["enabled", "items", "transition"],
                },
                "keep_segments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "start_time": {"type": "number"},
                            "end_time": {"type": "number"},
                        },
                        "required": ["start_time", "end_time"],
                    },
                },
            },
            "required": [
                "rank", "viral_score", "start_time", "end_time", "hook_start_time", "hook_end_time",
                "bgm_mood", "typography_plan", "broll_list", "recommended_visual_broll_hook", "title_indonesia",
                "title_inggris", "hastag", "description_hook", "description_context",
                "keyword_tags", "tiktok_title_id", "tiktok_caption_id", "tiktok_caption",
                "alasan", "klasifikasi_akun", "hook_v2", "keep_segments"
            ]
        }
    }

    completion = client.chat.completions.create(
        model=cfg.nvidia_model,
        messages=[
            {"role": "system", "content": "You are a professional video editor and strategist. Return JSON only. Follow the provided JSON schema exactly."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        top_p=1,
        max_tokens=16384,
        extra_body={
            "chat_template_kwargs": {"thinking": False},
            "nvext": {
                "guided_json": clips_schema
            }
        }
    )
    
    content = completion.choices[0].message.content
    
    if "```" in content:
        content = re.sub(r"```(json)?", "", content).strip()
        content = content.split("```")[0].strip()
        
    hasil = json.loads(content)
    
    # Guided JSON should return an array directly if schema says type: array
    # but we keep the unwrapper just in case of non-conforming fallbacks
    if isinstance(hasil, dict):
        for key in ["clips", "data", "highlights"]:
            if key in hasil and isinstance(hasil[key], list):
                hasil = hasil[key]
                break
                
    if not isinstance(hasil, list):
        if isinstance(hasil, dict):
            return [hasil]
        raise ValueError(f"NVIDIA provider returned an unsupported format (expected list or dict): {type(hasil)}")
        
    return hasil


def analyze_with_ai(transkrip_lengkap: str, cfg) -> list[dict]:
    """Dispatcher for AI analysis based on provider."""
    provider = getattr(cfg, "ai_provider", "gemini")
    
    if provider == "nvidia":
        if not cfg.api_key_nvidia:
            print("⚠️ NVIDIA_API_KEY was not found. Trying Gemini as a fallback...")
        else:
            try:
                return analyze_with_nvidia(transkrip_lengkap, cfg)
            except Exception as e:
                print(f"⚠️ NVIDIA API failed: {e}. Falling back to Gemini...")
    
    return analyze_with_gemini(transkrip_lengkap, cfg)


def analyze_with_gemini(
    transkrip_lengkap: str,
    cfg,
) -> list[dict]:
    """Analyse transcript with Gemini AI."""
    import google.genai as genai
    from google.genai import types

    print(f"[3/3] Analyzing the top {cfg.jumlah_clip} moments with Gemini...")

    prompt = get_analysis_prompt(transkrip_lengkap, cfg.jumlah_clip, cfg.durasi_hook, cfg=cfg)

    # JSON Schema definitions (same as before)
    schema_broll = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "start_time": {"type": "NUMBER"},
                "end_time": {"type": "NUMBER"},
                "search_query": {"type": "STRING"},
            },
            "required": ["start_time", "end_time", "search_query"],
        },
    }

    schema_visual_broll_hook = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "broll_idea": {"type": "STRING"},
                "search_keyword": {"type": "STRING"},
                "why_it_works": {"type": "STRING"},
            },
            "required": ["broll_idea", "search_keyword", "why_it_works"],
        },
    }

    schema_typography = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "kata_utama": {"type": "STRING"},
                "scale_level": {"type": "INTEGER"},
                "style": {"type": "STRING"},
                "animasi": {"type": "STRING"},
            },
            "required": ["kata_utama", "scale_level", "style", "animasi"],
        },
    }

    schema_klasifikasi = {
        "type": "OBJECT",
        "properties": {
            "tipe_akun": {"type": "STRING"},
            "akun_tujuan": {"type": "STRING"},
            "confidence": {"type": "INTEGER"},
            "angle_utama": {"type": "STRING"},
            "alasan": {"type": "STRING"},
            "kata_kunci_pendukung": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
            },
            "bio_akun": {"type": "STRING"},
            "alternatif_akun": {
                "type": "OBJECT",
                "properties": {
                    "tipe_akun": {"type": "STRING"},
                    "akun_tujuan": {"type": "STRING"},
                    "alasan": {"type": "STRING"},
                },
                "required": ["tipe_akun", "akun_tujuan", "alasan"],
            },
        },
        "required": ["tipe_akun", "akun_tujuan", "confidence", "angle_utama", "alasan", "kata_kunci_pendukung", "bio_akun", "alternatif_akun"],
    }

    client = genai.Client(
        api_key=cfg.api_key_gemini,
        # Use the API-key Developer API even when cloud/notebook flags are set.
        vertexai=False,
        http_options=types.HttpOptions(
            # Preserve beta routing for preview models and structured output
            # across SDK versions (SDK 3 changed its default to v1).
            api_version="v1beta",
            timeout=REQUEST_TIMEOUT_MS,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )

    gemini_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema={
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "rank": {"type": "INTEGER"},
                    "viral_score": {"type": "INTEGER"},
                    "hook_start_time": {"type": "NUMBER"},
                    "hook_end_time": {"type": "NUMBER"},
                    "start_time": {"type": "NUMBER"},
                    "end_time": {"type": "NUMBER"},
                    "typography_plan": schema_typography,
                    "broll_list": schema_broll,
                    "recommended_visual_broll_hook": schema_visual_broll_hook,
                    "alasan": {"type": "STRING"},
                    "bgm_mood": {"type": "STRING"},
                    "title_indonesia": {"type": "STRING"},
                    "title_inggris": {"type": "STRING"},
                    "hastag": {"type": "STRING"},
                    "description_hook": {"type": "STRING"},
                    "description_context": {"type": "STRING"},
                    "keyword_tags": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                    },
                    "tiktok_title_id": {"type": "STRING"},
                    "tiktok_caption_id": {"type": "STRING"},
                    "tiktok_caption": {"type": "STRING"},
                    "klasifikasi_akun": schema_klasifikasi,
                    "hook_v2": {
                        "type": "OBJECT",
                        "properties": {
                            "enabled": {"type": "BOOLEAN"},
                            "items": {
                                "type": "ARRAY",
                                "items": {
                                    "type": "OBJECT",
                                    "properties": {
                                        "start_time": {"type": "NUMBER"},
                                        "end_time": {"type": "NUMBER"},
                                        "text": {"type": "STRING"},
                                    },
                                    "required": ["start_time", "end_time", "text"],
                                },
                            },
                            "transition": {
                                "type": "OBJECT",
                                "properties": {
                                    "type": {"type": "STRING"},
                                },
                                "required": ["type"],
                            },
                        },
                        "required": ["enabled", "items", "transition"],
                    },
                    "keep_segments": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "start_time": {"type": "NUMBER"},
                                "end_time": {"type": "NUMBER"},
                            },
                            "required": ["start_time", "end_time"],
                        },
                    },
                },
                "required": [
                    "rank", "viral_score", "hook_start_time", "hook_end_time",
                    "start_time", "end_time", "typography_plan",
                    "broll_list", "recommended_visual_broll_hook", "alasan", "bgm_mood",
                    "title_indonesia", "title_inggris", "hastag",
                    "description_hook", "description_context",
                    "keyword_tags", "tiktok_title_id",
                    "tiktok_caption_id", "tiktok_caption",
                    "klasifikasi_akun", "hook_v2", "keep_segments",
                ],
            },
        },
    )

    return _generate_json_with_retry(
        client=client,
        model=cfg.gemini_model,
        fallback_model=getattr(cfg, "gemini_fallback_model", None),
        contents=prompt,
        config=gemini_config,
    )
