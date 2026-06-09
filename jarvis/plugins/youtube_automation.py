"""
YouTube Faceless Channel Automation Plugin.

Full autonomous pipeline:
  1. Niche research + trending topic discovery
  2. Script generation (AI-written, SEO-optimised)
  3. Text-to-speech voiceover generation
  4. Slide / image asset generation (using Pillow)
  5. Video assembly (using moviepy)
  6. Thumbnail creation
  7. YouTube Data API v3 upload
  8. Metadata optimisation (title, description, tags)
  9. Analytics retrieval
  10. Channel scheduling (daily/weekly)

Requirements (install separately):
  pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
  pip install moviepy Pillow gtts schedule

OAuth credentials → jarvis/data/youtube_credentials.json
Token cache       → jarvis/data/youtube_token.json
"""
from __future__ import annotations

import asyncio
import json
import os
import textwrap
import time
import uuid
from pathlib import Path
from typing import Any

from jarvis.config import settings
from jarvis.core.tools import ToolDef
from jarvis.plugins.base_plugin import BasePlugin


# ── Helpers ────────────────────────────────────────────────────────────────

_DATA = Path(settings.workspace_root).parent / "data"
_WORKSPACE = Path(settings.workspace_root)

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/youtube.force-ssl",
]


def _yt_service():
    """Build an authenticated YouTube service from stored credentials."""
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    token_path = _DATA / "youtube_token.json"
    creds_path = _DATA / "youtube_credentials.json"

    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not creds_path.exists():
                raise FileNotFoundError(
                    f"Place your YouTube OAuth2 credentials at {creds_path}. "
                    "Download from Google Cloud Console → APIs & Services → Credentials."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)
            creds = flow.run_local_server(port=0)
        token_path.write_text(creds.to_json())

    return build("youtube", "v3", credentials=creds)


# ── Tool implementations ───────────────────────────────────────────────────

async def research_niche(niche: str, num_topics: int = 10) -> str:
    """Search for trending topics in a YouTube niche using DuckDuckGo."""
    try:
        from duckduckgo_search import DDGS
        queries = [
            f"best faceless YouTube channel ideas {niche} 2025",
            f"trending {niche} YouTube topics high CPM",
            f"{niche} YouTube niche revenue statistics",
        ]
        results = []
        with DDGS() as ddgs:
            for q in queries:
                for r in ddgs.text(q, max_results=5):
                    results.append(f"• {r['title']}: {r.get('body', '')[:200]}")
        return f"Niche Research for '{niche}':\n\n" + "\n".join(results[:num_topics])
    except Exception as e:
        return f"Research error: {e}"


async def generate_video_script(
    topic: str,
    niche: str,
    duration_minutes: int = 8,
    style: str = "educational",
) -> str:
    """Generate a full faceless YouTube video script using Claude."""
    try:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        word_count = duration_minutes * 130  # ~130 words/min for TTS

        prompt = f"""Write a complete {style} YouTube video script for a FACELESS channel.

Topic: {topic}
Niche: {niche}
Target length: {duration_minutes} minutes (~{word_count} words)

Requirements:
- Hook in the first 30 seconds (creates curiosity/tension)
- Clear section headers marked as [SECTION: Name]
- No references to visuals or on-camera presenter
- Written for text-to-speech delivery (clear, punchy sentences)
- SEO-optimised — include the target keyword naturally 5-8 times
- Strong call-to-action at end (subscribe, comment, watch next)
- Include [PAUSE] markers where B-roll would change

Also provide at the end:
TITLE: (5 YouTube title options with numbers/curiosity gaps)
DESCRIPTION: (400-word SEO description)
TAGS: (30 comma-separated tags)
THUMBNAIL_TEXT: (3-word bold text for thumbnail)
"""
        resp = await client.messages.create(
            model=settings.planner_model,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
        )
        script = resp.content[0].text
        # Save to workspace
        slug = topic[:40].lower().replace(" ", "_").replace("/", "_")
        path = _WORKSPACE / "scripts" / f"{slug}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(script)
        return f"Script saved to workspace/scripts/{slug}.txt\n\n---\n\n{script[:2000]}..."
    except Exception as e:
        return f"Script generation error: {e}"


async def create_voiceover(script_path: str, output_name: str | None = None) -> str:
    """Convert script text to MP3 voiceover using TTS engine."""
    try:
        full_path = _WORKSPACE / script_path
        if not full_path.exists():
            return f"Script not found: {script_path}"

        text = full_path.read_text()
        # Strip metadata sections (TITLE:, DESCRIPTION:, etc.)
        if "TITLE:" in text:
            text = text[: text.index("TITLE:")].strip()
        # Remove [SECTION:...] markers (keep natural reading)
        import re
        text = re.sub(r"\[SECTION:[^\]]+\]", "", text)
        text = re.sub(r"\[PAUSE\]", "...", text)

        from jarvis.voice.tts import TTSEngine
        tts = TTSEngine()
        slug = output_name or Path(script_path).stem
        out_path = str(_WORKSPACE / "audio" / f"{slug}_voiceover.mp3")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        saved = await tts.speak_and_save(text, out_path)
        if saved:
            return f"Voiceover saved: {saved}"
        return "Voiceover failed — check TTS credentials."
    except Exception as e:
        return f"Voiceover error: {e}"


async def create_thumbnail(
    title_text: str,
    background_color: str = "#1a1a2e",
    accent_color: str = "#e94560",
    output_name: str | None = None,
) -> str:
    """Create a YouTube thumbnail (1280×720) using Pillow."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import textwrap as tw

        img = Image.new("RGB", (1280, 720), color=background_color)
        draw = ImageDraw.Draw(img)

        # Gradient overlay
        for y in range(720):
            alpha = int(y / 720 * 80)
            draw.line([(0, y), (1280, y)], fill=(*_hex_to_rgb(accent_color), alpha))

        # Bold title text (wrapped)
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 90)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 45)
        except Exception:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()

        lines = tw.wrap(title_text.upper(), width=18)
        y_text = 720 // 2 - len(lines) * 55
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font_large)
            w = bbox[2] - bbox[0]
            # Shadow
            draw.text((640 - w // 2 + 4, y_text + 4), line, fill="#000000", font=font_large)
            draw.text((640 - w // 2, y_text), line, fill="#FFFFFF", font=font_large)
            y_text += 110

        # Accent bar
        draw.rectangle([(0, 670), (1280, 720)], fill=accent_color)

        slug = (output_name or title_text[:30]).lower().replace(" ", "_")
        out_path = _WORKSPACE / "thumbnails" / f"{slug}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(out_path))
        return f"Thumbnail saved: {out_path} (1280×720)"
    except ImportError:
        return "Pillow not installed. Run: pip install Pillow"
    except Exception as e:
        return f"Thumbnail error: {e}"


async def assemble_video(
    script_path: str,
    audio_path: str | None = None,
    output_name: str | None = None,
    background_style: str = "dark",
) -> str:
    """
    Assemble a faceless video from script + voiceover.
    Creates slides with text segments synced to the TTS audio.
    Requires: pip install moviepy Pillow
    """
    try:
        from moviepy.editor import (
            AudioFileClip, TextClip, CompositeVideoClip,
            ColorClip, concatenate_videoclips, ImageClip,
        )
        from PIL import Image, ImageDraw, ImageFont
        import re

        full_script = (_WORKSPACE / script_path).read_text()
        # Split into sections
        sections = re.split(r"\[SECTION:[^\]]*\]", full_script)
        sections = [s.strip() for s in sections if s.strip()][:12]

        bg_color = (26, 26, 46) if background_style == "dark" else (245, 245, 245)
        slug = output_name or Path(script_path).stem

        clips = []
        for i, section in enumerate(sections):
            # ~8 seconds per section slide
            duration = 8
            bg = ColorClip(size=(1920, 1080), color=bg_color, duration=duration)
            # Wrap text
            wrapped = "\n".join(textwrap.wrap(section[:300], width=50))
            try:
                txt = TextClip(
                    wrapped, fontsize=48, color="white",
                    font="DejaVu-Sans-Bold", method="caption",
                    size=(1600, None), align="center",
                )
                txt = txt.set_position("center").set_duration(duration)
                clip = CompositeVideoClip([bg, txt])
            except Exception:
                clip = bg
            clips.append(clip)

        if not clips:
            return "No content sections found in script."

        final = concatenate_videoclips(clips, method="compose")

        # Add audio if available
        if audio_path:
            full_audio = _WORKSPACE / audio_path
            if full_audio.exists():
                audio = AudioFileClip(str(full_audio))
                final = final.set_audio(audio.subclip(0, min(audio.duration, final.duration)))

        out_path = str(_WORKSPACE / "videos" / f"{slug}.mp4")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        final.write_videofile(out_path, fps=24, codec="libx264", audio_codec="aac",
                              logger=None)
        size_mb = Path(out_path).stat().st_size / 1_000_000
        return f"Video assembled: {out_path} ({size_mb:.1f} MB)"
    except ImportError as e:
        return f"Missing dependency: {e}. Install: pip install moviepy Pillow"
    except Exception as e:
        return f"Video assembly error: {e}"


async def upload_to_youtube(
    video_path: str,
    title: str,
    description: str,
    tags: str,
    category_id: str = "22",  # 22 = People & Blogs, 27 = Education
    privacy: str = "public",
    thumbnail_path: str | None = None,
) -> str:
    """Upload a video to YouTube via the Data API v3."""
    try:
        from googleapiclient.http import MediaFileUpload

        full_path = _WORKSPACE / video_path
        if not full_path.exists():
            return f"Video not found: {video_path}"

        service = _yt_service()
        tag_list = [t.strip() for t in tags.split(",")][:30]

        body = {
            "snippet": {
                "title": title[:100],
                "description": description[:5000],
                "tags": tag_list,
                "categoryId": category_id,
            },
            "status": {
                "privacyStatus": privacy,
                "selfDeclaredMadeForKids": False,
            },
        }

        media = MediaFileUpload(str(full_path), mimetype="video/mp4", resumable=True)
        req = service.videos().insert(part="snippet,status", body=body, media_body=media)

        response = None
        while response is None:
            status, response = req.next_chunk()
            if status:
                pct = int(status.progress() * 100)
                print(f"[YouTube] Upload {pct}%")

        video_id = response.get("id", "unknown")
        url = f"https://youtu.be/{video_id}"

        # Set thumbnail if provided
        if thumbnail_path:
            thumb_path = _WORKSPACE / thumbnail_path
            if thumb_path.exists():
                service.thumbnails().set(
                    videoId=video_id,
                    media_body=str(thumb_path),
                ).execute()

        return f"Uploaded successfully!\nVideo ID: {video_id}\nURL: {url}"
    except FileNotFoundError as e:
        return str(e)
    except ImportError:
        return "Install: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
    except Exception as e:
        return f"Upload error: {e}"


async def get_channel_analytics(metric: str = "views") -> str:
    """Fetch basic channel analytics from YouTube."""
    try:
        service = _yt_service()
        channels = service.channels().list(part="statistics,snippet", mine=True).execute()
        items = channels.get("items", [])
        if not items:
            return "No channel found for authenticated account."
        ch = items[0]
        stats = ch["statistics"]
        name = ch["snippet"]["title"]
        return (
            f"Channel: {name}\n"
            f"Subscribers: {stats.get('subscriberCount', 'N/A')}\n"
            f"Total Views: {stats.get('viewCount', 'N/A')}\n"
            f"Videos: {stats.get('videoCount', 'N/A')}\n"
            f"(Revenue estimates require YouTube Partner Program access)"
        )
    except Exception as e:
        return f"Analytics error: {e}"


async def run_full_pipeline(
    topic: str,
    niche: str,
    duration_minutes: int = 8,
    auto_upload: bool = False,
    privacy: str = "private",
) -> str:
    """
    Run the COMPLETE faceless YouTube video pipeline end-to-end:
    research → script → voiceover → thumbnail → video → (optional upload)
    """
    results = []
    slug = topic[:30].lower().replace(" ", "_")

    results.append("=== JARVIS YOUTUBE PIPELINE ===\n")

    # 1. Research
    results.append("[1/6] Researching niche...")
    research = await research_niche(niche, num_topics=5)
    results.append(research[:500])

    # 2. Script
    results.append("\n[2/6] Generating script...")
    script_result = await generate_video_script(topic, niche, duration_minutes)
    results.append(script_result[:300])
    script_path = f"scripts/{slug}.txt"

    # 3. Voiceover
    results.append("\n[3/6] Generating voiceover...")
    vo_result = await create_voiceover(script_path, slug)
    results.append(vo_result)
    audio_path = f"audio/{slug}_voiceover.mp3"

    # 4. Thumbnail
    results.append("\n[4/6] Creating thumbnail...")
    thumb_result = await create_thumbnail(topic, output_name=slug)
    results.append(thumb_result)
    thumb_path = f"thumbnails/{slug}.png"

    # 5. Video assembly
    results.append("\n[5/6] Assembling video...")
    video_result = await assemble_video(script_path, audio_path, slug)
    results.append(video_result)
    video_path = f"videos/{slug}.mp4"

    # 6. Optional upload
    if auto_upload:
        results.append("\n[6/6] Uploading to YouTube...")
        title = f"{topic} — Everything You Need to Know"
        description = f"In this video we cover everything about {topic} in {niche}.\n\nSubscribe for more!"
        tags = f"{niche},{topic},faceless,educational,2025"
        upload_result = await upload_to_youtube(
            video_path, title, description, tags,
            privacy=privacy, thumbnail_path=thumb_path,
        )
        results.append(upload_result)
    else:
        results.append("\n[6/6] Upload skipped (auto_upload=False).")
        results.append(f"To upload manually: use the upload_to_youtube tool with video_path='{video_path}'")

    results.append("\n=== PIPELINE COMPLETE ===")
    return "\n".join(results)


# ── Helpers ────────────────────────────────────────────────────────────────

def _hex_to_rgb(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


# ── Plugin class ───────────────────────────────────────────────────────────

class YouTubeAutomationPlugin(BasePlugin):
    name = "youtube_automation"
    description = "Autonomous faceless YouTube channel pipeline"
    version = "1.0.0"

    def tools(self) -> list[ToolDef]:
        return [
            ToolDef(
                name="yt_research_niche",
                description="Research trending topics and revenue potential for a YouTube niche.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "niche": {"type": "string"},
                        "num_topics": {"type": "integer", "default": 10},
                    },
                    "required": ["niche"],
                },
                fn=research_niche,
            ),
            ToolDef(
                name="yt_generate_script",
                description="Generate a complete SEO-optimised faceless YouTube video script.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "niche": {"type": "string"},
                        "duration_minutes": {"type": "integer", "default": 8},
                        "style": {"type": "string", "default": "educational"},
                    },
                    "required": ["topic", "niche"],
                },
                fn=generate_video_script,
            ),
            ToolDef(
                name="yt_create_voiceover",
                description="Convert a saved script to an MP3 voiceover using TTS.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "script_path": {"type": "string"},
                        "output_name": {"type": "string"},
                    },
                    "required": ["script_path"],
                },
                fn=create_voiceover,
            ),
            ToolDef(
                name="yt_create_thumbnail",
                description="Generate a professional YouTube thumbnail image (1280×720).",
                input_schema={
                    "type": "object",
                    "properties": {
                        "title_text": {"type": "string"},
                        "background_color": {"type": "string", "default": "#1a1a2e"},
                        "accent_color": {"type": "string", "default": "#e94560"},
                        "output_name": {"type": "string"},
                    },
                    "required": ["title_text"],
                },
                fn=create_thumbnail,
            ),
            ToolDef(
                name="yt_assemble_video",
                description="Assemble a faceless video from a script and audio into an MP4.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "script_path": {"type": "string"},
                        "audio_path": {"type": "string"},
                        "output_name": {"type": "string"},
                        "background_style": {"type": "string", "default": "dark"},
                    },
                    "required": ["script_path"],
                },
                fn=assemble_video,
            ),
            ToolDef(
                name="yt_upload",
                description="Upload a video file to YouTube with title, description, tags, and thumbnail.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "video_path": {"type": "string"},
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "tags": {"type": "string", "description": "comma-separated tags"},
                        "category_id": {"type": "string", "default": "27"},
                        "privacy": {"type": "string", "default": "public"},
                        "thumbnail_path": {"type": "string"},
                    },
                    "required": ["video_path", "title", "description", "tags"],
                },
                fn=upload_to_youtube,
            ),
            ToolDef(
                name="yt_channel_analytics",
                description="Retrieve channel statistics: subscribers, total views, video count.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "metric": {"type": "string", "default": "views"},
                    },
                },
                fn=get_channel_analytics,
            ),
            ToolDef(
                name="yt_run_pipeline",
                description=(
                    "Run the FULL autonomous faceless YouTube pipeline: "
                    "niche research → script → voiceover → thumbnail → video → optional upload. "
                    "This is the main command for creating a complete video from scratch."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string", "description": "Video topic"},
                        "niche": {"type": "string", "description": "Channel niche e.g. 'finance', 'health', 'tech'"},
                        "duration_minutes": {"type": "integer", "default": 8},
                        "auto_upload": {"type": "boolean", "default": False},
                        "privacy": {"type": "string", "default": "private"},
                    },
                    "required": ["topic", "niche"],
                },
                fn=run_full_pipeline,
            ),
        ]
