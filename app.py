from __future__ import annotations

import os
# Hide CUDA from ctranslate2/faster-whisper on machines without CUDA runtime DLLs.
# Must be set before ctranslate2 is imported. Use setdefault so an explicit
# CUDA_VISIBLE_DEVICES in the environment (e.g. "0") is respected.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from modules.background_manager import BackgroundManager
from modules.cleanup_manager import force_memory_cleanup
from modules.ffmpeg_renderer import FFmpegRenderer, RenderConfig
from modules.llm_client import IMAGE_PROVIDERS, PROVIDERS, ImageGenerator, LLMClient
from modules.metadata_generator import generate_part_metadata
from modules.story_generator import StoryGenerator
from modules.story_splitter import StoryPart, split_story_into_parts
from modules.subtitle_engine import SubtitleEngine
from modules.tts_engine import KokoroTTSEngine
from modules import autopilot_controller
from modules import story_uploader
from modules.utils import (
    ensure_core_directories,
    ensure_story_output_structure,
    get_project_paths,
    has_ffmpeg_with_local_fallback,
    load_json,
    next_story_id,
    read_autopilot_control,
    setup_logging,
    write_autopilot_control,
    write_json,
)
from modules.video_engine import VideoPipelineEngine
from modules.youtube_accounts import YouTubeAccount, YouTubeAccountManager
from modules.youtube_uploader import YouTubeUploadResult, YouTubeUploader
from modules.auth_backend import AuthBackend


BASE_DIR = Path(__file__).resolve().parent
PATHS = get_project_paths(BASE_DIR)
ensure_core_directories(PATHS)
SETTINGS_PATH = PATHS.config_dir / "settings.json"
SECRETS_PATH = PATHS.config_dir / "secrets.json"


def _load_dotenv() -> None:
    """Load a local .env file (if present) into the environment.

    Only sets variables that are not already defined, so a real environment
    (e.g. systemd EnvironmentFile on the server) takes precedence.
    """
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()


def _load_settings() -> dict[str, Any]:
    if not SETTINGS_PATH.exists():
        raise RuntimeError(f"Missing settings file: {SETTINGS_PATH}")
    return load_json(SETTINGS_PATH)


def _default_secrets() -> dict[str, Any]:
    return {
        "llm": {
            "provider": "gemini",
            "api_key": "",
            "model": "gemini-1.5-flash",
            "base_url": "",
        },
        "image": {
            "provider": "openai",
            "api_key": "",
            "model": "gpt-image-1",
            "base_url": "",
            "size": "1024x1792",
        },
    }


def _load_secrets() -> dict[str, Any]:
    """Load persisted credentials; merge over defaults so new fields are added safely."""
    merged = _default_secrets()
    if SECRETS_PATH.exists():
        try:
            saved = load_json(SECRETS_PATH)
        except Exception:
            saved = {}
        for section in ("llm", "image"):
            if isinstance(saved.get(section), dict):
                merged[section].update(saved[section])
    return merged


def _save_secrets(secrets: dict[str, Any]) -> None:
    write_json(SECRETS_PATH, secrets)


def _read_autopilot_control(cache_dir: Path) -> dict[str, Any]:
    return read_autopilot_control(cache_dir)


def _write_autopilot_control(cache_dir: Path, control: dict[str, Any]) -> None:
    write_autopilot_control(cache_dir, control)


def _read_autopilot_state(cache_dir: Path) -> dict[str, Any]:
    state_path = cache_dir / "autopilot" / "state.json"
    if state_path.exists():
        try:
            return load_json(state_path)
        except Exception:
            return {}
    return {}


def _today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _load_optional_json(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            return load_json(path)
        except Exception:
            return {}
    return {}


def _infer_whisper_device() -> tuple[str, str]:
    # Do NOT import ctranslate2 here — it hard-crashes the Streamlit process on
    # machines without CUDA 12 (cublas64_12.dll missing).  Whisper runs in an
    # isolated subprocess (whisper_worker.py) so we just declare CPU/float32.
    # float32 avoids cuBLAS entirely and only needs libiomp5md.dll (bundled).
    return "cpu", "float32"


def _build_render_config(settings: dict[str, Any], prefer_gpu: bool) -> RenderConfig:
    render = settings["render"]
    return RenderConfig(
        width=int(render["width"]),
        height=int(render["height"]),
        fps=int(render["fps"]),
        video_codec=render["video_codec"],
        audio_codec=render["audio_codec"],
        video_bitrate=render["video_bitrate"],
        audio_bitrate=render["audio_bitrate"],
        threads=max(1, min(2, int(render["threads"]))),
        subtitle_font_size=int(render["subtitle_font_size"]),
        subtitle_alignment=int(render["subtitle_alignment"]),
        subtitle_margin_v=int(render["subtitle_margin_v"]),
        subtitle_primary_color=render["subtitle_primary_color"],
        subtitle_outline_color=render["subtitle_outline_color"],
        prefer_nvenc=prefer_gpu,
    )


# Voice metadata: (display label, preview sentence)
_VOICE_META: dict[str, tuple[str, str]] = {
    "af_heart":   ("Heart (Female, Warm)",     "Hi, I'm Heart. I'll be your warm and friendly narrator."),
    "af_sarah":   ("Sarah (Female, Clear)",     "Hi, I'm Sarah. I narrate with a clear and steady voice."),
    "af_sky":     ("Sky (Female, Bright)",      "Hi, I'm Sky. I bring a bright and upbeat tone to every story."),
    "am_michael": ("Michael (Male, Deep)",      "Hi, I'm Michael. I'll guide you through every story with a deep, calm voice."),
    "am_adam":    ("Adam (Male, Calm)",         "Hi, I'm Adam. I narrate with a smooth and relaxed delivery."),
}


def _default_voice_choices(default_voice: str) -> list[str]:
    base = list(_VOICE_META.keys())
    extras = {default_voice} - set(base)
    return base + sorted(extras)


def _ensure_voice_previews(cache_dir: Path) -> None:
    """Pre-generate all voice preview WAVs if they don't exist yet."""
    missing = [
        v for v, (_, text) in _VOICE_META.items()
        if not (cache_dir / "voice_previews" / f"{v}.wav").exists()
    ]
    if not missing:
        return
    try:
        settings = _load_settings()
        tts_cfg = settings["tts"]
        engine = KokoroTTSEngine(
            model_path=BASE_DIR / tts_cfg["model_path"],
            voices_path=BASE_DIR / tts_cfg["voices_path"],
            model_url=tts_cfg["model_url"],
            voices_url=tts_cfg["voices_url"],
            language=tts_cfg["language"],
            speed=float(tts_cfg["speed"]),
        )
        (cache_dir / "voice_previews").mkdir(parents=True, exist_ok=True)
        for v in missing:
            out = cache_dir / "voice_previews" / f"{v}.wav"
            _, text = _VOICE_META[v]
            try:
                engine.synthesize_to_file(text, v, out)
            except Exception:
                pass
    except Exception:
        pass


@st.cache_data(show_spinner=False)
def _generate_voice_preview(voice: str, text: str, cache_dir: str) -> bytes | None:
    """Load a pre-generated voice preview WAV from disk."""
    out = Path(cache_dir) / "voice_previews" / f"{voice}.wav"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        settings = _load_settings()
        tts_cfg = settings["tts"]
        try:
            engine = KokoroTTSEngine(
                model_path=BASE_DIR / tts_cfg["model_path"],
                voices_path=BASE_DIR / tts_cfg["voices_path"],
                model_url=tts_cfg["model_url"],
                voices_url=tts_cfg["voices_url"],
                language=tts_cfg["language"],
                speed=float(tts_cfg["speed"]),
            )
            engine.synthesize_to_file(text, voice, out)
        except Exception:
            return None
    try:
        return out.read_bytes() if out.exists() else None
    except Exception:
        return None


def _story_to_metadata(
    story_title: str,
    topic: str,
    parts: list[StoryPart],
) -> dict[str, Any]:
    part_payload = []
    for part in parts:
        md = generate_part_metadata(story_title, topic, part)
        part_payload.append(
            {
                "part_index": part.index,
                "part_title": part.part_title,
                "caption": md.caption,
                "hashtags": md.hashtags,
                "script": part.text,
            }
        )
    return {
        "story_title": story_title,
        "topic": topic,
        "platform_targets": ["tiktok", "instagram_reels", "facebook_reels", "youtube_shorts"],
        "parts": part_payload,
    }


def _save_uploaded_background(uploaded_file: Any, backgrounds_dir: Path) -> Path:
    """Save a Streamlit uploaded video file to assets/backgrounds/uploaded/ and return its path."""
    upload_dir = backgrounds_dir / "uploaded"
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / uploaded_file.name
    dest.write_bytes(uploaded_file.getbuffer())
    return dest


_YT_CATEGORIES = {
    "Film & Animation": "1",
    "Autos & Vehicles": "2",
    "Music": "10",
    "Pets & Animals": "15",
    "Sports": "17",
    "Travel & Events": "19",
    "Gaming": "20",
    "People & Blogs": "22",
    "Comedy": "23",
    "Entertainment": "24",
    "News & Politics": "25",
    "Howto & Style": "26",
    "Education": "27",
    "Science & Technology": "28",
}


def _save_youtube_client_secret(uploaded_file: Any, cache_dir: Path) -> Path:
    dest_dir = cache_dir / "youtube"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "client_secret.json"
    dest.write_bytes(uploaded_file.getbuffer())
    return dest


def _build_youtube_uploader(cache_dir: Path) -> YouTubeUploader | None:
    secret = cache_dir / "youtube" / "client_secret.json"
    token = cache_dir / "youtube" / "token.json"
    if not secret.exists():
        return None
    return YouTubeUploader(client_secrets_path=secret, token_path=token)


def _publish_part_to_youtube(
    uploader: YouTubeUploader,
    artifact: Any,
    part: StoryPart,
    meta: dict[str, Any] | None,
    story_title: str,
    youtube_title: str,
    youtube_description: str,
    youtube_tags: str,
    youtube_category: str,
    youtube_privacy: str,
) -> YouTubeUploadResult:
    if meta and meta.get("youtube_title"):
        title = meta["youtube_title"]
    else:
        title = youtube_title.strip() or f"{story_title} - Part {part.index}"

    if meta and meta.get("youtube_description"):
        description = meta["youtube_description"]
    else:
        caption = meta.get("caption", "") if meta else part.cliffhanger
        hashtags = " ".join(meta.get("hashtags", [])) if meta else ""
        description = youtube_description.strip() or f"{caption}\n\n{hashtags}".strip()

    if meta and meta.get("youtube_tags"):
        tags = list(meta["youtube_tags"])
    else:
        tags = [t.strip() for t in youtube_tags.split(",") if t.strip()]

    result = uploader.upload_video(
        video_path=artifact.final_video_path,
        title=title,
        description=description,
        tags=tags,
        category_id=_YT_CATEGORIES.get(youtube_category, "24"),
        privacy_status=youtube_privacy,
    )
    st.success(f"Uploaded Part {part.index} to YouTube: {result.video_url}")
    return result


def _enrich_parts_with_ai_youtube_metadata(
    story_generator: StoryGenerator,
    story_title: str,
    topic: str,
    parts: list[StoryPart],
    payload: dict[str, Any],
) -> None:
    """Generate per-part YouTube title/description/tags with Gemini and attach to payload."""
    for part in parts:
        meta = story_generator.generate_youtube_metadata(
            story_title=story_title,
            topic=topic,
            part_index=part.index,
            part_title=part.part_title,
            part_text=part.text,
        )
        for p in payload["parts"]:
            if p["part_index"] == part.index:
                p["youtube_title"] = meta.title
                p["youtube_description"] = meta.description
                p["youtube_tags"] = meta.tags
                break


def render_app() -> None:
    st.set_page_config(page_title="Local AI Shorts Factory", layout="wide")

    auth = AuthBackend(PATHS.cache_dir / "auth.db")
    bootstrap_username = os.getenv("APP_USERNAME", "")
    bootstrap_email = os.getenv("APP_EMAIL", "")
    bootstrap_password = os.getenv("APP_PASSWORD", "")
    if bootstrap_username and bootstrap_email and bootstrap_password:
        auth.ensure_user(bootstrap_username, bootstrap_email, bootstrap_password)

    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        st.title("Local AI Shorts Factory")
        tab_login, tab_reset = st.tabs(["Sign in", "Forgot password"])

        with tab_login:
            username = st.text_input("Username or email", key="auth_identity")
            password = st.text_input("Password", type="password", key="auth_password")
            if st.button("Sign in", type="primary", use_container_width=True):
                if auth.authenticate(username, password):
                    st.session_state["authenticated"] = True
                    st.session_state["authenticated_user"] = username.lower()
                    st.rerun()
                else:
                    st.error("Invalid credentials or account temporarily locked.")

        with tab_reset:
            reset_identity = st.text_input("Username or email", key="reset_identity")
            if st.button("Send reset link", use_container_width=True):
                reset_request = auth.create_reset_token(reset_identity)
                if reset_request:
                    token, account_email = reset_request
                    reset_url = os.getenv("APP_PUBLIC_URL", "http://79.72.89.208:8501")
                    sent = auth.send_reset_email(account_email, token, reset_url)
                    if sent:
                        st.success("If SMTP is configured, a reset link was sent to the account email.")
                    else:
                        st.warning("SMTP is not configured. Use server administration to issue a reset.")
                else:
                    st.success("If that account exists, a reset link will be sent.")

        token = st.query_params.get("token")
        if token:
            with st.expander("Create a new password", expanded=True):
                new_password = st.text_input("New password", type="password", key="new_password")
                confirm_password = st.text_input("Confirm password", type="password", key="confirm_password")
                if st.button("Reset password"):
                    if len(new_password) < 12:
                        st.error("Password must be at least 12 characters.")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match.")
                    elif auth.reset_password(str(token), new_password):
                        st.query_params.clear()
                        st.success("Password reset. Sign in with the new password.")
                    else:
                        st.error("Reset link is invalid or expired.")
        st.stop()

    st.title("Local AI Shorts Automation Engine")
    st.caption("Connected story generation, local narration, subtitles, and vertical short export")
    if st.sidebar.button("Sign out"):
        st.session_state["authenticated"] = False
        st.rerun()

    settings = _load_settings()

    account_manager = YouTubeAccountManager(PATHS.cache_dir)

    # Load persisted API keys / provider config once, so they survive restarts.
    if "app_secrets" not in st.session_state:
        st.session_state["app_secrets"] = _load_secrets()

    if "log_path" not in st.session_state:
        st.session_state["log_path"] = str(setup_logging(PATHS.logs_dir, "pipeline"))

    logger = logging.getLogger("app")

    with st.sidebar:
        st.header("Pipeline Settings")

        # ── Script mode ──────────────────────────────────────────────────────
        script_mode = st.radio(
            "Script Mode",
            options=["AI Generate", "Manual Script"],
            horizontal=True,
        )

        st.divider()

        # ── Mode-specific inputs ──────────────────────────────────────────────
        _DEFAULT_MODELS = {
            "gemini": "gemini-1.5-flash",
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-5-haiku-latest",
            "openai_compatible": "",
        }
        if script_mode == "AI Generate":
            secrets = st.session_state["app_secrets"]
            saved_llm = secrets["llm"]

            provider_keys = list(PROVIDERS.keys())
            if "llm_provider" not in st.session_state:
                st.session_state["llm_provider"] = (
                    saved_llm["provider"] if saved_llm["provider"] in provider_keys else provider_keys[0]
                )
            llm_provider = st.selectbox(
                "LLM Provider",
                options=provider_keys,
                format_func=lambda k: PROVIDERS[k],
                key="llm_provider",
            )

            if "llm_api_key" not in st.session_state:
                st.session_state["llm_api_key"] = saved_llm["api_key"]
            llm_api_key = st.text_input("API Key", type="password", key="llm_api_key")

            if "llm_model" not in st.session_state:
                st.session_state["llm_model"] = saved_llm["model"] or _DEFAULT_MODELS.get(llm_provider, "")
            llm_model = st.text_input(
                "Model",
                key="llm_model",
                placeholder="e.g. gpt-4o-mini / gemini-1.5-flash / claude-3-5-haiku-latest",
            )

            llm_base_url = ""
            if llm_provider == "openai_compatible":
                if "llm_base_url" not in st.session_state:
                    st.session_state["llm_base_url"] = saved_llm["base_url"]
                llm_base_url = st.text_input(
                    "Base URL",
                    key="llm_base_url",
                    placeholder="https://api.openai.com/v1 (or DeepSeek/Groq/Ollama endpoint)",
                )

            topic = st.text_area("Story Topic", value="Scary historical shipwrecks", height=80)
            story_batches = st.selectbox("Number of Stories", options=[1, 2, 3, 5], index=0)
            manual_title: str = ""
            manual_script: str = ""

            # Persist credentials so they survive restarts.
            secrets["llm"] = {
                "provider": llm_provider,
                "api_key": llm_api_key,
                "model": llm_model,
                "base_url": llm_base_url,
            }
            _save_secrets(secrets)
        else:
            llm_provider = "gemini"
            llm_api_key = ""
            llm_model = ""
            llm_base_url = ""
            topic = ""
            story_batches = 1
            manual_title = st.text_input("Story Title", value="My Story", placeholder="Enter a title for your story")
            manual_script = st.text_area(
                "Paste Full Script",
                height=220,
                placeholder=(
                    "Paste your entire story here. It will be automatically split into "
                    "parts. Aim for 150–200 words per part you want to produce."
                ),
            )

        shorts_per_story = st.selectbox("Shorts per Story", options=[1, 5, 10], index=0)
        total_story_minutes = st.number_input(
            "Total Story Duration (minutes)",
            min_value=1,
            max_value=60,
            value=10,
            step=1,
            help="Total narration time for the full story. Each short gets an equal share.",
        )
        per_short_minutes = float(total_story_minutes) / float(shorts_per_story)
        st.caption(f"Per short target duration: {per_short_minutes:.2f} min")
        subtitle_enabled = st.toggle("Enable Subtitles", value=True)
        if subtitle_enabled:
            subtitle_position = st.radio(
                "Caption Position",
                options=["Top", "Middle", "Bottom"],
                index=2,
                horizontal=True,
                help="Where on the screen the subtitles appear.",
            )
            col1, col2 = st.columns([3, 1])
            with col1:
                subtitle_font_size = st.slider(
                    "Caption Text Size",
                    min_value=10,
                    max_value=20,
                    value=int(settings["render"]["subtitle_font_size"]),
                    step=1,
                    help="Font size of the subtitle text. Smaller values keep captions compact.",
                )
            with col2:
                st.metric("Size", f"{subtitle_font_size}px")
            
            # Show size description
            if subtitle_font_size <= 12:
                size_desc = "Small and compact"
            elif subtitle_font_size <= 16:
                size_desc = "Balanced"
            elif subtitle_font_size <= 18:
                size_desc = "Large"
            else:
                size_desc = "Very large"
            st.caption(size_desc)
        else:
            subtitle_position = "Bottom"
            subtitle_font_size = int(settings["render"]["subtitle_font_size"])

        long_video_enabled = st.toggle(
            "Generate Long Video (full story)",
            value=True,
            help="Renders the complete story as one long vertical video in addition to the shorts.",
        )

        st.divider()

        # ── Voice & quality ───────────────────────────────────────────────────
        tts_cfg = settings["tts"]
        voice_options = _default_voice_choices(tts_cfg["default_voice"])
        voice_labels = [_VOICE_META.get(v, (v, ""))[0] for v in voice_options]
        voice_idx = st.selectbox(
            "Voice",
            options=range(len(voice_options)),
            format_func=lambda i: voice_labels[i],
            index=0,
        )
        voice = voice_options[voice_idx]

        # Pre-generate all previews in the background (first sidebar load only)
        if "voice_previews_ready" not in st.session_state:
            with st.spinner("Preparing voice previews\u2026"):
                _ensure_voice_previews(PATHS.cache_dir)
            st.session_state["voice_previews_ready"] = True

        # Voice preview — instant playback from cached file
        with st.expander("\u25b6 Preview selected voice", expanded=False):
            preview_path = PATHS.cache_dir / "voice_previews" / f"{voice}.wav"
            if preview_path.exists():
                st.audio(str(preview_path), format="audio/wav")
            else:
                if st.button("Generate preview", key="voice_preview_btn"):
                    with st.spinner("Generating\u2026"):
                        audio_bytes = _generate_voice_preview(
                            voice,
                            _VOICE_META.get(voice, (voice, f"Hi, I am {voice}."))[1],
                            str(PATHS.cache_dir),
                        )
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/wav")
                    else:
                        st.warning("Could not generate preview \u2014 check TTS model files.")

        quality = st.selectbox("Render Quality", options=["Draft", "Balanced", "High"], index=2)

        st.divider()

        # ── Background source ─────────────────────────────────────────────────
        st.markdown("**Background Video**")
        uploaded_bg = st.file_uploader(
            "Upload Video (optional)",
            type=["mp4", "mov", "mkv"],
            help="Upload a background video file. If provided it overrides the category picker below.",
        )

        bg_manager = BackgroundManager(PATHS.backgrounds_dir)
        # Save upload immediately so the new 'uploaded' category appears in the list
        if uploaded_bg is not None:
            _save_uploaded_background(uploaded_bg, PATHS.backgrounds_dir)

        categories = bg_manager.categories()
        if uploaded_bg is not None:
            default_cat = "uploaded"
        else:
            # Prefer the first category that actually contains video files
            non_empty = [c for c in categories if bg_manager.list_category_files(c)]
            default_cat = non_empty[0] if non_empty else (categories[0] if categories else "minecraft")

        cat_options = categories if categories else ["minecraft", "subway", "horror", "rain"]
        background_category = st.selectbox(
            "Background Category",
            options=cat_options,
            index=cat_options.index(default_cat) if default_cat in cat_options else 0,
        )

        st.divider()
        st.markdown("**Story Thumbnail**")
        enable_thumbnail = st.toggle("Generate thumbnail (AI image)", value=False)
        secrets = st.session_state["app_secrets"]
        saved_img = secrets["image"]
        if enable_thumbnail:
            img_keys = list(IMAGE_PROVIDERS.keys())
            if "image_provider" not in st.session_state:
                st.session_state["image_provider"] = (
                    saved_img["provider"] if saved_img["provider"] in img_keys else img_keys[0]
                )
            image_provider = st.selectbox(
                "Image Provider",
                options=img_keys,
                format_func=lambda k: IMAGE_PROVIDERS[k],
                key="image_provider",
            )
            if "image_api_key" not in st.session_state:
                st.session_state["image_api_key"] = saved_img["api_key"]
            image_api_key = st.text_input("Image API Key", type="password", key="image_api_key")
            if "image_model" not in st.session_state:
                st.session_state["image_model"] = saved_img["model"] or (
                    "imagen-3.0-generate-002" if image_provider == "gemini" else "gpt-image-1"
                )
            image_model = st.text_input("Image Model", key="image_model")
            image_base_url = ""
            if image_provider == "openai_compatible":
                if "image_base_url" not in st.session_state:
                    st.session_state["image_base_url"] = saved_img["base_url"]
                image_base_url = st.text_input(
                    "Image Base URL", key="image_base_url", placeholder="https://api.openai.com/v1"
                )
            if "image_size" not in st.session_state:
                st.session_state["image_size"] = saved_img["size"]
            image_size = st.text_input(
                "Image Size", key="image_size", help="e.g. 1024x1792 (vertical 9:16)."
            )
            secrets["image"] = {
                "provider": image_provider,
                "api_key": image_api_key,
                "model": image_model,
                "base_url": image_base_url,
                "size": image_size,
            }
            _save_secrets(secrets)
        else:
            image_provider = saved_img["provider"]
            image_api_key = saved_img["api_key"]
            image_model = saved_img["model"]
            image_base_url = saved_img["base_url"]
            image_size = saved_img["size"]

        st.divider()
        st.markdown("**YouTube Publishing**")
        enable_youtube = st.toggle("Publish to YouTube", value=False)

        youtube_client_secret = st.file_uploader(
            "client_secret.json (Google OAuth)",
            type=["json"],
            help="Google Cloud Console → APIs & Services → Credentials → OAuth 2.0 Client ID (Desktop app). Download the JSON and upload it here.",
        )
        if youtube_client_secret is not None:
            _save_youtube_client_secret(youtube_client_secret, PATHS.cache_dir)

        youtube_title = ""
        youtube_description = ""
        youtube_tags = "shorts, storytime, viral, fyp"
        youtube_category = "Entertainment"
        youtube_privacy = "private"
        youtube_connect_btn = False
        if enable_youtube:
            if script_mode == "Manual Script":
                youtube_title = st.text_input("Video Title", value="", placeholder="Defaults to <Story Title> - Part N")
                youtube_description = st.text_area(
                    "Video Description",
                    value="",
                    height=80,
                    placeholder="Defaults to the part caption + hashtags.",
                )
                youtube_tags = st.text_input("Tags (comma separated)", value="shorts, storytime, viral, fyp")
            else:
                st.caption("Title, description and hashtags will be auto-generated for each part.")
            col_cat, col_priv = st.columns(2)
            with col_cat:
                youtube_category = st.selectbox(
                    "Category",
                    options=list(_YT_CATEGORIES.keys()),
                    index=list(_YT_CATEGORIES.keys()).index("Entertainment"),
                )
            with col_priv:
                youtube_privacy = st.selectbox("Privacy", options=["private", "unlisted", "public"], index=0)

            # ── Multi-account management ─────────────────────────────────────
            st.markdown("**YouTube Accounts**")
            accounts = account_manager.list_accounts()
            if accounts:
                for acc in accounts:
                    kw = ", ".join(acc.keywords) if acc.keywords else "default (catches everything)"
                    conn = "connected" if acc.connected else "not connected"
                    info_col, del_col = st.columns([5, 1])
                    with info_col:
                        st.markdown(
                            f"**{acc.name}** — {acc.channel_title or 'unknown channel'} · "
                            f"keywords: `{kw}` · _{conn}_"
                        )
                    with del_col:
                        if st.button("Remove", key=f"del_{acc.name}"):
                            st.session_state["delete_account_name"] = acc.name

                    prompt_key = f"story_prompt_{acc.name}"
                    current_prompt = str(getattr(acc, "story_prompt", ""))
                    updated_prompt = st.text_area(
                        f"Generation prompt — {acc.name}",
                        value=current_prompt,
                        key=prompt_key,
                        height=220,
                        help="Used as the story topic/prompt for every new story routed to this account.",
                    )
                    if st.button("Save prompt", key=f"save_prompt_{acc.name}"):
                        account_dir = account_manager.accounts_dir / account_manager._slug(acc.name)
                        metadata_path = account_dir / "account.json"
                        metadata = load_json(metadata_path) if metadata_path.exists() else {}
                        metadata["story_prompt"] = updated_prompt.strip()
                        write_json(metadata_path, metadata)
                        st.success(f"Saved story prompt for {acc.name}.")
                        st.rerun()
            else:
                st.caption("No accounts connected yet.")

            with st.expander("➕ Connect a new YouTube account"):
                new_account_name = st.text_input(
                    "Account label", value="", placeholder="e.g. Horror Channel"
                )
                new_account_keywords = st.text_input(
                    "Topic keywords (comma separated)",
                    value="",
                    placeholder="e.g. horror, scary, ghost",
                    help="Stories whose title/topic contain any of these words are routed to this account. Leave blank to make it the default (catch-all) account.",
                )
                new_account_story_prompt = st.text_area(
                    "Story generation prompt",
                    value="",
                    placeholder="Describe the exact story type this account should always generate.",
                    help="The scheduler passes this prompt to the story generator for every story routed to this account.",
                )
                youtube_connect_btn = st.button("Connect this account")

        st.divider()
        st.markdown("**🤖 Autopilot (continuous uploads)**")
        autopilot_control = _read_autopilot_control(PATHS.cache_dir)
        autopilot_enabled = bool(autopilot_control.get("enabled", True))
        autopilot_running = autopilot_controller.is_running(PATHS.cache_dir)

        autopilot_toggle = st.toggle(
            "Enable Autopilot",
            value=autopilot_enabled,
            help=(
                "Turning this ON actually starts the autopilot scheduler in the "
                "background — it keeps generating + uploading stories on a schedule "
                "until you turn it OFF (or the machine/process is stopped). Turning it "
                "OFF pauses and stops the scheduler. Survives refreshing/closing this page."
            ),
        )
        if autopilot_toggle != autopilot_enabled:
            if autopilot_toggle:
                # Turn ON: ensure the scheduler process is actually running.
                _write_autopilot_control(PATHS.cache_dir, {"enabled": True})
                if not autopilot_running:
                    try:
                        pid = autopilot_controller.start(PATHS.cache_dir, BASE_DIR)
                        st.success(f"Autopilot started (PID {pid}).")
                    except Exception as exc:
                        st.error(f"Failed to start autopilot: {exc}")
                else:
                    st.info("Autopilot already running — resumed.")
            else:
                # Turn OFF: pause + terminate the scheduler process.
                _write_autopilot_control(PATHS.cache_dir, {"enabled": False})
                autopilot_controller.stop(PATHS.cache_dir)
                st.info("Autopilot paused and stopped.")
            st.rerun()

        if autopilot_running:
            st.caption(f"● Autopilot is **running** (PID {autopilot_controller.read_pid(PATHS.cache_dir)})")
        elif autopilot_enabled:
            st.warning(
                "⚠️ Autopilot is **enabled but the worker is not running** — "
                "it may have crashed or been stopped. Nothing is being generated."
            )
            if st.button("↻ Restart autopilot", key="restart_autopilot"):
                try:
                    pid = autopilot_controller.start(PATHS.cache_dir, BASE_DIR)
                    if pid:
                        st.success(f"Autopilot restarted (PID {pid}).")
                    else:
                        st.error("Could not start autopilot — see the logs.")
                except Exception as exc:
                    st.error(f"Failed to restart autopilot: {exc}")
                st.rerun()
        else:
            st.caption("○ Autopilot is **stopped**")

        # Auto voice: pick a narrator (male/female, deep/warm/…) per story.
        autopilot_config_path = PATHS.config_dir / "autopilot.json"
        _ap_cfg = _load_optional_json(autopilot_config_path)
        background_files = [
            path
            for category in bg_manager.categories()
            for path in bg_manager.list_category_files(category)
        ]
        background_options = [""] + [str(path) for path in background_files]
        saved_background = str(_ap_cfg.get("background_video") or "")
        if saved_background and saved_background not in background_options:
            background_options.append(saved_background)
        selected_background = st.selectbox(
            "Autopilot background video",
            options=background_options,
            index=background_options.index(saved_background),
            format_func=lambda value: "Use category clips" if not value else Path(value).name,
            help="This file must be stored on the machine running the app. It is saved for future Autopilot stories.",
        )
        if selected_background != saved_background:
            _ap_cfg["background_video"] = selected_background
            write_json(autopilot_config_path, _ap_cfg)

        ap_categories = [
            category for category in bg_manager.categories()
            if bg_manager.list_category_files(category)
        ]
        if ap_categories:
            saved_category = _ap_cfg.get("background_category", "minecraft")
            if saved_category not in ap_categories:
                saved_category = ap_categories[0]
            selected_category = st.selectbox(
                "Autopilot background category",
                options=ap_categories,
                index=ap_categories.index(saved_category),
                help="Used when 'Use category clips' is selected above.",
            )
            if selected_category != _ap_cfg.get("background_category"):
                _ap_cfg["background_category"] = selected_category
                write_json(autopilot_config_path, _ap_cfg)

        auto_voice_current = bool(_ap_cfg.get("auto_voice", True))
        auto_voice_toggle = st.toggle(
            "Auto voice (pick per story)",
            value=auto_voice_current,
            help=(
                "When ON, the autopilot asks the LLM for the best narrator "
                "(male/female, deep/warm/bright/…) for each story and picks a matching "
                "voice automatically. When OFF, it always uses the fixed voice set in "
                "config/autopilot.json. Takes effect on the next scheduler restart."
            ),
        )
        if auto_voice_toggle != auto_voice_current:
            _ap_cfg["auto_voice"] = auto_voice_toggle
            write_json(autopilot_config_path, _ap_cfg)

        _autopilot_status = _read_autopilot_state(PATHS.cache_dir)
        if _autopilot_status.get("last_run_iso"):
            st.caption(f"Last run: {_autopilot_status['last_run_iso']}")
        today_count = sum(
            1 for h in _autopilot_status.get("history", [])
            if h.get("date") == _today_str()
        )
        st.caption(f"Stories published today: {today_count}")

        st.divider()
        run_btn = st.button("Start Generation Queue", type="primary", use_container_width=True)

    if enable_youtube and youtube_connect_btn:
        if not new_account_name.strip():
            st.error("Give the account a label first.")
        else:
            with st.spinner("Opening your browser to authorize this account\u2026"):
                try:
                    keywords = [k.strip() for k in new_account_keywords.split(",") if k.strip()]
                    acc = account_manager.connect_account(new_account_name, keywords, new_account_story_prompt)
                    st.success(f"Connected {acc.name} ({acc.channel_title or 'unknown channel'}).")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Account connection failed: {exc}")

    if st.session_state.get("delete_account_name"):
        target = st.session_state.pop("delete_account_name")
        try:
            account_manager.delete_account(target)
            st.success(f"Removed account '{target}'.")
            st.rerun()
        except Exception as exc:
            st.error(f"Failed to remove '{target}': {exc}")

    # ── Status panel ──────────────────────────────────────────────────────────
    status_col, logs_col = st.columns([2, 1])

    with status_col:
        st.subheader("Runtime Status")
        ffmpeg_ok = has_ffmpeg_with_local_fallback(PATHS.base)
        if ffmpeg_ok:
            st.success("FFmpeg and FFprobe detected")
        else:
            st.error("FFmpeg is not available in PATH. Install FFmpeg before rendering.")

        if not categories:
            st.warning(
                "No background categories were found with files. "
                "Add mp4/mov/mkv clips under assets/backgrounds/<category> or upload a video above."
            )

    with logs_col:
        st.subheader("Log File")
        st.write(st.session_state["log_path"])

    # ── Stories & Uploads ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("📚 Stories & Uploads")

    autopilot_live = autopilot_controller.is_running(PATHS.cache_dir)
    active_story = story_uploader.read_active_story(PATHS.cache_dir)
    currently_creating = active_story if autopilot_live else None
    youtube_ready = account_manager.has_client_secret() and bool(account_manager.list_accounts())

    stories = story_uploader.list_stories(PATHS)
    if not stories:
        st.caption("No stories have been created yet.")
    else:
        if currently_creating:
            st.info(f"🔄 Currently creating: **{currently_creating}** — its upload is locked.")
        if not youtube_ready:
            st.warning("Connect a YouTube account in the sidebar to enable story uploads.")

        for s in stories:
            sid = s["story_id"]
            ups = story_uploader.upload_status(PATHS.cache_dir, sid)

            uploaded_idx = {p["index"] for p in s["parts"] if p["uploaded"]}
            rendered_idx = set(s["rendered_parts"])
            pending_parts = rendered_idx - uploaded_idx
            pending_full = s["full_rendered"] and not s["long_uploaded"]
            uploadable = bool(pending_parts) or pending_full

            if currently_creating == sid:
                badge = "🔄 Creating…"
            elif ups["running"]:
                badge = "⬆️ Uploading…"
            elif ups["state"] == "error":
                badge = "❌ Upload error"
            elif s["state"] == "in_progress":
                badge = "🛠 Incomplete — needs resume"
            elif uploadable:
                n_parts = len(pending_parts)
                badge = f"📦 {n_parts} part(s) ready to upload" + (" + full" if pending_full else "")
            elif s["parts_uploaded"] or s["long_uploaded"]:
                badge = "✅ Uploaded"
            else:
                badge = "📝 Draft (no videos rendered)"

            col_main, col_action = st.columns([4, 1])
            with col_main:
                st.markdown(f"**{sid}** — {s['title']}")
                st.caption(
                    f"{badge}  ·  account: {s['account'] or '—'}  ·  state: {s['state']}"
                )

            with col_action:
                if currently_creating == sid:
                    st.button("🔒 Locked", key=f"up_{sid}", disabled=True)
                elif ups["running"]:
                    st.button("⬆️ Uploading…", key=f"up_{sid}", disabled=True)
                elif s["state"] == "in_progress":
                    if st.button("🛠 Resume", key=f"resume_{sid}", type="primary"):
                        pid = autopilot_controller.start(PATHS.cache_dir, BASE_DIR)
                        if pid:
                            st.success("Resume task started. It will render missing videos and retry YouTube uploads.")
                        else:
                            st.error("Could not start the resume task. Check the server log.")
                        st.rerun()
                elif uploadable:
                    if st.button("⬆️ Upload", key=f"up_{sid}", type="primary"):
                        try:
                            pid = story_uploader.spawn_upload(BASE_DIR, PATHS.cache_dir, sid)
                            if pid:
                                st.success(f"Upload task started for {sid} (PID {pid}).")
                            else:
                                st.error(f"Could not start upload for {sid} — check the logs.")
                        except Exception as exc:
                            st.error(f"Failed to start upload: {exc}")
                        st.rerun()
                elif ups["state"] == "done":
                    st.button("✅ Done", key=f"up_{sid}", disabled=True)

            story_output = ensure_story_output_structure(PATHS, sid)
            watch_full = story_output.final / "full.mp4"
            watch_part = story_output.final / f"part1.mp4"
            watch_path = watch_full if watch_full.exists() else watch_part

            watch_col, delete_col = st.columns([4, 1])
            with watch_col:
                if watch_path.exists():
                    with st.popover("▶ Watch"):
                        st.video(str(watch_path))
                else:
                    st.button("▶ Watch", key=f"watch_{sid}", disabled=True)
            with delete_col:
                confirm_key = f"confirm_delete_{sid}"
                if st.session_state.get(confirm_key):
                    if st.button("Confirm delete", key=f"confirm_delete_btn_{sid}", type="primary"):
                        try:
                            story_uploader.delete_story(PATHS, sid)
                            st.session_state.pop(confirm_key, None)
                            st.success(f"Deleted {sid} and its related files.")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Could not delete {sid}: {exc}")
                    if st.button("Cancel", key=f"cancel_delete_{sid}"):
                        st.session_state.pop(confirm_key, None)
                        st.rerun()
                elif st.button("🗑 Delete", key=f"delete_{sid}"):
                    st.session_state[confirm_key] = True
                    st.rerun()

            with st.expander(f"Details — {sid}", expanded=False):
                if s["parts"]:
                    for p in s["parts"]:
                        mark = "✅" if p["uploaded"] else ("❌" if p["failed"] else "⬜")
                        st.write(f"{mark} Part {p['index']}  {p['url'] or ''}")
                if s["full_rendered"]:
                    mark = "✅" if s["long_uploaded"] else "⬜"
                    st.write(f"{mark} Full story  {s['long_url'] or ''}")
                if ups["videos"]:
                    st.caption("Uploaded this session:")
                    for u in ups["videos"]:
                        st.write(u)
                if ups["error"]:
                    st.error(f"Upload error: {ups['error']}")

    if not run_btn:
        return

    # ── Validation ────────────────────────────────────────────────────────────
    if not ffmpeg_ok:
        st.stop()

    if script_mode == "AI Generate":
        if not llm_api_key.strip():
            st.error("API key is required in AI Generate mode.")
            st.stop()
        if not topic.strip():
            st.error("Story topic is required.")
            st.stop()
    else:
        if not manual_script.strip():
            st.error("Please paste your script in the Manual Script box.")
            st.stop()

    queue = [{"batch_index": i + 1} for i in range(story_batches)]
    st.info(f"Queue created with {len(queue)} story job(s) | mode: {script_mode}")

    story_cfg = settings["story"]
    gemini_cfg = settings["gemini"]
    subtitle_cfg = settings["subtitle"]

    device, whisper_compute = _infer_whisper_device()
    render_cfg = _build_render_config(settings, prefer_gpu=True)  # Prefer NVENC GPU; falls back to CPU libx264 automatically
    # Video bitrate is always 20M (Original quality) from settings.json

    # Apply subtitle position override
    # For vertical shorts (1920px height), adjust margins proportionally
    _SUBTITLE_POS = {
        "Top":    (8, 60),    # ASS alignment 8 = top-center, 60px from top
        "Middle": (5, 0),     # ASS alignment 5 = middle-center, no margin
        "Bottom": (2, 60),    # ASS alignment 2 = bottom-center, 60px from bottom
    }
    _align, _margin = _SUBTITLE_POS.get(subtitle_position, (2, 60))
    render_cfg.subtitle_alignment = _align
    render_cfg.subtitle_margin_v = _margin
    render_cfg.subtitle_font_size = int(subtitle_font_size)
    print(f"[CONFIG] Caption: position={subtitle_position}, size={render_cfg.subtitle_font_size}px, alignment={_align}, margin={_margin}")

    # Only instantiate the LLM client + story generator when needed
    story_generator = (
        StoryGenerator(
            LLMClient(
                provider=llm_provider,
                api_key=llm_api_key.strip(),
                model=llm_model.strip(),
                base_url=llm_base_url.strip() or None,
                temperature=float(gemini_cfg.get("temperature", 0.9)),
            )
        )
        if script_mode == "AI Generate"
        else None
    )

    tts_engine = KokoroTTSEngine(
        model_path=PATHS.base / tts_cfg["model_path"],
        voices_path=PATHS.base / tts_cfg["voices_path"],
        model_url=tts_cfg["model_url"],
        voices_url=tts_cfg["voices_url"],
        language=tts_cfg["language"],
        speed=float(tts_cfg["speed"]),
    )

    subtitle_engine = SubtitleEngine(
        model_size=subtitle_cfg["model_size"],
        compute_type=whisper_compute,
        device=device,
    )

    ffmpeg_renderer = FFmpegRenderer(render_cfg)
    pipeline = VideoPipelineEngine(
        tts_engine=tts_engine,
        subtitle_engine=subtitle_engine,
        background_manager=bg_manager,
        ffmpeg_renderer=ffmpeg_renderer,
    )

    global_progress = st.progress(0, text="Starting queue")
    processed_stories = []

    # Pre-connect all YouTube accounts once (tokens are cached, so this only
    # opens the browser the first time an account is used).
    account_uploaders: dict[str, tuple[YouTubeAccount, YouTubeUploader]] = {}
    accounts: list[YouTubeAccount] = []
    if enable_youtube:
        if not account_manager.has_client_secret():
            st.error("YouTube is enabled but no client_secret.json was uploaded.")
            st.stop()
        accounts = account_manager.list_accounts()
        if not accounts:
            st.error("No YouTube account connected. Connect one in the sidebar, then rerun.")
            st.stop()
        for acc in accounts:
            uploader_for_acc = account_manager.get_uploader(acc)
            with st.spinner(f"Connecting to YouTube account '{acc.name}'…"):
                try:
                    uploader_for_acc.authenticate()
                except Exception as exc:
                    st.error(f"Authentication failed for '{acc.name}': {exc}")
                    st.stop()
            account_uploaders[acc.name] = (acc, uploader_for_acc)

    for idx, job in enumerate(queue, start=1):
        story_id = next_story_id(PATHS.output_dir)
        story_output = ensure_story_output_structure(PATHS, story_id)
        st.markdown(f"### Processing {story_id} ({idx}/{len(queue)})")

        step_status = st.status("Preparing story", expanded=True)
        try:
            # Duration-driven story planning.
            target_wpm = 145
            total_target_words = max(200, int(float(total_story_minutes) * target_wpm))
            words_per_part_target = max(80, int(total_target_words / max(1, shorts_per_story)))
            min_words = max(60, int(words_per_part_target * 0.9))
            max_words = max(min_words + 10, int(words_per_part_target * 1.1))

            # ── Resolve narrative + title ─────────────────────────────────────
            if script_mode == "AI Generate":
                assert story_generator is not None
                generated = story_generator.generate_story(
                    topic.strip(),
                    requested_parts=shorts_per_story,
                    target_total_words=total_target_words,
                )
                narrative = generated.narrative
                story_title = generated.title
                display_topic = topic.strip()
            else:
                narrative = manual_script.strip()
                story_title = manual_title.strip() or "Manual Story"
                display_topic = story_title

            (PATHS.stories_dir / f"{story_id}.txt").write_text(narrative + "\n", encoding="utf-8")

            parts = split_story_into_parts(
                narrative,
                min_words=min_words,
                max_words=max_words,
                max_parts=shorts_per_story,
            )
            if not parts:
                raise RuntimeError("No story parts were generated from the script.")

            # Ensure each short ends with the profile CTA.
            for part in parts:
                text = part.text.strip()
                if not text.lower().endswith("full story in profile."):
                    part.text = f"{text} Full story in profile."

            metadata_payload = _story_to_metadata(story_title, display_topic, parts)
            if script_mode == "AI Generate" and enable_youtube:
                step_status.update(label="Generating YouTube metadata\u2026", state="running")
                _enrich_parts_with_ai_youtube_metadata(
                    story_generator, story_title, display_topic, parts, metadata_payload
                )
            write_json(story_output.scripts / "metadata.json", metadata_payload)

            thumbnail_path: Path | None = None
            if enable_thumbnail:
                step_status.update(label="Generating story thumbnail\u2026", state="running")
                try:
                    image_gen = ImageGenerator(
                        provider=image_provider,
                        api_key=image_api_key.strip(),
                        model=image_model.strip(),
                        base_url=image_base_url.strip() or None,
                    )
                    if story_generator is not None:
                        thumb_prompt = story_generator.generate_thumbnail_prompt(
                            story_title, display_topic
                        )
                    else:
                        thumb_prompt = (
                            f"A dramatic vertical thumbnail for a story titled '{story_title}'."
                        )
                    image_bytes = image_gen.generate(thumb_prompt, size=image_size)
                    thumbnail_path = story_output.thumbnails / "cover.png"
                    thumbnail_path.write_bytes(image_bytes)
                except Exception as exc:
                    logger.exception("Thumbnail generation failed")
                    st.error(f"Thumbnail generation failed: {exc}")

            step_status.update(label=f"Script ready — {len(parts)} part(s)", state="complete")

            part_progress = st.progress(0, text=f"Rendering 0/{len(parts)} parts")
            artifacts = []
            
            # Determine the source background video
            source_bg_video = None
            if uploaded_bg is not None:
                # Use uploaded video as continuous source
                uploaded_path = PATHS.backgrounds_dir / "uploaded" / uploaded_bg.name
                if uploaded_path.exists():
                    source_bg_video = uploaded_path
            
            bg_position = 0.0  # Track position through video for continuous playback

            # ── YouTube publishing setup ─────────────────────────────────────
            youtube_uploader = None
            meta_by_index = {p["part_index"]: p for p in metadata_payload["parts"]}
            if enable_youtube:
                youtube_account = account_manager.resolve_account(
                    f"{story_title} {display_topic}", accounts
                )
                if youtube_account is None:
                    st.error("No YouTube account available for routing.")
                    st.stop()
                youtube_uploader = account_uploaders[youtube_account.name][1]
                st.info(
                    f"Routing to YouTube account **{youtube_account.name}** "
                    f"({youtube_account.channel_title or 'unknown channel'})"
                )

            for part in parts:
                artifact, bg_position = pipeline.process_part(
                    story_output=story_output,
                    part_index=part.index,
                    part_text=part.text,
                    voice=voice,
                    background_video=source_bg_video,
                    category=background_category if not source_bg_video else None,
                    enable_subtitles=subtitle_enabled,
                    quality=quality,
                    background_position=bg_position,
                )
                artifacts.append(artifact)
                part_progress.progress(
                    int((part.index / len(parts)) * 100),
                    text=f"Rendered {part.index}/{len(parts)} parts",
                )

                if youtube_uploader is not None:
                    try:
                        result = _publish_part_to_youtube(
                            youtube_uploader,
                            artifact=artifact,
                            part=part,
                            meta=meta_by_index.get(part.index),
                            story_title=story_title,
                            youtube_title=youtube_title,
                            youtube_description=youtube_description,
                            youtube_tags=youtube_tags,
                            youtube_category=youtube_category,
                            youtube_privacy=youtube_privacy,
                        )
                        if thumbnail_path is not None and thumbnail_path.exists():
                            youtube_uploader.set_thumbnail(result.video_id, thumbnail_path)
                    except Exception as exc:
                        logger.exception("YouTube upload failed for part %s", part.index)
                        st.error(f"YouTube upload failed for Part {part.index}: {exc}")

            long_artifact = None
            if long_video_enabled:
                step_status.update(label="Rendering full-story long video\u2026", state="running")
                try:
                    long_artifact = pipeline.process_full_story(
                        story_output=story_output,
                        narrative=narrative,
                        voice=voice,
                        background_video=source_bg_video,
                        category=background_category if not source_bg_video else None,
                        enable_subtitles=subtitle_enabled,
                        quality=quality,
                    )
                    if youtube_uploader is not None:
                        try:
                            long_title = story_title
                            long_description = ""
                            long_tags = [t.strip() for t in youtube_tags.split(",") if t.strip()]
                            if story_generator is not None:
                                md = story_generator.generate_youtube_metadata(
                                    story_title, display_topic, 0, "Full Story", narrative
                                )
                                long_title = md.title
                                long_description = md.description
                                long_tags = md.tags
                            long_result = youtube_uploader.upload_video(
                                video_path=long_artifact.final_video_path,
                                title=long_title,
                                description=long_description or f"{display_topic} — full story.",
                                tags=long_tags,
                                category_id=_YT_CATEGORIES.get(youtube_category, "24"),
                                privacy_status=youtube_privacy,
                            )
                            if thumbnail_path is not None and thumbnail_path.exists():
                                youtube_uploader.set_thumbnail(
                                    long_result.video_id, thumbnail_path
                                )
                            st.success(f"Uploaded full story to YouTube: {long_result.video_url}")
                        except Exception as exc:
                            logger.exception("YouTube upload failed for full story")
                            st.error(f"YouTube upload failed for full story: {exc}")
                except Exception as exc:
                    logger.exception("Long video render failed")
                    st.error(f"Long video render failed: {exc}")

            processed_stories.append((story_id, story_title, story_output, metadata_payload, artifacts, long_artifact, thumbnail_path))
            logger.info("Completed story %s", story_id)
        except Exception as exc:
            logger.exception("Story job failed: %s", exc)
            step_status.update(label=f"Error: {exc}", state="error")
            st.error(f"Story job failed for {story_id}: {exc}")
        finally:
            force_memory_cleanup()

        global_progress.progress(int((idx / len(queue)) * 100), text=f"Completed {idx}/{len(queue)} stories")

    st.success("Queue finished")

    for story_id, title, out_paths, metadata, artifacts, long_artifact, thumbnail_path in processed_stories:
        with st.expander(f"{story_id} | {title}", expanded=False):
            st.subheader("Scripts")
            for part in metadata["parts"]:
                st.markdown(f"**Part {part['part_index']}**")
                st.write(part["script"])
                st.caption(f"Caption: {part['caption']}")
                st.caption(" ".join(part["hashtags"]))

            st.subheader("Audio Preview")
            for artifact in artifacts:
                st.markdown(f"Part {artifact.index}")
                st.audio(str(artifact.audio_path))

            st.subheader("Video Preview")
            for artifact in artifacts:
                st.markdown(f"Part {artifact.index}")
                st.video(str(artifact.final_video_path))

            if long_artifact is not None:
                st.subheader("Long Video (Full Story)")
                st.video(str(long_artifact.final_video_path))

            if thumbnail_path is not None and thumbnail_path.exists():
                st.subheader("Thumbnail")
                st.image(str(thumbnail_path))

            st.subheader("Paths")
            st.write(str(out_paths.root))

    log_path = Path(st.session_state["log_path"])
    if log_path.exists():
        st.subheader("Recent Logs")
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-80:]
        st.code("\n".join(lines) if lines else "No logs yet", language="text")


if __name__ == "__main__":
    render_app()
