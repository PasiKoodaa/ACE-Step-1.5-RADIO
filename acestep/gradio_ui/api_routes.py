"""
Gradio API Routes Module
Add API endpoints compatible with api_server.py and CustomAceStep to Gradio application
"""
import json
import os
import random
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, Depends, Header
from fastapi.responses import FileResponse

# Global results directory inside project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_RESULTS_DIR = os.path.join(PROJECT_ROOT, "gradio_outputs").replace("\\", "/")
os.makedirs(DEFAULT_RESULTS_DIR, exist_ok=True)

# API Key storage (set via setup_api_routes)
_api_key: Optional[str] = None


def set_api_key(key: Optional[str]):
    """Set the API key for authentication"""
    global _api_key
    _api_key = key


def _wrap_response(data: Any, code: int = 200, error: Optional[str] = None) -> Dict[str, Any]:
    """Wrap response data in standard format compatible with CustomAceStep."""
    return {
        "data": data,
        "code": code,
        "error": error,
        "timestamp": int(time.time() * 1000),
        "extra": None,
    }


def verify_token_from_request(body: dict, authorization: Optional[str] = None) -> Optional[str]:
    """
    Verify API key from request body (ai_token) or Authorization header.
    Returns the token if valid, None if no auth required.
    """
    if _api_key is None:
        return None  # No auth required

    # Try ai_token from body first
    ai_token = body.get("ai_token") if body else None
    if ai_token:
        if ai_token == _api_key:
            return ai_token
        raise HTTPException(status_code=401, detail="Invalid ai_token")

    # Fallback to Authorization header
    if authorization:
        if authorization.startswith("Bearer "):
            token = authorization[7:]
        else:
            token = authorization
        if token == _api_key:
            return token
        raise HTTPException(status_code=401, detail="Invalid API key")

    # No token provided but auth is required
    raise HTTPException(status_code=401, detail="Missing ai_token or Authorization header")


async def verify_api_key(authorization: Optional[str] = Header(None)):
    """Verify API key from Authorization header (legacy, for non-body endpoints)"""
    if _api_key is None:
        return  # No auth required

    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    # Support "Bearer <key>" format
    if authorization.startswith("Bearer "):
        token = authorization[7:]
    else:
        token = authorization

    if token != _api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


# Use diskcache to store results
try:
    import diskcache
    _cache_dir = os.path.join(os.path.dirname(__file__), ".cache", "api_results")
    os.makedirs(_cache_dir, exist_ok=True)
    _result_cache = diskcache.Cache(_cache_dir)
    DISKCACHE_AVAILABLE = True
except ImportError:
    _result_cache = {}
    DISKCACHE_AVAILABLE = False

RESULT_EXPIRE_SECONDS = 7 * 24 * 60 * 60  # 7 days expiration
RESULT_KEY_PREFIX = "ace_step_v1.5_"
SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}

# =============================================================================
# Example Data for Random Sample
# =============================================================================

def _get_project_root() -> str:
    """Get project root directory"""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_all_examples(sample_mode: str = "simple_mode") -> List[Dict[str, Any]]:
    """Load all example JSON files from examples directory"""
    project_root = _get_project_root()
    if sample_mode == "simple_mode":
        examples_dir = os.path.join(project_root, "examples", "simple_mode")
    else:
        examples_dir = os.path.join(project_root, "examples", "text2music")

    if not os.path.isdir(examples_dir):
        return []

    all_examples = []
    for filename in os.listdir(examples_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(examples_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_examples.extend(data)
                    elif isinstance(data, dict):
                        all_examples.append(data)
            except Exception:
                pass
    return all_examples


# Pre-load example data
SIMPLE_EXAMPLE_DATA = _load_all_examples("simple_mode")
CUSTOM_EXAMPLE_DATA = _load_all_examples("custom_mode")


def store_result(task_id: str, result: dict, status: str = "succeeded"):
    """Store result to diskcache"""
    data = {
        "result": result,
        "created_at": time.time(),
        "status": status
    }
    key = f"{RESULT_KEY_PREFIX}{task_id}"
    if DISKCACHE_AVAILABLE:
        _result_cache.set(key, data, expire=RESULT_EXPIRE_SECONDS)
    else:
        _result_cache[key] = data


def get_result(task_id: str) -> Optional[dict]:
    """Get result from diskcache"""
    key = f"{RESULT_KEY_PREFIX}{task_id}"
    if DISKCACHE_AVAILABLE:
        return _result_cache.get(key)
    else:
        return _result_cache.get(key)


router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return _wrap_response({
        "status": "ok",
        "service": "ACE-Step Gradio API",
        "version": "1.0",
    })


def _collect_audio_files(folder: str, recursive: bool, limit: int, extensions: Optional[set[str]] = None) -> List[str]:
    """Collect audio file paths from a folder."""
    if not folder:
        raise HTTPException(status_code=400, detail="Missing folder parameter")

    resolved = os.path.realpath(folder)
    allowed_exts = extensions or SUPPORTED_AUDIO_EXTENSIONS
    # Be permissive: if caller passes a single audio file path, return it.
    if os.path.isfile(resolved):
        ext = os.path.splitext(resolved)[1].lower()
        if ext in allowed_exts:
            return [resolved.replace("\\", "/")]
        return []

    if not os.path.isdir(resolved):
        raise HTTPException(status_code=400, detail=f"Folder not found: {folder}")

    files: List[str] = []
    if recursive:
        walker = os.walk(resolved)
        for root, _, names in walker:
            for name in names:
                ext = os.path.splitext(name)[1].lower()
                if ext in allowed_exts:
                    files.append(os.path.join(root, name).replace("\\", "/"))
                    if len(files) >= limit:
                        return files
    else:
        for name in os.listdir(resolved):
            path = os.path.join(resolved, name)
            if not os.path.isfile(path):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext in allowed_exts:
                files.append(path.replace("\\", "/"))
                if len(files) >= limit:
                    return files

    return files


def _cleanup_lyrics_text(text: Optional[str]) -> Optional[str]:
    """Normalize lyrics text and strip common LRC timestamps."""
    if not text:
        return None
    cleaned_lines: List[str] = []
    for raw in str(text).splitlines():
        line = raw.strip()
        while line.startswith("[") and "]" in line:
            head = line[1:line.index("]")]
            if ":" not in head:
                break
            line = line[line.index("]") + 1:].lstrip()
        if line:
            cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines).strip()
    return cleaned or None


def _tag_value_to_text(value: Any) -> Optional[str]:
    """Convert mutagen tag value/frame into plain text."""
    if value is None:
        return None
    if hasattr(value, "text"):
        frame_text = getattr(value, "text")
        if isinstance(frame_text, list):
            return " ".join(str(x) for x in frame_text if str(x).strip()).strip() or None
        return str(frame_text).strip() or None
    if isinstance(value, list):
        return " ".join(str(x) for x in value if str(x).strip()).strip() or None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip() or None
    return str(value).strip() or None


def _extract_audio_info(audio_path: str, debug_tags: bool = False) -> Dict[str, Any]:
    """Extract title/artist/lyrics from sidecar files and embedded tags."""
    if not audio_path:
        raise HTTPException(status_code=400, detail="Missing audio_path parameter")

    resolved = os.path.realpath(audio_path)
    if not os.path.isfile(resolved):
        raise HTTPException(status_code=400, detail=f"Audio file not found: {audio_path}")

    file_stem = os.path.splitext(os.path.basename(resolved))[0]
    title: Optional[str] = None
    artist: Optional[str] = None
    lyrics: Optional[str] = None
    lyrics_source: Optional[str] = None
    debug: Dict[str, Any] = {"library": "mutagen", "tag_items": []} if debug_tags else {}

    base, _ = os.path.splitext(resolved)
    sidecars = [f"{base}.lrc", f"{base}.txt"]
    for sidecar in sidecars:
        if os.path.isfile(sidecar):
            try:
                with open(sidecar, "r", encoding="utf-8") as f:
                    text = _cleanup_lyrics_text(f.read())
                if text:
                    lyrics = text
                    lyrics_source = os.path.basename(sidecar)
                    break
            except Exception:
                pass

    # Embedded tags (best-effort, optional dependency)
    try:
        from mutagen import File as MutagenFile  # type: ignore

        m = MutagenFile(resolved)
        if m and getattr(m, "tags", None):
            tags = m.tags
            if debug_tags:
                for key, value in tags.items():
                    key_str = str(key)
                    desc = str(getattr(value, "desc", ""))
                    val_text = _tag_value_to_text(value) or ""
                    preview = val_text[:300] + ("..." if len(val_text) > 300 else "")
                    debug["tag_items"].append({
                        "key": key_str,
                        "desc": desc,
                        "value_preview": preview,
                    })
            for key in ("TIT2", "\u00a9nam", "TITLE", "title"):
                val = _tag_value_to_text(tags.get(key))
                if val:
                    title = val
                    break

            for key in ("TPE1", "\u00a9ART", "ARTIST", "artist", "ALBUMARTIST", "albumartist"):
                val = _tag_value_to_text(tags.get(key))
                if val:
                    artist = val
                    break

            if not lyrics:
                for key in (
                    "USLT::eng", "USLT::XXX", "USLT",
                    "ULST::eng", "ULST::XXX", "ULST",
                    "LYRICS", "lyrics", "\u00a9lyr",
                ):
                    val = _cleanup_lyrics_text(_tag_value_to_text(tags.get(key)))
                    if val:
                        lyrics = val
                        lyrics_source = f"embedded:{key}"
                        break
            if not lyrics:
                for key, value in tags.items():
                    key_str = str(key)
                    key_lower = key_str.lower()
                    desc = str(getattr(value, "desc", "")).lower()
                    if any(tok in key_lower for tok in ("lyr", "lyric", "uslt", "ulst", "sylt")) or any(
                        tok in desc for tok in ("lyr", "lyric")
                    ):
                        val = _cleanup_lyrics_text(_tag_value_to_text(value))
                        if val:
                            lyrics = val
                            lyrics_source = f"embedded:{key_str}"
                            break
    except Exception as e:
        if debug_tags:
            debug["error"] = f"mutagen parse failed: {type(e).__name__}: {e}"

    if not title:
        title = file_stem

    out = {
        "title": title,
        "artist": artist,
        "lyrics": lyrics,
        "lyrics_source": lyrics_source,
        "file_name": os.path.basename(resolved),
    }
    if debug_tags:
        out["tag_debug"] = debug
    return out


@router.get("/v1/audio-files")
async def list_audio_files(
    folder: str,
    recursive: bool = True,
    limit: int = 1000,
    extensions: str = "",
    _: None = Depends(verify_api_key),
):
    """List audio files from a server-side folder."""
    allowed_exts = SUPPORTED_AUDIO_EXTENSIONS
    if extensions.strip():
        allowed_exts = {
            (ext.strip().lower() if ext.strip().startswith(".") else f".{ext.strip().lower()}")
            for ext in extensions.split(",")
            if ext.strip()
        } or SUPPORTED_AUDIO_EXTENSIONS
    safe_limit = max(1, min(limit, 5000))
    files = _collect_audio_files(
        folder=folder,
        recursive=recursive,
        limit=safe_limit,
        extensions=allowed_exts,
    )
    return _wrap_response({
        "folder": os.path.realpath(folder).replace("\\", "/"),
        "count": len(files),
        "files": files,
    })


@router.get("/v1/audio-info")
async def get_audio_info(
    audio_path: str,
    debug_tags: bool = False,
    _: None = Depends(verify_api_key),
):
    """Extract title/artist/lyrics from sidecar files and embedded tags."""
    data = _extract_audio_info(audio_path, debug_tags=debug_tags)
    resp = {
        "audio_path": os.path.realpath(audio_path).replace("\\", "/"),
        "title": data["title"],
        "artist": data["artist"],
        "lyrics": data["lyrics"],
        "lyrics_source": data["lyrics_source"],
        "file_name": data["file_name"],
    }
    if debug_tags:
        resp["tag_debug"] = data.get("tag_debug", {})
    return _wrap_response(resp)


@router.get("/v1/models")
async def list_models(request: Request, _: None = Depends(verify_api_key)):
    """List available DiT models"""
    dit_handler = request.app.state.dit_handler

    models = []
    if dit_handler and dit_handler.model is not None:
        # Get current loaded model name
        config_path = getattr(dit_handler, 'config_path', '') or ''
        model_name = os.path.basename(config_path.rstrip("/\\")) if config_path else "unknown"
        models.append({
            "name": model_name,
            "is_default": True,
        })

    return _wrap_response({
        "models": models,
        "default_model": models[0]["name"] if models else None,
    })


@router.get("/v1/audio")
async def get_audio(path: str, _: None = Depends(verify_api_key)):
    """Download audio file"""
    # Security: Validate path is within allowed directory to prevent path traversal
    resolved_path = os.path.realpath(path)
    allowed_dir = os.path.realpath(DEFAULT_RESULTS_DIR)
    if not resolved_path.startswith(allowed_dir + os.sep) and resolved_path != allowed_dir:
        raise HTTPException(status_code=403, detail="Access denied: path outside allowed directory")
    if not os.path.exists(resolved_path):
        raise HTTPException(status_code=404, detail="Audio file not found")

    ext = os.path.splitext(resolved_path)[1].lower()
    media_types = {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
    }
    media_type = media_types.get(ext, "audio/mpeg")

    return FileResponse(resolved_path, media_type=media_type)


@router.post("/create_random_sample")
async def create_random_sample(request: Request, authorization: Optional[str] = Header(None)):
    """Get random sample parameters from pre-loaded example data"""
    content_type = (request.headers.get("content-type") or "").lower()

    if "json" in content_type:
        body = await request.json()
    else:
        form = await request.form()
        body = {k: v for k, v in form.items()}

    verify_token_from_request(body, authorization)
    sample_type = body.get("sample_type", "simple_mode") or "simple_mode"

    if sample_type == "simple_mode":
        example_data = SIMPLE_EXAMPLE_DATA
    else:
        example_data = CUSTOM_EXAMPLE_DATA

    if not example_data:
        return _wrap_response(None, code=500, error="No example data available")

    random_example = random.choice(example_data)
    return _wrap_response(random_example)


@router.post("/query_result")
async def query_result(request: Request, authorization: Optional[str] = Header(None)):
    """Batch query task results"""
    content_type = (request.headers.get("content-type") or "").lower()

    if "json" in content_type:
        body = await request.json()
    else:
        form = await request.form()
        body = {k: v for k, v in form.items()}

    verify_token_from_request(body, authorization)
    task_ids = body.get("task_id_list", [])

    if isinstance(task_ids, str):
        try:
            task_ids = json.loads(task_ids)
        except Exception:
            task_ids = []

    results = []
    for task_id in task_ids:
        data = get_result(task_id)
        if data and data.get("status") == "succeeded":
            results.append({
                "task_id": task_id,
                "status": 1,
                "result": json.dumps(data["result"], ensure_ascii=False)
            })
        else:
            results.append({
                "task_id": task_id,
                "status": 0,
                "result": "[]"
            })

    return _wrap_response(results)


@router.post("/format_input")
async def format_input(request: Request, authorization: Optional[str] = Header(None)):
    """Format and enhance lyrics/caption via LLM"""
    llm_handler = request.app.state.llm_handler

    if not llm_handler or not llm_handler.llm_initialized:
        return _wrap_response(None, code=500, error="LLM not initialized")

    content_type = (request.headers.get("content-type") or "").lower()
    if "json" in content_type:
        body = await request.json()
    else:
        form = await request.form()
        body = {k: v for k, v in form.items()}

    verify_token_from_request(body, authorization)

    caption = body.get("prompt", "") or ""
    lyrics = body.get("lyrics", "") or ""
    temperature = float(body.get("temperature", 0.85))

    from acestep.inference import format_sample

    try:
        result = format_sample(
            llm_handler=llm_handler,
            caption=caption,
            lyrics=lyrics,
            temperature=temperature,
            use_constrained_decoding=True,
        )

        if not result.success:
            return _wrap_response(None, code=500, error=result.status_message)

        return _wrap_response({
            "caption": result.caption or caption,
            "lyrics": result.lyrics or lyrics,
            "bpm": result.bpm,
            "key_scale": result.keyscale,
            "time_signature": result.timesignature,
            "duration": result.duration,
            "vocal_language": result.language or "unknown",
        })
    except Exception as e:
        return _wrap_response(None, code=500, error=str(e))


@router.post("/release_task")
async def release_task(request: Request, authorization: Optional[str] = Header(None)):
    """Create music generation task"""
    dit_handler = request.app.state.dit_handler
    llm_handler = request.app.state.llm_handler

    if not dit_handler or dit_handler.model is None:
        raise HTTPException(status_code=500, detail="DiT model not initialized")

    content_type = (request.headers.get("content-type") or "").lower()
    if "json" in content_type:
        body = await request.json()
    else:
        form = await request.form()
        body = {k: v for k, v in form.items()}

    verify_token_from_request(body, authorization)
    task_id = str(uuid4())

    from acestep.inference import generate_music, GenerationParams, GenerationConfig, create_sample, format_sample

    # Parse param_obj if provided
    param_obj = body.get("param_obj", {})
    if isinstance(param_obj, str):
        try:
            param_obj = json.loads(param_obj)
        except Exception:
            param_obj = {}

    # Helper to get param with aliases
    def get_param(key, *aliases, default=None):
        for k in [key] + list(aliases):
            if k in body and body[k] is not None:
                return body[k]
            if k in param_obj and param_obj[k] is not None:
                return param_obj[k]
        return default

    def to_bool(val, default=False):
        if val is None:
            return default
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in ("true", "1", "yes")
        return bool(val)

    try:
        # Get sample_mode and sample_query parameters
        sample_mode = to_bool(get_param("sample_mode", "sampleMode"), False)
        sample_query = get_param("sample_query", "sampleQuery", "description", "desc", default="") or ""
        use_format = to_bool(get_param("use_format", "useFormat"), False)
        has_sample_query = bool(sample_query and sample_query.strip())

        # Get base parameters
        caption = get_param("prompt", "caption", default="") or ""
        lyrics = get_param("lyrics", default="") or ""
        vocal_language = get_param("vocal_language", "language", default="en") or "en"
        lm_temperature = float(get_param("lm_temperature", "temperature", default=0.85) or 0.85)

        # Process sample_mode: use LLM to auto-generate caption/lyrics/metas
        if sample_mode or has_sample_query:
            if not llm_handler or not llm_handler.llm_initialized:
                raise HTTPException(status_code=500, detail="sample_mode requires LLM to be initialized")

            query = sample_query if has_sample_query else "NO USER INPUT"
            sample_result = create_sample(
                llm_handler=llm_handler,
                query=query,
                vocal_language=vocal_language if vocal_language not in ("en", "unknown", "") else None,
                temperature=lm_temperature,
            )

            if not sample_result.success:
                raise HTTPException(status_code=500, detail=sample_result.error or sample_result.status_message)

            # Use generated values
            caption = sample_result.caption or caption
            lyrics = sample_result.lyrics or lyrics
            # Override metas from sample result if available
            sample_bpm = sample_result.bpm
            sample_duration = sample_result.duration
            sample_keyscale = sample_result.keyscale
            sample_timesignature = sample_result.timesignature
            sample_language = sample_result.language or vocal_language
        else:
            sample_bpm = None
            sample_duration = None
            sample_keyscale = None
            sample_timesignature = None
            sample_language = vocal_language

        # Process use_format: enhance caption/lyrics via LLM
        if use_format and not sample_mode and not has_sample_query:
            if llm_handler and llm_handler.llm_initialized:
                format_result = format_sample(
                    llm_handler=llm_handler,
                    caption=caption,
                    lyrics=lyrics,
                    temperature=lm_temperature,
                )
                if format_result.success:
                    caption = format_result.caption or caption
                    lyrics = format_result.lyrics or lyrics
                    if format_result.bpm:
                        sample_bpm = format_result.bpm
                    if format_result.duration:
                        sample_duration = format_result.duration
                    if format_result.keyscale:
                        sample_keyscale = format_result.keyscale
                    if format_result.timesignature:
                        sample_timesignature = format_result.timesignature
                    if format_result.language:
                        sample_language = format_result.language

        # Build generation params with alias support
        params = GenerationParams(
            task_type=get_param("task_type", default="text2music"),
            caption=caption,
            lyrics=lyrics,
            bpm=sample_bpm or get_param("bpm"),
            keyscale=sample_keyscale or get_param("key_scale", "keyscale", "key", default=""),
            timesignature=sample_timesignature or get_param("time_signature", "timesignature", default=""),
            duration=sample_duration or get_param("audio_duration", "duration", default=-1),
            vocal_language=sample_language,
            inference_steps=get_param("inference_steps", default=8),
            guidance_scale=float(get_param("guidance_scale", default=7.0) or 7.0),
            seed=int(get_param("seed", default=-1) or -1),
            thinking=to_bool(get_param("thinking"), False),
            lm_temperature=lm_temperature,
            lm_cfg_scale=float(get_param("lm_cfg_scale", default=2.0) or 2.0),
            lm_negative_prompt=get_param("lm_negative_prompt", default="NO USER INPUT") or "NO USER INPUT",
        )

        config = GenerationConfig(
            batch_size=get_param("batch_size", default=2),
            use_random_seed=get_param("use_random_seed", default=True),
            audio_format=get_param("audio_format", default="mp3"),
        )

        # Get output directory
        save_dir = os.path.join(DEFAULT_RESULTS_DIR, f"api_{int(time.time())}").replace("\\", "/")
        os.makedirs(save_dir, exist_ok=True)

        # Call generation function
        result = generate_music(
            dit_handler=dit_handler,
            llm_handler=llm_handler if llm_handler and llm_handler.llm_initialized else None,
            params=params,
            config=config,
            save_dir=save_dir,
        )

        if not result.success:
            raise HTTPException(status_code=500, detail=result.error or result.status_message)

        # Extract audio paths
        audio_paths = [a["path"] for a in result.audios if a.get("path")]

        # Build result data with download URLs
        from urllib.parse import urlencode
        result_data = [{
            "file": p,
            "url": f"/v1/audio?{urlencode({'path': p})}",
            "status": 1,
            "create_time": int(time.time()),
        } for p in audio_paths]

        # Store result
        store_result(task_id, result_data)

        return _wrap_response({"task_id": task_id, "status": "succeeded"})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def setup_api_routes_to_app(app, dit_handler, llm_handler, api_key: Optional[str] = None):
    """
    Mount API routes to a FastAPI application (for use with gr.mount_gradio_app)

    Args:
        app: FastAPI application instance
        dit_handler: DiT handler
        llm_handler: LLM handler
        api_key: Optional API key for authentication
    """
    set_api_key(api_key)
    app.state.dit_handler = dit_handler
    app.state.llm_handler = llm_handler
    app.include_router(router)


def setup_api_routes(demo, dit_handler, llm_handler, api_key: Optional[str] = None):
    """
    Mount API routes to Gradio application

    Args:
        demo: Gradio Blocks instance
        dit_handler: DiT handler
        llm_handler: LLM handler
        api_key: Optional API key for authentication
    """
    set_api_key(api_key)
    app = demo.app
    app.state.dit_handler = dit_handler
    app.state.llm_handler = llm_handler
    app.include_router(router)

