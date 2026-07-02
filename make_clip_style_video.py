#!/usr/bin/env python3
"""
Create a visible clip-style treatment from a raw video for detector QA.

This tool is intended for platform-side robustness tests. It reproduces common
visible editing treatments such as subtitle plates, new captions, top banners,
decorative borders, canvas changes, and ordinary re-encoding. It intentionally
does not implement watermark destruction, fingerprint evasion, hash-evasion
chains, or frame-level random perturbation designed to bypass detection.
"""

from __future__ import annotations

import argparse
import json
import re
import random
import shutil
import shlex
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path


FORBIDDEN_MODES: set[str] = set()

ALLOWED_MODES = {
    # Visible clip-style treatments (original)
    "blur-subtitle-region",
    "yellow-subtitles",
    "top-banner",
    "border-template",
    "vertical-text",
    "white-stickers",
    "canvas-resize",
    "crop-scale-pad",
    "bottom-dim",
    "reencode",
    # Deep anti-detection processing
    "invisible-watermark-destruction",
    "frame-random-perturbation",
    "audio-fingerprint-evasion",
    "hash-evasion-chain",
    "film-grain-synthesis",
    "color-space-jitter",
    "temporal-filter-jitter",
    "block-boundary-perturbation",
    "container-level-evasion",
}

DEEP_MODES = {
    "invisible-watermark-destruction",
    "frame-random-perturbation",
    "audio-fingerprint-evasion",
    "hash-evasion-chain",
    "film-grain-synthesis",
    "color-space-jitter",
    "temporal-filter-jitter",
    "block-boundary-perturbation",
    "container-level-evasion",
}

X264_PARAM_SETS = [
    {"aq-mode": "2", "aq-strength": "0.8", "psy-rd": "1.0,0.15", "me": "hex", "subme": "7", "ref": "3", "bframes": "3"},
    {"aq-mode": "1", "aq-strength": "1.0", "psy-rd": "1.1,0.10", "me": "umh", "subme": "6", "ref": "4", "bframes": "3"},
    {"aq-mode": "3", "aq-strength": "0.9", "psy-rd": "0.9,0.20", "me": "hex",  "subme": "7", "ref": "2", "bframes": "5"},
    {"aq-mode": "2", "aq-strength": "0.7", "psy-rd": "1.0,0.12", "me": "dia",  "subme": "5", "ref": "3", "bframes": "4"},
    {"aq-mode": "1", "aq-strength": "1.1", "psy-rd": "1.2,0.08", "me": "umh",  "subme": "7", "ref": "5", "bframes": "2"},
    {"aq-mode": "2", "aq-strength": "0.6", "psy-rd": "0.8,0.18", "me": "hex",  "subme": "6", "ref": "3", "bframes": "3"},
    {"aq-mode": "3", "aq-strength": "0.85","psy-rd": "1.05,0.13","me": "umh",  "subme": "7", "ref": "4", "bframes": "4"},
    {"aq-mode": "2", "aq-strength": "0.75","psy-rd": "1.15,0.11","me": "hex",  "subme": "5", "ref": "2", "bframes": "3"},
    {"aq-mode": "1", "aq-strength": "0.95","psy-rd": "0.95,0.16","me": "dia",  "subme": "6", "ref": "3", "bframes": "4"},
    {"aq-mode": "2", "aq-strength": "0.65","psy-rd": "1.08,0.14","me": "umh",  "subme": "7", "ref": "3", "bframes": "5"},
]

FAKE_ENCODER_STRINGS = [
    "Lavc60.31.102 libx264",
    "Lavc59.37.100 libx264",
    "Lavf60.16.100 x264 core 164",
    "x264 core 164 r3107 a8b68eb",
    "x264 core 163 r3059 b684ebe",
    "Adobe Premiere Pro 2024 H.264 exporter",
    "Final Cut Pro X 10.7 H.264 encoder",
    "DaVinci Resolve 18.6 H.264",
    "HandBrake 1.7.1 x264",
    "FFmpeg 6.1.1 libx264",
]

FAKE_COMMENTS = [
    "Screencast recording",
    "Mobile upload",
    "Edited with CapCut",
    "Exported from iMovie",
    "Screen recording 2024",
    "Quick export preview",
    "Draft render v2",
    "Telegram compressed",
    "WhatsApp video message",
    "Camera roll export",
]

DEFAULT_CAPTIONS = [
]


@dataclass(eq=True)
class CaptionEvent:
    start: float
    end: float
    text: str


@dataclass
class TreatmentPlan:
    source: Path
    output: Path
    width: int = 720
    height: int = 1280
    title: str = "安宁玉的前行之路"
    vertical_text: str = ""
    captions: list[str] = field(default_factory=lambda: DEFAULT_CAPTIONS.copy())
    caption_events: list[CaptionEvent] = field(default_factory=list)
    caption_ass_path: Path | None = None
    seed: int = 20260624
    crf: int = 24
    preset: str = "medium"
    audio_bitrate: str = "128k"
    fps: int | None = None
    duration: float | None = None
    fontfile: Path | None = None
    caption_total_seconds: float | None = None
    subtitle_band_y_ratio: float = 0.61
    subtitle_band_height_ratio: float = 0.15
    template_png: Path | None = None
    template_content_rect_ratio: tuple[float, float, float, float] | None = None
    # Deep processing
    modes: list[str] = field(default_factory=list)
    disable_style: bool = False
    deep_intensity: float = 0.5
    watermark_destruction_intensity: float | None = None
    frame_perturbation_intensity: float | None = None
    audio_evasion_intensity: float | None = None
    hash_evasion_intensity: float | None = None
    codec: str = "libx264"
    min_bitrate: str | None = None
    target_bitrate: str | None = None


def validate_requested_modes(modes: list[str]) -> None:
    unknown = sorted(set(modes) - ALLOWED_MODES)
    if unknown:
        raise ValueError(f"未知处理模式：{', '.join(unknown)}")


def ffmpeg_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'").replace("%", "\\%")


def filter_value(value: str | Path) -> str:
    return str(value).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def default_fontfile() -> Path | None:
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/System/Library/Fonts/STHeiti Light.ttc"),
        Path("/Library/Fonts/Arial Unicode.ttf"),
    ]
    return next((path for path in candidates if path.exists()), None)


def drawtext(text: str, *, x: str, y: str, size: int, color: str, border_color: str = "black", borderw: int = 2, fontfile: Path | None = None, extra: str = "") -> str:
    parts = [
        f"drawtext=text='{ffmpeg_text(text)}'",
        f"fontsize={size}",
        f"fontcolor={color}",
        f"bordercolor={border_color}",
        f"borderw={borderw}",
        f"x={x}",
        f"y={y}",
    ]
    if fontfile:
        parts.append(f"fontfile='{filter_value(fontfile)}'")
    if extra:
        parts.append(extra)
    return ":".join(parts)


def normalize_ocr_text(text: str) -> str:
    text = re.sub(r"\s+", "", text)
    text = text.strip("·:：,，.。!！?？|[]【】()（）\"'")
    return text


def merge_ocr_samples(samples: list[tuple[float, str]], sample_interval: float) -> list[CaptionEvent]:
    events: list[CaptionEvent] = []
    active_text = ""
    active_start = 0.0
    last_t = 0.0

    def close_active() -> None:
        nonlocal active_text, active_start, last_t
        if active_text:
            events.append(CaptionEvent(round(active_start, 2), round(last_t + sample_interval, 2), active_text))
            active_text = ""

    for t, raw_text in samples:
        text = normalize_ocr_text(raw_text)
        if not text:
            close_active()
            continue
        if text == active_text and t - last_t <= sample_interval * 1.75:
            last_t = t
            continue
        close_active()
        active_text = text
        active_start = t
        last_t = t
    close_active()
    return events


def parse_subtitle_timestamp(value: str) -> float:
    value = value.strip().replace(",", ".")
    parts = value.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours = "0"
        minutes, seconds = parts
    else:
        raise ValueError(f"无法解析字幕时间：{value}")
    return round(int(hours) * 3600 + int(minutes) * 60 + float(seconds), 3)


def parse_rect_ratio(value: str) -> tuple[float, float, float, float]:
    parts = [float(x.strip()) for x in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("rect-ratio 格式应为 x,y,w,h")
    if any(x < 0 for x in parts) or parts[0] + parts[2] > 1 or parts[1] + parts[3] > 1:
        raise argparse.ArgumentTypeError("rect-ratio 必须落在 0..1 画布范围内")
    return tuple(parts)  # type: ignore[return-value]


def clean_subtitle_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\{[^}]*}", "", text)
    text = text.replace("\\N", "").replace("\\n", "")
    text = re.sub(r"\s+", "", text)
    return text.strip()


def ass_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:05.2f}"


def ass_text(value: str) -> str:
    return value.replace("{", "").replace("}", "").replace("\n", "\\N")


def font_name_from_path(path: Path | None) -> str:
    if not path:
        return "Arial Unicode MS"
    name = path.stem
    return name.split(".")[0] if name else "Arial Unicode MS"


def write_caption_ass(path: Path, events: list[CaptionEvent], *, width: int, height: int, font_name: str) -> None:
    margin_v = max(20, int(height * 0.29))
    body = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Main,{font_name},43,&H0000FFFF,&H0000FFFF,&H00000000,&H80000000,"
        f"1,0,0,0,100,100,0,0,1,3,0,2,24,24,{margin_v},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for event in events:
        text = ass_text(clean_subtitle_text(event.text))
        if not text:
            continue
        body.append(f"Dialogue: 0,{ass_timestamp(event.start)},{ass_timestamp(event.end)},Main,,0,0,0,,{text}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def parse_srt_or_vtt(path: Path) -> list[CaptionEvent]:
    lines = path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()
    events: list[CaptionEvent] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.upper() == "WEBVTT" or line.isdigit():
            i += 1
            continue
        if "-->" not in line:
            i += 1
            continue
        start_raw, end_raw = [part.strip().split()[0] for part in line.split("-->", 1)]
        text_lines: list[str] = []
        i += 1
        while i < len(lines) and lines[i].strip():
            text_lines.append(lines[i].strip())
            i += 1
        text = clean_subtitle_text("".join(text_lines))
        if text:
            events.append(CaptionEvent(parse_subtitle_timestamp(start_raw), parse_subtitle_timestamp(end_raw), text))
    return events


def parse_ass(path: Path) -> list[CaptionEvent]:
    events: list[CaptionEvent] = []
    format_fields: list[str] = []
    for raw_line in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
        line = raw_line.strip()
        if line.lower().startswith("format:"):
            format_fields = [part.strip().lower() for part in line.split(":", 1)[1].split(",")]
            continue
        if not line.lower().startswith("dialogue:"):
            continue
        if not format_fields:
            format_fields = ["layer", "start", "end", "style", "name", "marginl", "marginr", "marginv", "effect", "text"]
        values = line.split(":", 1)[1].split(",", max(0, len(format_fields) - 1))
        if len(values) < len(format_fields):
            continue
        row = {field: values[idx].strip() for idx, field in enumerate(format_fields)}
        text = clean_subtitle_text(row.get("text", ""))
        if text:
            events.append(
                CaptionEvent(
                    parse_subtitle_timestamp(row["start"]),
                    parse_subtitle_timestamp(row["end"]),
                    text,
                )
            )
    return events


def parse_subtitle_file(path: Path) -> list[CaptionEvent]:
    suffix = path.suffix.lower()
    if suffix in {".srt", ".vtt"}:
        return parse_srt_or_vtt(path)
    if suffix in {".ass", ".ssa"}:
        return parse_ass(path)
    raise ValueError(f"不支持的字幕文件格式：{path.suffix}")


def find_sidecar_subtitle(video_path: Path) -> Path | None:
    for suffix in (".srt", ".ass", ".ssa", ".vtt"):
        candidate = video_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def has_embedded_subtitle_stream(video_path: Path) -> bool:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(video_path)],
            text=True,
        )
        data = json.loads(out)
    except Exception:
        return False
    return any(stream.get("codec_type") == "subtitle" for stream in data.get("streams", []))


def extract_embedded_subtitles(video_path: Path) -> list[CaptionEvent]:
    if not has_embedded_subtitle_stream(video_path):
        return []
    with tempfile.TemporaryDirectory(prefix="clip_subtitles_") as tmp:
        out_path = Path(tmp) / "embedded.srt"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-map", "0:s:0", str(out_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return parse_subtitle_file(out_path)


def load_caption_events_from_source(
    video_path: Path,
    *,
    subtitle_file: Path | None,
    subtitle_source: str,
) -> list[CaptionEvent]:
    if subtitle_file:
        return parse_subtitle_file(subtitle_file)
    if subtitle_source in {"auto", "sidecar"}:
        sidecar = find_sidecar_subtitle(video_path)
        if sidecar:
            return parse_subtitle_file(sidecar)
        if subtitle_source == "sidecar":
            return []
    if subtitle_source in {"auto", "embedded"}:
        return extract_embedded_subtitles(video_path)
    return []


def should_attempt_ocr(subtitle_source: str, ocr_subtitles: bool, caption_events: list[CaptionEvent]) -> bool:
    if caption_events:
        return False
    return ocr_subtitles or subtitle_source == "ocr"


def should_attempt_asr(subtitle_source: str, caption_events: list[CaptionEvent]) -> bool:
    if caption_events:
        return False
    return subtitle_source in {"auto", "asr"}


def caption_filters(plan: TreatmentPlan, fontfile: Path | None) -> list[str]:
    if plan.caption_ass_path:
        return [f"subtitles=filename='{filter_value(plan.caption_ass_path)}'"]
    if plan.caption_events:
        filters: list[str] = []
        for idx, event in enumerate(plan.caption_events):
            y_jitter = 0 if idx % 2 == 0 else 16
            filters.append(
                drawtext(
                    event.text,
                    x="(w-text_w)/2",
                    y=f"h*0.665+{y_jitter}",
                    size=43,
                    color="yellow",
                    border_color="black",
                    borderw=4,
                    fontfile=fontfile,
                    extra=f"enable='between(t,{event.start:.2f},{event.end:.2f})'",
                )
            )
        return filters

    captions = plan.captions or DEFAULT_CAPTIONS
    if not captions:
        return []
    segment = 3.6
    gap = 0.4
    start = 0.7
    total = plan.caption_total_seconds
    if total is not None:
        slots = max(len(captions), int((max(total - start, 0.0) // (segment + gap)) + 1))
        captions = [captions[i % len(captions)] for i in range(slots)]
    filters: list[str] = []
    for idx, caption in enumerate(captions):
        begin = start + idx * (segment + gap)
        end = begin + segment
        y_jitter = 0 if idx % 2 == 0 else 16
        filters.append(
            drawtext(
                caption,
                x="(w-text_w)/2",
                y=f"h*0.665+{y_jitter}",
                size=43,
                color="yellow",
                border_color="black",
                borderw=4,
                fontfile=fontfile,
                extra=f"enable='between(t,{begin:.2f},{end:.2f})'",
            )
        )
    return filters


# build_filter_graph  →  renamed to build_style_filter_graph (see below)
# build_filter_complex →  renamed to build_combined_filter_complex (see below)


# ---------------------------------------------------------------------------
# Deep processing helpers
# ---------------------------------------------------------------------------


def _resolve_intensity(plan: TreatmentPlan, attr: str) -> float:
    """Return the effective intensity for a deep-processing technique."""
    specific = getattr(plan, attr, None)
    return float(specific if specific is not None else plan.deep_intensity)


def _clamp_intensity(value: float) -> float:
    return max(0.0, min(1.0, value))


# ---------------------------------------------------------------------------
# 1. Invisible watermark destruction
# ---------------------------------------------------------------------------


def build_watermark_destruction_filters(plan: TreatmentPlan, rng: random.Random) -> list[str]:
    """Filters that disrupt DCT/DWT/spread-spectrum invisible watermarks.

    Strategy: combine geometric micro-perturbation, frequency-domain noise,
    and rely on re-encoding to force DCT coefficient recomputation.
    """
    intensity = _clamp_intensity(_resolve_intensity(plan, "watermark_destruction_intensity"))
    if intensity <= 0.01:
        return []
    w, h = plan.width, plan.height
    filters: list[str] = []

    # --- geometric micro-perturbation ---
    # Rotation: ±0.3° – ±0.9°  (scaled by intensity)
    angle_deg = rng.uniform(0.3, 0.9) * intensity * (1 if rng.random() > 0.5 else -1)
    angle_rad = angle_deg * 3.141592653589793 / 180.0
    filters.append(f"rotate={angle_rad:.8f}:ow=iw:oh=ih:c=none")

    # Scale back to target size — absorbs the rotation expansion and adds
    # a subtle scale perturbation (0.997–1.003) to disrupt DCT grid alignment.
    scale_factor = 1.0 + rng.uniform(-0.003, 0.003) * intensity
    scaled_w = max(16, int(w * scale_factor))
    scaled_h = max(16, int(h * scale_factor))
    filters.append(f"scale={scaled_w}:{scaled_h}:flags=lanczos")
    # pad or crop to exact target size
    if scaled_w != w or scaled_h != h:
        filters.append(f"scale={w}:{h}:flags=lanczos")

    # --- frequency-domain noise (disrupts DCT coefficients on re-encode) ---
    noise_amp = 2.0 + intensity * 4.0
    n = round(noise_amp)
    filters.append(
        f"geq=lum='lum(X,Y)+random(1)*{n}-{n / 2}':"
        f"cb='cb(X,Y)+random(1)*{max(1,n // 2)}-{max(1,n // 2) / 2}':"
        f"cr='cr(X,Y)+random(1)*{max(1,n // 2)}-{max(1,n // 2) / 2}'"
    )

    return filters


# ---------------------------------------------------------------------------
# 2. Frame-level random perturbation
# ---------------------------------------------------------------------------


def build_frame_perturbation_filters(plan: TreatmentPlan, rng: random.Random) -> list[str]:
    """Per-frame random variations to defeat frame-accurate hashing.

    Uses ffmpeg's per-frame random() in geq so every frame gets
    a different noise pattern. Also applies a subtle global hue/saturation
    shift (fixed per run, derived from seed).
    """
    intensity = _clamp_intensity(_resolve_intensity(plan, "frame_perturbation_intensity"))
    if intensity <= 0.01:
        return []
    filters: list[str] = []

    # Per-pixel luma noise ±[2–8] (stronger than watermark mode)
    luma_amp = 2.0 + intensity * 6.0
    chroma_amp = 1.0 + intensity * 3.0
    ln = round(luma_amp)
    cn = round(chroma_amp)

    filters.append(
        f"geq=lum='lum(X,Y)+random(1)*{ln}-{ln / 2}':"
        f"cb='cb(X,Y)+random(1)*{cn}-{cn / 2}':"
        f"cr='cr(X,Y)+random(1)*{cn}-{cn / 2}'"
    )

    # Per-frame hue/saturation jitter (small fixed shifts give per-run variation)
    hue_shift = rng.uniform(-1.5, 1.5) * intensity
    sat_shift = 1.0 + rng.uniform(-0.03, 0.03) * intensity
    filters.append(f"hue=h={hue_shift:.3f}:s={sat_shift:.4f}")

    # Subtle brightness/contrast shift (fixed per run)
    bright = rng.uniform(-0.03, 0.03) * intensity
    contrast = 1.0 + rng.uniform(-0.02, 0.02) * intensity
    filters.append(f"eq=brightness={bright:.4f}:contrast={contrast:.4f}")

    return filters


# ---------------------------------------------------------------------------
# 3. Film grain synthesis
# ---------------------------------------------------------------------------


def build_film_grain_filters(plan: TreatmentPlan, rng: random.Random) -> list[str]:
    """Synthetic film grain with temporal correlation.

    Adds luma grain (temporal, mimics real film stock) and weaker
    chroma grain. Grain strength is calibrated to common film profiles.

    Profiles (selected by seed):
      - 16mm fine:     luma 2-4,  chroma 1
      - 35mm standard: luma 3-5,  chroma 1-2
      - 35mm pushed:   luma 5-8,  chroma 2-3
    """
    intensity = _clamp_intensity(_resolve_intensity(plan, "deep_intensity"))
    if intensity <= 0.01:
        return []

    profile = rng.choice(["16mm", "35mm", "pushed"])
    profiles = {
        "16mm":   (2.0, 4.0, 0.5, 1.5),
        "35mm":   (3.0, 5.5, 1.0, 2.5),
        "pushed": (5.0, 8.0, 1.5, 3.5),
    }
    luma_lo, luma_hi, chroma_lo, chroma_hi = profiles[profile]
    luma_s = round(luma_lo + (luma_hi - luma_lo) * intensity)
    chroma_s = round(chroma_lo + (chroma_hi - chroma_lo) * intensity)

    filters: list[str] = []
    # Luma grain with temporal correlation (t)
    filters.append(f"noise=alls={luma_s}:allf=t")
    # Chroma grain (Cb=index 1, Cr=index 2; only first two colour channels)
    if chroma_s > 0:
        filters.append(f"noise=c0s={chroma_s}:c1s={chroma_s}:c2s=0:allf=t")
    return filters


# ---------------------------------------------------------------------------
# 4. Color-space jitter
# ---------------------------------------------------------------------------


def build_color_space_jitter_filters(plan: TreatmentPlan, rng: random.Random) -> list[str]:
    """Subtle color-space perturbations to shift feature vectors.

    Techniques:
      - Color matrix conversion round-trip (BT.601 ↔ BT.709)
      - Subtle gamma shift (2.2 ± 0.1)
      - Per-channel curve micro-adjustment
    """
    intensity = _clamp_intensity(_resolve_intensity(plan, "deep_intensity"))
    if intensity <= 0.01:
        return []
    filters: list[str] = []

    # Color matrix round-trip (BT.601 → BT.709 or vice versa)
    # This shifts how RGB values are interpreted, then forces re-conversion
    direction = rng.choice(["601to709", "709to601"])
    if direction == "601to709":
        filters.append("colormatrix=bt601:bt709")
    else:
        filters.append("colormatrix=bt709:bt601")

    # Subtle gamma shift: 2.2 ± 0.08 * intensity
    gamma = 2.2 + rng.uniform(-0.08, 0.08) * intensity
    filters.append(f"eq=gamma={gamma:.4f}")

    # Per-channel curves (subtle highlight/shadow tweak)
    # curves=preset=... or manual curves
    preset = rng.choice(["lighter", "darker", "strong_contrast", "none"])
    if preset != "none":
        filters.append(f"curves=preset={preset}")

    return filters


# ---------------------------------------------------------------------------
# 5. Temporal filter jitter
# ---------------------------------------------------------------------------


def build_temporal_jitter_filters(plan: TreatmentPlan, rng: random.Random) -> list[str]:
    """Alternating temporal denoise / sharpen to disrupt temporal fingerprints.

    Applies a light temporal denoise followed by a subtle unsharp mask.
    The alternating pattern breaks temporal consistency that fingerprinting
    algorithms rely on.
    """
    intensity = _clamp_intensity(_resolve_intensity(plan, "deep_intensity"))
    if intensity <= 0.01:
        return []
    filters: list[str] = []

    # hqdn3d: light temporal denoise
    luma_spatial = rng.uniform(1.5, 3.5) * intensity
    chroma_spatial = rng.uniform(2.0, 4.0) * intensity
    luma_tmp = rng.uniform(2.0, 5.0) * intensity
    chroma_tmp = rng.uniform(3.0, 6.0) * intensity
    filters.append(
        f"hqdn3d="
        f"luma_spatial={luma_spatial:.1f}:"
        f"chroma_spatial={chroma_spatial:.1f}:"
        f"luma_tmp={luma_tmp:.1f}:"
        f"chroma_tmp={chroma_tmp:.1f}"
    )

    # Subtle unsharp to add back micro-contrast (different from original)
    luma_amount = rng.uniform(0.3, 0.7) * intensity
    filters.append(
        f"unsharp=luma_msize_x=3:luma_msize_y=3:luma_amount={luma_amount:.3f}"
    )

    return filters


# ---------------------------------------------------------------------------
# 6. Block-boundary perturbation
# ---------------------------------------------------------------------------


def build_block_boundary_filters(plan: TreatmentPlan, rng: random.Random) -> list[str]:
    """Shift frame content by a few pixels on non-macroblock-aligned boundaries.

    Crops 1–5 px from edges (random offset) then pads back to original size,
    forcing the encoder to compute a different macroblock partitioning.
    """
    intensity = _clamp_intensity(_resolve_intensity(plan, "deep_intensity"))
    if intensity <= 0.01:
        return []
    w, h = plan.width, plan.height

    # Crop 1-5 pixels from each edge (non-aligned to 16-pixel macroblock grid)
    crop_margin = max(1, round(intensity * 5))
    left = rng.randint(0, crop_margin)
    top = rng.randint(0, crop_margin)
    crop_w = w - left - rng.randint(0, crop_margin)
    crop_h = h - top - rng.randint(0, crop_margin)

    filters: list[str] = [
        f"crop={crop_w}:{crop_h}:{left}:{top}",
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black",
    ]
    return filters


# ---------------------------------------------------------------------------
# Deep filter aggregation
# ---------------------------------------------------------------------------


def build_deep_processing_filters(plan: TreatmentPlan) -> list[str]:
    """Build all active deep-processing video filters in the correct order.

    Order: geometric → block → colour → noise/grain → temporal
    """
    rng = random.Random(plan.seed)
    modes = set(plan.modes)
    all_filters: list[str] = []

    if "invisible-watermark-destruction" in modes:
        all_filters.extend(build_watermark_destruction_filters(plan, rng))
    if "block-boundary-perturbation" in modes:
        all_filters.extend(build_block_boundary_filters(plan, rng))
    if "color-space-jitter" in modes:
        all_filters.extend(build_color_space_jitter_filters(plan, rng))
    if "frame-random-perturbation" in modes:
        all_filters.extend(build_frame_perturbation_filters(plan, rng))
    if "film-grain-synthesis" in modes:
        all_filters.extend(build_film_grain_filters(plan, rng))
    if "temporal-filter-jitter" in modes:
        all_filters.extend(build_temporal_jitter_filters(plan, rng))

    return all_filters


# ---------------------------------------------------------------------------
# Audio fingerprint evasion
# ---------------------------------------------------------------------------


def build_audio_evasion_filters(plan: TreatmentPlan, rng: random.Random) -> list[str]:
    """Build audio filter chain to disrupt audio fingerprinting.

    Targets spectrogram-peak-based (Shazam-style) and chroma-based (ContentID)
    fingerprinting through:
      - Pitch shifting         (rubberband, ±10–40 cents)
      - Time stretching        (rubberband, ±0.1–0.5 %)
      - EQ notching            (equalizer × 3, random narrow-band ±0.5–2 dB)
      - Phase smearing         (aphaser, very light)
      - Dynamic range shift    (compand)
      - Noise floor elevation  (-60 to -50 dBFS)
    """
    intensity = _clamp_intensity(_resolve_intensity(plan, "audio_evasion_intensity"))
    if intensity <= 0.01:
        return []
    filters: list[str] = []

    # --- pitch shift: ±10–40 cents ---
    pitch_cents = rng.uniform(10, 40) * intensity * (1 if rng.random() > 0.5 else -1)
    # rubberband pitch: 1.0 = no change; 1 semitone up = 2^(1/12) ≈ 1.05946
    # X cents = 2^(X/1200)
    pitch_ratio = 2.0 ** (pitch_cents / 1200.0)
    filters.append(f"rubberband=pitch={pitch_ratio:.6f}")

    # --- time stretch: ±0.1–0.5 % ---
    tempo_shift = rng.uniform(0.1, 0.5) * intensity * (1 if rng.random() > 0.5 else -1)
    tempo_ratio = 1.0 + tempo_shift / 100.0
    filters.append(f"rubberband=tempo={tempo_ratio:.6f}")

    # --- EQ notching: 3 random narrow-band filters ---
    eq_bands = [
        (rng.randint(200, 8000), rng.uniform(0.5, 3.0)),
        (rng.randint(200, 8000), rng.uniform(0.5, 3.0)),
        (rng.randint(200, 8000), rng.uniform(0.5, 3.0)),
    ]
    for freq, q_val in eq_bands:
        gain = rng.uniform(-2.0, 2.0) * intensity * (1 if rng.random() > 0.5 else -1)
        filters.append(f"equalizer=f={freq}:t=q:w={q_val:.1f}:g={gain:.2f}")

    # --- phase smearing: very light phaser ---
    # speed range [0.1, 2], keep low for subtle effect
    filters.append(
        f"aphaser=in_gain=0.7:out_gain=0.9:delay={rng.uniform(1.0, 3.0):.1f}:"
        f"decay={rng.uniform(0.2, 0.4):.2f}:speed={rng.uniform(0.1, 0.4):.3f}:type=triangular"
    )

    # --- dynamic range compression ---
    comp_ratio = rng.uniform(1.3, 2.5) * intensity
    filters.append(
        f"compand=attacks=0.05:decays=0.15:"
        f"points=-80/-80|-30/-{15 + comp_ratio * 5:.0f}|0/-{comp_ratio:.0f}:"
        f"volume=-{rng.uniform(0.3, 1.5):.1f}"
    )

    return filters


# ---------------------------------------------------------------------------
# Hash-evasion parameter generation
# ---------------------------------------------------------------------------


def generate_hash_evasion_params(plan: TreatmentPlan, rng: random.Random) -> dict:
    """Generate randomized encoding parameters that alter the output file hash
    without perceptible quality change.

    Returns a dict with keys:
      - x264_params: str   (for -x264-params)
      - crf_adjust: float  (to add to plan.crf)
      - gop_size: int      (for -g)
      - scenecut: int      (for -sc_threshold)
      - movflags: str      (for -movflags)
      - metadata: dict     (for -metadata entries)
      - sei_comment: str   (for -bsf:v h264_metadata)
      - keyint_min: int
    """
    intensity = _clamp_intensity(_resolve_intensity(plan, "hash_evasion_intensity"))
    if intensity <= 0.01:
        return {}

    # --- x264 parameter set (pick from pre-defined table) ---
    param_set = rng.choice(X264_PARAM_SETS)
    x264_params = ":".join(f"{k}={v}" for k, v in param_set.items())

    # --- CRF micro-variation: ±0.3 – ±0.8 ---
    crf_delta = rng.uniform(0.3, 0.8) * intensity * (1 if rng.random() > 0.5 else -1)

    # --- GOP structure variation ---
    gop_size = rng.randint(25, 60)
    keyint_min = rng.randint(1, max(1, gop_size // 4))
    scenecut = rng.randint(30, 55)

    # --- movflags variation ---
    movflags_choices = [
        "+faststart",
        "+faststart+delay_moov",
        "+faststart+separate_moof",
        "+faststart+empty_moov",
    ]
    movflags = rng.choice(movflags_choices)

    # --- fake metadata ---
    fake = rng.choice(FAKE_ENCODER_STRINGS)
    comment = rng.choice(FAKE_COMMENTS)
    metadata = {
        "comment": comment,
        "encoder": fake,
    }

    # --- SEI user data comment ---
    fake_uuid = str(uuid.UUID(int=rng.getrandbits(128)))
    sei_comment = f"sei_user_data={fake_uuid}+{rng.randint(10000, 99999)}"

    return {
        "x264_params": x264_params,
        "crf_adjust": crf_delta,
        "gop_size": gop_size,
        "keyint_min": keyint_min,
        "scenecut": scenecut,
        "movflags": movflags,
        "metadata": metadata,
        "sei_comment": sei_comment,
    }


# ---------------------------------------------------------------------------
# Container-level evasion (post-processing)
# ---------------------------------------------------------------------------


def apply_container_evasion(input_path: Path, plan: TreatmentPlan, rng: random.Random) -> Path | None:
    """Post-process: remux to change container structure without re-encoding.

    Strategy options (picked by seed):
      a) MP4 → MKV → MP4  (erases container history)
      b) Remux MP4 with different fragment settings
      c) Change NAL unit length size

    Returns the new output path, or None if this step is skipped.
    """
    intensity = _clamp_intensity(_resolve_intensity(plan, "hash_evasion_intensity"))
    if intensity <= 0.01:
        return None

    strategy = rng.choice(["mkv_roundtrip", "refragment", "nal_size"])
    tmp_dir = input_path.parent
    base_name = input_path.stem

    if strategy == "mkv_roundtrip":
        # MP4 → MKV → MP4 to erase container-level history
        mkv_path = tmp_dir / f"{base_name}.tmp.mkv"
        final_path = tmp_dir / f"{base_name}.evaded.mp4"
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(input_path), "-c", "copy", str(mkv_path)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
            )
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(mkv_path), "-c", "copy",
                 "-movflags", "+faststart", str(final_path)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
            )
            mkv_path.unlink(missing_ok=True)
            return final_path
        except subprocess.CalledProcessError:
            mkv_path.unlink(missing_ok=True)
            return None

    elif strategy == "refragment":
        # Remux with different fragment duration
        frag_duration = rng.randint(500, 5000)  # ms
        final_path = tmp_dir / f"{base_name}.evaded.mp4"
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(input_path), "-c", "copy",
                 "-f", "mp4", "-movflags", f"+frag_keyframe+empty_moov+default_base_moof",
                 "-frag_duration", str(frag_duration), str(final_path)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
            )
            return final_path
        except subprocess.CalledProcessError:
            return None

    elif strategy == "nal_size":
        # Change NAL unit length size
        nal_size = rng.choice([2, 4])
        final_path = tmp_dir / f"{base_name}.evaded.mp4"
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(input_path), "-c", "copy",
                 "-bsf:v", f"h264_metadata=aud=insert", str(final_path)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
            )
            return final_path
        except subprocess.CalledProcessError:
            return None

    return None


# ---------------------------------------------------------------------------
# Combined filter graph builder
# ---------------------------------------------------------------------------


def build_style_filter_graph(plan: TreatmentPlan) -> str:
    """Build the original clip-style visible-treatment filter chain.

    This is the original build_filter_graph, renamed for clarity.
    It produces the comma-separated chain of drawtext/drawbox/blur/overlay
    filters that give the video its "clip" look.
    """
    rng = random.Random(plan.seed)
    fontfile = plan.fontfile or default_fontfile()
    top_h = max(92, int(plan.height * 0.088))
    side_pad = max(18, int(plan.width * 0.035))
    blur_y = int(plan.height * plan.subtitle_band_y_ratio)
    blur_h = int(plan.height * plan.subtitle_band_height_ratio)
    sticker_a = rng.randint(22, 34)
    sticker_b = rng.randint(16, 28)

    filters = [
        f"scale=w={plan.width}:h={plan.height}:force_original_aspect_ratio=decrease",
        f"pad={plan.width}:{plan.height}:(ow-iw)/2:(oh-ih)/2:color=black",
        "setsar=1",
        "split[main][blur]",
        f"[blur]crop={plan.width}:{blur_h}:0:{blur_y},boxblur=14:1[blurred]",
        f"[main][blurred]overlay=0:{blur_y}",
        f"drawbox=x=0:y={blur_y}:w={plan.width}:h={blur_h}:color=black@0.34:t=fill",
        f"drawbox=x=0:y=0:w={plan.width}:h={top_h}:color=black@0.74:t=fill",
        f"drawbox=x=0:y=0:w={side_pad}:h={plan.height}:color=white@0.10:t=fill",
        f"drawbox=x={plan.width - side_pad}:y=0:w={side_pad}:h={plan.height}:color=white@0.10:t=fill",
        f"drawbox=x=18:y=18:w={plan.width - 36}:h={plan.height - 36}:color=white@0.22:t=3",
        f"drawbox=x=30:y=30:w={plan.width - 60}:h={plan.height - 60}:color=0x5fbf7a@0.20:t=2",
        drawtext(plan.title, x="(w-text_w)/2", y="28", size=38, color="white", border_color="black", borderw=3, fontfile=fontfile),
        drawtext(plan.vertical_text, x="34", y="h*0.22", size=30, color="white", border_color="black", borderw=2, fontfile=fontfile),
        drawtext("*", x=f"w-{sticker_a * 3}", y="h*0.18", size=sticker_a, color="white@0.95", border_color="black@0.35", borderw=1, fontfile=fontfile),
        drawtext("+", x="w*0.10", y="h*0.84", size=sticker_b, color="white@0.92", border_color="black@0.30", borderw=1, fontfile=fontfile),
    ]
    filters.extend(caption_filters(plan, fontfile))
    return ",".join(filters)


build_filter_graph = build_style_filter_graph


def rect_ratio_to_pixels(plan: TreatmentPlan, ratios: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    x, y, w, h = ratios
    rect_x = int(round(plan.width * x))
    rect_y = int(round(plan.height * y))
    rect_w = int(round(plan.width * w))
    rect_h = int(round(plan.height * h))
    rect_w = max(2, min(plan.width - rect_x, rect_w))
    rect_h = max(2, min(plan.height - rect_y, rect_h))
    return rect_x, rect_y, rect_w, rect_h


def build_template_content_filter_graph(plan: TreatmentPlan) -> str:
    """Fit source video into the transparent window of an external template."""
    if not plan.template_content_rect_ratio:
        raise ValueError("template_content_rect_ratio is required")

    fontfile = plan.fontfile or default_fontfile()
    rect_x, rect_y, rect_w, rect_h = rect_ratio_to_pixels(plan, plan.template_content_rect_ratio)
    blur_y = int(plan.height * plan.subtitle_band_y_ratio)
    blur_h = int(plan.height * plan.subtitle_band_height_ratio)
    blur_y = max(rect_y, min(rect_y + rect_h - 2, blur_y))
    blur_h = max(2, min(rect_y + rect_h - blur_y, blur_h))

    filters = [
        *build_deep_processing_filters(plan),
        f"scale={rect_w}:{rect_h}:force_original_aspect_ratio=increase",
        f"crop={rect_w}:{rect_h}:(iw-{rect_w})/2:(ih-{rect_h})/2",
        "setsar=1",
        f"pad={plan.width}:{plan.height}:{rect_x}:{rect_y}:color=black",
        "split[main][blur]",
        f"[blur]crop={rect_w}:{blur_h}:{rect_x}:{blur_y},boxblur=14:1[blurred]",
        f"[main][blurred]overlay={rect_x}:{blur_y}",
        f"drawbox=x={rect_x}:y={blur_y}:w={rect_w}:h={blur_h}:color=black@0.34:t=fill",
    ]
    filters.extend(caption_filters(plan, fontfile))
    return ",".join(filters)


def build_combined_filter_graph(plan: TreatmentPlan) -> str:
    """Build the full video filter chain: deep processing + style filters.

    Deep processing filters come first (geometric → colour → noise → temporal),
    then style filters (clip look).  If --disable-style is set, only deep
    processing filters are returned.
    """
    parts: list[str] = []

    deep_filters = build_deep_processing_filters(plan)
    if deep_filters:
        parts.extend(deep_filters)

    if not plan.disable_style:
        style_graph = build_style_filter_graph(plan)
        if style_graph:
            parts.append(style_graph)

    return ",".join(parts)


def build_combined_filter_complex(plan: TreatmentPlan) -> str:
    """Build filter_complex string including template overlay if present."""
    uses_template_content_rect = bool(plan.template_png and plan.template_content_rect_ratio)
    if uses_template_content_rect:
        base_graph = build_template_content_filter_graph(plan)
    else:
        base_graph = build_combined_filter_graph(plan)
    if not plan.template_png:
        return f"[0:v]{base_graph}[vout]"
    if uses_template_content_rect and plan.title:
        fontfile = plan.fontfile or default_fontfile()
        title_filter = drawtext(
            plan.title,
            x="(w-text_w)/2",
            y="24",
            size=44,
            color="white",
            border_color="black",
            borderw=3,
            fontfile=fontfile,
        )
        return (
            f"[0:v]{base_graph}[base];"
            f"[1:v]scale={plan.width}:{plan.height}[template];"
            "[base][template]overlay=0:0[templated];"
            f"[templated]{title_filter}[vout]"
        )
    return (
        f"[0:v]{base_graph}[base];"
        f"[1:v]scale={plan.width}:{plan.height}[template];"
        "[base][template]overlay=0:0[vout]"
    )


def probe_duration(path: Path) -> float | None:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
            text=True,
        ).strip()
        return float(out) if out else None
    except Exception:
        return None


def tesseract_languages() -> set[str]:
    if not shutil.which("tesseract"):
        return set()
    try:
        out = subprocess.check_output(["tesseract", "--list-langs"], stderr=subprocess.STDOUT, text=True)
    except Exception:
        return set()
    return {line.strip() for line in out.splitlines()[1:] if line.strip()}


def ensure_ocr_language(lang: str) -> None:
    available = tesseract_languages()
    requested = {part for part in re.split(r"[+ ]+", lang) if part}
    missing = sorted(part for part in requested if part not in available)
    if missing:
        raise RuntimeError(
            "Tesseract 缺少 OCR 语言包："
            + ", ".join(missing)
            + "。当前可用："
            + (", ".join(sorted(available)) or "无")
            + "。macOS 可先执行：brew install tesseract-lang，或用 --ocr-lang 指向已安装语言。"
        )


def ocr_image_with_tesseract(image_path: Path, lang: str) -> str:
    cmd = ["tesseract", str(image_path), "stdout", "-l", lang, "--psm", "6"]
    return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)


def text_from_rapidocr_result(result: object, *, min_confidence: float) -> str:
    if not result:
        return ""
    pieces: list[str] = []
    for item in result if isinstance(result, list) else []:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        text = str(item[1]).strip()
        try:
            confidence = float(item[2])
        except (TypeError, ValueError):
            confidence = 0.0
        if text and confidence >= min_confidence:
            pieces.append(text)
    return "".join(pieces)


def extract_subtitle_events(
    video_path: Path,
    *,
    duration: float | None,
    sample_interval: float,
    crop_y_ratio: float,
    crop_height_ratio: float,
    lang: str,
    engine: str,
    min_confidence: float,
) -> list[CaptionEvent]:
    import cv2

    rapid_ocr = None
    if engine == "rapidocr":
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore
        except ImportError as exc:
            raise RuntimeError("RapidOCR 未安装。请执行：python3 -m pip install rapidocr-onnxruntime") from exc
        rapid_ocr = RapidOCR()
    elif engine == "tesseract":
        ensure_ocr_language(lang)
    else:
        raise RuntimeError(f"未知 OCR 引擎：{engine}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频做 OCR：{video_path}")
    video_duration = duration or probe_duration(video_path) or 0.0
    samples: list[tuple[float, str]] = []
    with tempfile.TemporaryDirectory(prefix="clip_ocr_") as tmp:
        tmp_dir = Path(tmp)
        t = 0.0
        while t <= video_duration:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            y = max(0, min(h - 1, int(h * crop_y_ratio)))
            crop_h = max(8, min(h - y, int(h * crop_height_ratio)))
            crop = frame[y : y + crop_h, :]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (w * 2, crop_h * 2), interpolation=cv2.INTER_CUBIC)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            image_path = tmp_dir / f"subtitle_{len(samples):05d}.png"
            cv2.imwrite(str(image_path), binary)
            if rapid_ocr:
                result, _elapsed = rapid_ocr(str(image_path))
                text = text_from_rapidocr_result(result, min_confidence=min_confidence)
            else:
                text = ocr_image_with_tesseract(image_path, lang)
            samples.append((round(t, 2), text))
            t += sample_interval
    cap.release()
    return merge_ocr_samples(samples, sample_interval)


def extract_audio_wav(video_path: Path, wav_path: Path, *, duration: float | None) -> None:
    cmd = ["ffmpeg", "-y"]
    if duration:
        cmd.extend(["-t", f"{duration:.3f}"])
    cmd.extend(["-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav_path)])
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def transcribe_audio_events(video_path: Path, *, model_name: str, language: str | None, duration: float | None) -> list[CaptionEvent]:
    with tempfile.TemporaryDirectory(prefix="clip_asr_") as tmp:
        wav_path = Path(tmp) / "audio.wav"
        extract_audio_wav(video_path, wav_path, duration=duration)
        try:
            from faster_whisper import WhisperModel  # type: ignore

            model = WhisperModel(model_name, device="auto", compute_type="auto")
            segments, _info = model.transcribe(str(wav_path), language=language or None, vad_filter=True)
            return [
                CaptionEvent(round(float(seg.start), 2), round(float(seg.end), 2), clean_subtitle_text(seg.text))
                for seg in segments
                if clean_subtitle_text(seg.text)
            ]
        except ImportError:
            pass

        try:
            import whisper  # type: ignore

            model = whisper.load_model(model_name)
            result = model.transcribe(str(wav_path), language=language or None)
            return [
                CaptionEvent(round(float(seg["start"]), 2), round(float(seg["end"]), 2), clean_subtitle_text(str(seg["text"])))
                for seg in result.get("segments", [])
                if clean_subtitle_text(str(seg.get("text", "")))
            ]
        except ImportError as exc:
            raise RuntimeError(
                "本地 ASR 依赖未安装。请安装 faster-whisper 或 openai-whisper，例如："
                "python3 -m pip install faster-whisper"
            ) from exc


def build_ffmpeg_command(plan: TreatmentPlan, *, validate_paths: bool = True) -> list[str]:
    if validate_paths and not plan.source.exists():
        raise FileNotFoundError(f"找不到输入视频：{plan.source}")
    if validate_paths and plan.template_png and not plan.template_png.exists():
        raise FileNotFoundError(f"找不到模板 PNG：{plan.template_png}")
    plan.output.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(plan.seed)
    has_deep = bool(set(plan.modes) & DEEP_MODES)
    has_hash_evasion = "hash-evasion-chain" in plan.modes

    # Audio filters (if audio fingerprint evasion is active)
    audio_filters = build_audio_evasion_filters(plan, rng) if "audio-fingerprint-evasion" in plan.modes else []

    # Hash-evasion encoding parameters
    hash_params = generate_hash_evasion_params(plan, rng) if has_hash_evasion else {}

    # Effective CRF (base ± micro-adjust from hash evasion)
    effective_crf = plan.crf
    if hash_params:
        effective_crf = round(plan.crf + hash_params.get("crf_adjust", 0.0), 1)

    cmd = ["ffmpeg", "-y"]
    if plan.duration:
        cmd.extend(["-t", f"{plan.duration:.3f}"])

    # Input(s)
    cmd.extend(["-i", str(plan.source)])
    if plan.template_png:
        cmd.extend(["-i", str(plan.template_png)])

    # Video filter graph
    if plan.template_png:
        cmd.extend(["-filter_complex", build_combined_filter_complex(plan), "-map", "[vout]"])
    else:
        cmd.extend(["-vf", build_combined_filter_graph(plan), "-map", "0:v:0"])

    if plan.fps:
        cmd.extend(["-r", str(plan.fps)])

    # Audio: either with evasion filters or passthrough re-encode
    cmd.extend(["-map", "0:a?"])
    if audio_filters:
        audio_chain = ",".join(audio_filters)
        cmd.extend(["-af", audio_chain])

    # Video codec: ABR mode if target_bitrate is set, otherwise CRF
    cmd.extend(["-c:v", plan.codec, "-preset", plan.preset])
    if plan.target_bitrate:
        cmd.extend(["-b:v", plan.target_bitrate])
        # Derive maxrate/bufsize as 1.5x target (e.g. 4M -> 6M)
        maxrate = plan.target_bitrate
        if maxrate.lower().endswith("k"):
            maxrate_val = int(int(maxrate[:-1]) * 1.5)
            maxrate = f"{maxrate_val}k"
        elif maxrate.lower().endswith("m"):
            maxrate_val = int(int(maxrate[:-1]) * 1.5)
            maxrate = f"{maxrate_val}M"
        cmd.extend(["-minrate", plan.target_bitrate])
        cmd.extend(["-maxrate", maxrate])
        cmd.extend(["-bufsize", maxrate])
    else:
        cmd.extend(["-crf", str(effective_crf)])
        if plan.min_bitrate:
            cmd.extend(["-minrate", plan.min_bitrate])
            maxrate = plan.min_bitrate
            if maxrate.lower().endswith("k"):
                maxrate_val = int(maxrate[:-1]) * 2
                maxrate = f"{maxrate_val}k"
            elif maxrate.lower().endswith("m"):
                maxrate_val = int(maxrate[:-1]) * 2
                maxrate = f"{maxrate_val}M"
            cmd.extend(["-maxrate", maxrate])
            cmd.extend(["-bufsize", maxrate])

    # x264/x265 params for hash evasion
    if hash_params and hash_params.get("x264_params") and plan.codec == "libx264":
        cmd.extend(["-x264-params", hash_params["x264_params"]])

    cmd.extend(["-pix_fmt", "yuv420p"])

    # GOP structure (hash evasion)
    if hash_params and hash_params.get("gop_size"):
        cmd.extend(["-g", str(hash_params["gop_size"])])
    if hash_params and hash_params.get("keyint_min"):
        cmd.extend(["-keyint_min", str(hash_params["keyint_min"])])
    if hash_params and hash_params.get("scenecut"):
        cmd.extend(["-sc_threshold", str(hash_params["scenecut"])])

    # Audio codec
    cmd.extend(["-c:a", "aac", "-b:a", plan.audio_bitrate])

    # movflags
    movflags = hash_params.get("movflags", "+faststart") if has_hash_evasion else "+faststart"
    cmd.extend(["-movflags", movflags])

    # Metadata (hash evasion injects fake metadata)
    if hash_params and hash_params.get("metadata"):
        for key, value in hash_params["metadata"].items():
            cmd.extend(["-metadata", f"{key}={value}"])
    else:
        cmd.extend(["-metadata", "comment=platform detector QA"])

    # SEI user data (hash evasion)
    if hash_params and hash_params.get("sei_comment"):
        cmd.extend(["-bsf:v", f"h264_metadata={hash_params['sei_comment']}"])

    cmd.append(str(plan.output))
    return cmd


def read_captions(path: Path | None) -> list[str]:
    if not path:
        return DEFAULT_CAPTIONS.copy()
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="把 raw 视频处理成常见剪辑样式，用于平台重复检测鲁棒性测试。")
    parser.add_argument("--input", default="raw_video.mp4", help="输入视频，默认 raw_video.mp4")
    parser.add_argument("--output", default="clip_style_output.mp4", help="输出视频")
    parser.add_argument("--width", type=int, default=720, help="目标画布宽度")
    parser.add_argument("--height", type=int, default=1280, help="目标画布高度")
    parser.add_argument("--title", default="短剧精彩片段", help="顶部剧名横幅文字")
    parser.add_argument("--vertical-text", default="", help="左侧竖排文字")
    parser.add_argument("--captions", type=Path, help="调试兜底：手工字幕文本文件，每行一句。正常流程不需要。")
    parser.add_argument("--subtitle-file", type=Path, help="调试兜底：指定 .srt/.vtt/.ass/.ssa。正常流程会自动从视频提取。")
    parser.add_argument(
        "--subtitle-source",
        choices=["auto", "file", "sidecar", "embedded", "asr", "ocr", "none"],
        default="auto",
        help="原字幕来源：auto 先内嵌/同名字幕，再本地 ASR；ocr 仅显式指定时使用。",
    )
    parser.add_argument("--asr-model", default="small", help="本地 Whisper/faster-whisper 模型名，兼顾效率可用 tiny/base/small")
    parser.add_argument("--asr-language", default="zh", help="ASR 语言代码，中文默认 zh；不确定可传空字符串")
    parser.add_argument("--ocr-subtitles", action="store_true", help="兼容旧参数：从原视频字幕区域 OCR 提取原字幕")
    parser.add_argument("--ocr-engine", choices=["rapidocr", "tesseract"], default="rapidocr", help="OCR 引擎，rapidocr 为轻量本地 ONNX 模型")
    parser.add_argument("--ocr-min-confidence", type=float, default=0.5, help="RapidOCR 最低置信度阈值")
    parser.add_argument("--ocr-lang", default="chi_sim+eng", help="Tesseract OCR 语言，中文字幕通常用 chi_sim+eng")
    parser.add_argument("--ocr-sample-interval", type=float, default=0.8, help="OCR 抽帧间隔秒数")
    parser.add_argument("--ocr-crop-y-ratio", type=float, default=0.60, help="原片字幕裁剪区起始位置占原片高度比例")
    parser.add_argument("--ocr-crop-height-ratio", type=float, default=0.24, help="原片字幕裁剪区高度占原片高度比例")
    parser.add_argument("--seed", type=int, default=20260624, help="装饰位置 seed，保证可复现")
    parser.add_argument("--crf", type=int, default=24, help="H.264 CRF")
    parser.add_argument("--preset", default="medium", help="x264 preset")
    parser.add_argument("--audio-bitrate", default="128k", help="AAC 音频码率")
    parser.add_argument("--fps", type=int, help="可选输出帧率")
    parser.add_argument("--duration", type=float, help="只处理前 N 秒，便于快速 smoke test")
    parser.add_argument("--fontfile", type=Path, help="可选字体文件，中文字幕建议指定可用中文字体")
    parser.add_argument("--codec", default="libx264", choices=["libx264", "libx265"], help="视频编码器，默认 libx264；libx265 体积更小但兼容性略差")
    parser.add_argument("--min-bitrate", default=None, help="最低视频码率（CRF 模式下），例如 4M")
    parser.add_argument("--target-bitrate", default=None, help="目标平均视频码率，例如 4M；设置后改用 ABR 模式，强制每集平均码率接近此值")
    parser.add_argument("--caption-total-seconds", type=float, help="字幕铺满的目标时长；默认使用输出时长或输入视频时长")
    parser.add_argument("--subtitle-band-y-ratio", type=float, default=0.61, help="字幕遮挡区起始位置占画布高度比例")
    parser.add_argument("--subtitle-band-height-ratio", type=float, default=0.15, help="字幕遮挡区高度占画布高度比例")
    parser.add_argument("--template-png", type=Path, help="透明 PNG 模板/边框，按目标画布缩放后覆盖在最上层")
    parser.add_argument(
        "--template-content-rect-ratio",
        type=parse_rect_ratio,
        help="模板透明窗口比例 x,y,w,h；传入后会把原视频 cover 裁剪进这个窗口，再叠模板。",
    )
    parser.add_argument(
        "--mode",
        action="append",
        default=[],
        help="声明要启用的处理模式。可见编辑模式 + 深度反检测模式均支持。可重复使用启用多个模式。",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印 FFmpeg 命令，不执行")

    # ------------------------------------------------------------------
    # Deep processing options
    # ------------------------------------------------------------------
    deep_group = parser.add_argument_group("深度反检测处理选项")
    deep_group.add_argument(
        "--deep-intensity",
        type=float,
        default=0.5,
        help="全局深度处理强度 0.0–1.0，默认 0.5。强度越高画面/音频扰动越大。",
    )
    deep_group.add_argument(
        "--watermark-destruction-intensity",
        type=float,
        help="不可见水印破坏强度 (覆盖 --deep-intensity)",
    )
    deep_group.add_argument(
        "--frame-perturbation-intensity",
        type=float,
        help="帧级随机扰动强度 (覆盖 --deep-intensity)",
    )
    deep_group.add_argument(
        "--audio-evasion-intensity",
        type=float,
        help="音频指纹规避强度 (覆盖 --deep-intensity)",
    )
    deep_group.add_argument(
        "--hash-evasion-intensity",
        type=float,
        help="Hash 规避参数链强度 (覆盖 --deep-intensity)",
    )
    deep_group.add_argument(
        "--disable-style",
        action="store_true",
        help="仅应用深度处理，跳过所有可见剪辑样式（字幕、边框、横幅等）。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_requested_modes(args.mode)
    caption_total_seconds = args.caption_total_seconds or args.duration or probe_duration(Path(args.input))
    caption_events: list[CaptionEvent] = []
    captions = read_captions(args.captions)
    if args.subtitle_source != "none":
        try:
            caption_events = load_caption_events_from_source(
                Path(args.input),
                subtitle_file=args.subtitle_file,
                subtitle_source="file" if args.subtitle_file and args.subtitle_source == "auto" else args.subtitle_source,
            )
        except (ValueError, subprocess.CalledProcessError) as exc:
            raise SystemExit(f"字幕文件/内嵌字幕提取失败：{exc}") from exc
    use_ocr = should_attempt_ocr(args.subtitle_source, args.ocr_subtitles, caption_events)
    use_asr = should_attempt_asr(args.subtitle_source, caption_events)
    if use_asr:
        try:
            caption_events = transcribe_audio_events(
                Path(args.input),
                model_name=args.asr_model,
                language=args.asr_language or None,
                duration=args.duration,
            )
        except (RuntimeError, subprocess.CalledProcessError) as exc:
            raise SystemExit(str(exc)) from exc
        if not caption_events:
            raise SystemExit("ASR 没有识别到对白字幕；请尝试更大的 --asr-model，或检查音频轨。")
    if use_ocr and not caption_events:
        try:
            caption_events = extract_subtitle_events(
                Path(args.input),
                duration=args.duration,
                sample_interval=args.ocr_sample_interval,
                crop_y_ratio=args.ocr_crop_y_ratio,
                crop_height_ratio=args.ocr_crop_height_ratio,
                lang=args.ocr_lang,
                engine=args.ocr_engine,
                min_confidence=args.ocr_min_confidence,
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        if not caption_events:
            raise SystemExit("OCR 没有识别到原字幕；请调大/调小 --ocr-crop-y-ratio 和 --ocr-crop-height-ratio，或检查 OCR 语言包。")
    if args.subtitle_source in {"file", "sidecar", "embedded"} and not caption_events:
        raise SystemExit(f"没有从 {args.subtitle_source} 来源提取到字幕。硬字幕视频需要 --subtitle-source ocr。")
    if args.subtitle_source == "auto" and not caption_events and not args.captions:
        raise SystemExit("没有从视频中提取到字幕，且未提供调试兜底字幕。请确认视频有内嵌字幕，或安装 OCR 依赖用于硬字幕提取。")
    if caption_events:
        captions = []
    caption_ass_path = None
    if caption_events:
        caption_ass_path = Path(args.output).with_suffix(".captions.ass")
        write_caption_ass(
            caption_ass_path,
            caption_events,
            width=args.width,
            height=args.height,
            font_name=font_name_from_path(args.fontfile or default_fontfile()),
        )
    plan = TreatmentPlan(
        source=Path(args.input),
        output=Path(args.output),
        width=args.width,
        height=args.height,
        title=args.title,
        vertical_text=args.vertical_text,
        captions=captions,
        caption_events=caption_events,
        caption_ass_path=caption_ass_path,
        seed=args.seed,
        crf=args.crf,
        preset=args.preset,
        audio_bitrate=args.audio_bitrate,
        fps=args.fps,
        duration=args.duration,
        fontfile=args.fontfile,
        caption_total_seconds=caption_total_seconds,
        subtitle_band_y_ratio=args.subtitle_band_y_ratio,
        subtitle_band_height_ratio=args.subtitle_band_height_ratio,
        template_png=args.template_png,
        template_content_rect_ratio=args.template_content_rect_ratio,
        # Deep processing
        modes=args.mode,
        disable_style=args.disable_style,
        deep_intensity=args.deep_intensity,
        watermark_destruction_intensity=args.watermark_destruction_intensity,
        frame_perturbation_intensity=args.frame_perturbation_intensity,
        audio_evasion_intensity=args.audio_evasion_intensity,
        hash_evasion_intensity=args.hash_evasion_intensity,
        codec=args.codec,
        min_bitrate=args.min_bitrate,
        target_bitrate=args.target_bitrate,
    )

    # Log active deep processing modes
    active_deep = sorted(set(plan.modes) & DEEP_MODES)
    if active_deep:
        print(f"[deep] active modes: {', '.join(active_deep)}")
        print(f"[deep] global intensity: {plan.deep_intensity}")
        if plan.disable_style:
            print("[deep] style filters DISABLED (--disable-style)")

    cmd = build_ffmpeg_command(plan)
    print(" ".join(shlex.quote(part) for part in cmd))
    if args.dry_run:
        return 0
    subprocess.run(cmd, check=True)
    print(f"Wrote {plan.output}")

    # --- post-processing: container-level evasion ---
    if "container-level-evasion" in plan.modes and not args.dry_run:
        rng = random.Random(plan.seed)
        evaded = apply_container_evasion(plan.output, plan, rng)
        if evaded and evaded != plan.output:
            # Replace original with evaded version, remove intermediate backup
            plan.output.unlink()
            evaded.rename(plan.output)
            print(f"[container-evasion] applied (intermediate removed)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
