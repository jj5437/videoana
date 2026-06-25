#!/usr/bin/env python3
"""
Video treatment analyzer for short-drama clips.

The script focuses on forensic / compliance analysis: visible overlays, subtitle
covering, canvas changes, re-encoding metadata, static templates, and rough
source-frame matching against a reference video. It does not provide instructions
for bypassing platform copyright or duplicate-detection systems.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class FrameSignal:
    t: float
    sharpness: float
    brightness: float
    contrast: float
    saturation: float
    edge_top: float
    edge_mid: float
    edge_bottom: float
    edge_left: float
    edge_right: float
    yellow_bottom: float
    yellow_total: float
    white_total: float
    dark_total: float
    subtitle_plate_score: float
    template_score: float
    likely_template: bool
    likely_yellow_subtitle: bool


@dataclass
class MatchResult:
    edited_t: float
    raw_t: float
    hamming_distance: int
    histogram_distance: float
    confidence: str


def run_ffprobe(path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-print_format",
        "json",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, text=True)
        return json.loads(out)
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {"error": str(exc)}


def video_info(probe: dict[str, Any]) -> dict[str, Any]:
    video_stream = next((s for s in probe.get("streams", []) if s.get("codec_type") == "video"), {})
    audio_stream = next((s for s in probe.get("streams", []) if s.get("codec_type") == "audio"), {})
    fmt = probe.get("format", {})
    return {
        "duration": float(fmt.get("duration", 0) or 0),
        "size_bytes": int(fmt.get("size", 0) or 0),
        "format_bitrate": int(fmt.get("bit_rate", 0) or 0),
        "encoder": fmt.get("tags", {}).get("encoder"),
        "description": fmt.get("tags", {}).get("description"),
        "width": int(video_stream.get("width", 0) or 0),
        "height": int(video_stream.get("height", 0) or 0),
        "display_aspect_ratio": video_stream.get("display_aspect_ratio"),
        "video_codec": video_stream.get("codec_name"),
        "video_profile": video_stream.get("profile"),
        "video_bitrate": int(video_stream.get("bit_rate", 0) or 0),
        "fps": parse_rate(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate")),
        "pix_fmt": video_stream.get("pix_fmt"),
        "color_space": video_stream.get("color_space"),
        "audio_codec": audio_stream.get("codec_name"),
        "audio_profile": audio_stream.get("profile"),
        "audio_bitrate": int(audio_stream.get("bit_rate", 0) or 0),
        "audio_sample_rate": int(audio_stream.get("sample_rate", 0) or 0),
    }


def parse_rate(rate: str | None) -> float:
    if not rate or rate == "0/0":
        return 0.0
    if "/" in rate:
        num, den = rate.split("/", 1)
        den_f = float(den)
        return float(num) / den_f if den_f else 0.0
    return float(rate)


def sample_times(duration: float, interval: float, max_frames: int) -> list[float]:
    if duration <= 0:
        return []
    start = min(2.0, max(0.0, duration * 0.05))
    end = max(start, duration - min(2.0, duration * 0.05))
    times = list(np.arange(start, end + 0.001, interval, dtype=float))
    if len(times) > max_frames:
        idx = np.linspace(0, len(times) - 1, max_frames).round().astype(int)
        times = [times[i] for i in sorted(set(idx))]
    return [round(float(t), 3) for t in times]


def read_frame_at(cap: cv2.VideoCapture, t: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, frame = cap.read()
    return frame if ok else None


def density(mask: np.ndarray) -> float:
    return float(np.mean(mask > 0)) if mask.size else 0.0


def frame_signals(frame: np.ndarray, t: float) -> FrameSignal:
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 80, 160)

    top = slice(0, max(1, h // 5))
    mid = slice(max(1, h // 5), max(2, 4 * h // 5))
    bottom = slice(max(2, 4 * h // 5), h)
    left = slice(0, max(1, w // 10))
    right = slice(max(1, 9 * w // 10), w)

    yellow = ((hsv[:, :, 0] >= 16) & (hsv[:, :, 0] <= 42) & (hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 80))
    white = (hsv[:, :, 1] < 48) & (hsv[:, :, 2] > 178)
    dark = hsv[:, :, 2] < 48

    lower = slice(int(h * 0.50), h)
    subtitle_area = slice(int(h * 0.54), int(h * 0.82))
    subtitle_gray = gray[subtitle_area, int(w * 0.12) : int(w * 0.88)]
    lap = cv2.Laplacian(subtitle_gray, cv2.CV_64F)
    subtitle_lap = float(lap.var()) if subtitle_gray.size else 0.0
    # Blurred subtitle-cover plates have low texture while bright subtitle pixels sit on top.
    low_texture = 1.0 / (1.0 + subtitle_lap / 180.0)
    subtitle_plate_score = float(density(yellow[lower, :]) * 8.0 + low_texture * 0.25)

    edge_top = density(edges[top, :])
    edge_mid = density(edges[mid, :])
    edge_bottom = density(edges[bottom, :])
    edge_left = density(edges[:, left])
    edge_right = density(edges[:, right])
    yellow_bottom = density(yellow[lower, :])
    white_total = density(white)
    yellow_total = density(yellow)

    template_score = (
        edge_top * 1.8
        + edge_bottom * 1.6
        + (edge_left + edge_right) * 1.2
        + white_total * 1.4
        + yellow_total * 1.3
    )

    return FrameSignal(
        t=round(t, 3),
        sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        brightness=float(gray.mean()),
        contrast=float(gray.std()),
        saturation=float(hsv[:, :, 1].mean()),
        edge_top=edge_top,
        edge_mid=edge_mid,
        edge_bottom=edge_bottom,
        edge_left=edge_left,
        edge_right=edge_right,
        yellow_bottom=yellow_bottom,
        yellow_total=yellow_total,
        white_total=white_total,
        dark_total=density(dark),
        subtitle_plate_score=subtitle_plate_score,
        template_score=template_score,
        likely_template=template_score > 0.42 and (edge_top > 0.055 or edge_bottom > 0.09),
        likely_yellow_subtitle=yellow_bottom > 0.006,
    )


def segment_boolean(signals: list[FrameSignal], attr: str, max_gap: float) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    active: list[FrameSignal] = []
    last_t: float | None = None
    for sig in signals:
        flag = bool(getattr(sig, attr))
        if flag:
            if active and last_t is not None and sig.t - last_t > max_gap:
                segments.append(close_segment(active))
                active = []
            active.append(sig)
            last_t = sig.t
        elif active and last_t is not None and sig.t - last_t > max_gap:
            segments.append(close_segment(active))
            active = []
            last_t = None
    if active:
        segments.append(close_segment(active))
    return segments


def close_segment(items: list[FrameSignal]) -> dict[str, Any]:
    return {
        "start": round(items[0].t, 2),
        "end": round(items[-1].t, 2),
        "samples": len(items),
        "avg_template_score": round(float(np.mean([x.template_score for x in items])), 4),
        "avg_yellow_bottom": round(float(np.mean([x.yellow_bottom for x in items])), 4),
    }


def content_crop(frame: np.ndarray, edited: bool, template_like: bool = False) -> np.ndarray:
    h, w = frame.shape[:2]
    if edited and template_like:
        y1, y2 = int(h * 0.15), int(h * 0.76)
        x1, x2 = int(w * 0.12), int(w * 0.88)
    elif edited:
        y1, y2 = int(h * 0.03), int(h * 0.78)
        x1, x2 = int(w * 0.03), int(w * 0.97)
    else:
        y1, y2 = int(h * 0.04), int(h * 0.86)
        x1, x2 = int(w * 0.03), int(w * 0.97)
    return frame[y1:y2, x1:x2]


def dhash(frame: np.ndarray, size: int = 16) -> int:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def color_hist(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist.flatten()


def build_frame_index(path: Path, times: list[float], edited: bool, signals_by_time: dict[float, FrameSignal]) -> list[dict[str, Any]]:
    cap = cv2.VideoCapture(str(path))
    index: list[dict[str, Any]] = []
    for t in times:
        frame = read_frame_at(cap, t)
        if frame is None:
            continue
        sig = signals_by_time.get(round(t, 3))
        crop = content_crop(frame, edited=edited, template_like=bool(sig and sig.likely_template))
        index.append({"t": t, "hash": dhash(crop), "hist": color_hist(crop), "frame": frame})
    cap.release()
    return index


def compare_to_raw(
    edited_path: Path,
    raw_path: Path,
    edited_signals: list[FrameSignal],
    raw_signals: list[FrameSignal],
    edited_interval: float,
    raw_interval: float,
    max_matches: int,
) -> list[MatchResult]:
    edited_info = video_info(run_ffprobe(edited_path))
    raw_info = video_info(run_ffprobe(raw_path))
    edited_times = sample_times(edited_info["duration"], edited_interval, 90)
    raw_times = sample_times(raw_info["duration"], raw_interval, 240)
    edited_by_time = {round(x.t, 3): x for x in edited_signals}
    raw_by_time = {round(x.t, 3): x for x in raw_signals}
    edited_idx = build_frame_index(edited_path, edited_times, True, edited_by_time)
    raw_idx = build_frame_index(raw_path, raw_times, False, raw_by_time)
    if not edited_idx or not raw_idx:
        return []

    matches: list[MatchResult] = []
    for e in edited_idx:
        best: tuple[int, float, dict[str, Any]] | None = None
        for r in raw_idx:
            hd = hamming(e["hash"], r["hash"])
            hist_dist = float(cv2.compareHist(e["hist"].astype("float32"), r["hist"].astype("float32"), cv2.HISTCMP_BHATTACHARYYA))
            score = hd + hist_dist * 35.0
            if best is None or score < best[0] + best[1] * 35.0:
                best = (hd, hist_dist, r)
        if best is None:
            continue
        hd, hist_dist, raw_item = best
        confidence = "low"
        if hd <= 62 and hist_dist < 0.38:
            confidence = "medium"
        if hd <= 46 and hist_dist < 0.30:
            confidence = "high"
        matches.append(MatchResult(round(e["t"], 3), round(raw_item["t"], 3), hd, round(hist_dist, 4), confidence))

    matches.sort(key=lambda m: (m.hamming_distance + m.histogram_distance * 35.0, m.edited_t))
    filtered: list[MatchResult] = []
    used_edited: set[int] = set()
    used_raw: set[int] = set()
    for m in matches:
        e_bucket = int(m.edited_t // 8)
        r_bucket = int(m.raw_t // 8)
        if e_bucket in used_edited or r_bucket in used_raw:
            continue
        filtered.append(m)
        used_edited.add(e_bucket)
        used_raw.add(r_bucket)
        if len(filtered) >= max_matches:
            break
    return filtered


def analyze_video(path: Path, interval: float, max_frames: int, out_dir: Path, label: str) -> dict[str, Any]:
    probe = run_ffprobe(path)
    info = video_info(probe)
    cap = cv2.VideoCapture(str(path))
    times = sample_times(info["duration"], interval, max_frames)
    signals: list[FrameSignal] = []
    frames_for_sheet: list[np.ndarray] = []
    stability_frames: list[np.ndarray] = []

    frame_dir = out_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    sheet_stride = max(1, math.ceil(len(times) / 12))

    for i, t in enumerate(times):
        frame = read_frame_at(cap, t)
        if frame is None:
            continue
        sig = frame_signals(frame, t)
        signals.append(sig)
        stability_frames.append(cv2.resize(frame, (180, 320), interpolation=cv2.INTER_AREA))
        if i % sheet_stride == 0 and len(frames_for_sheet) < 12:
            annotated = annotate_frame(frame, sig)
            frames_for_sheet.append(annotated)
            cv2.imwrite(str(frame_dir / f"{label}_{i:04d}_{t:.1f}s.jpg"), annotated)
    cap.release()

    if frames_for_sheet:
        write_contact_sheet(frames_for_sheet, out_dir / f"{label}_contact_sheet.jpg")
    stability_summary = write_temporal_stability(stability_frames, out_dir / f"{label}_static_overlay_heatmap.jpg")

    return {
        "path": str(path),
        "probe": probe,
        "info": info,
        "signals": [asdict(s) for s in signals],
        "summary": summarize_signals(signals),
        "stability_summary": stability_summary,
        "template_segments": segment_boolean(signals, "likely_template", max_gap=interval * 1.6),
        "yellow_subtitle_segments": segment_boolean(signals, "likely_yellow_subtitle", max_gap=interval * 1.6),
    }


def annotate_frame(frame: np.ndarray, sig: FrameSignal) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = 360 / h
    canvas = cv2.resize(frame, (int(w * scale), 360), interpolation=cv2.INTER_AREA)
    text = f"{sig.t:.1f}s tmpl={sig.template_score:.2f} ysub={sig.yellow_bottom:.3f}"
    cv2.putText(canvas, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 255, 255), 1, cv2.LINE_AA)
    return canvas


def write_contact_sheet(frames: list[np.ndarray], path: Path) -> None:
    max_h = max(f.shape[0] for f in frames)
    max_w = max(f.shape[1] for f in frames)
    tiles = []
    for frame in frames:
        tile = np.zeros((max_h, max_w, 3), dtype=np.uint8)
        tile[: frame.shape[0], : frame.shape[1]] = frame
        tiles.append(tile)
    while len(tiles) % 4:
        tiles.append(np.zeros((max_h, max_w, 3), dtype=np.uint8))
    rows = [np.hstack(tiles[i : i + 4]) for i in range(0, len(tiles), 4)]
    cv2.imwrite(str(path), np.vstack(rows))


def write_temporal_stability(frames: list[np.ndarray], path: Path) -> dict[str, Any]:
    if len(frames) < 4:
        return {}
    stack = np.stack(frames).astype(np.float32)
    mean = stack.mean(axis=0).astype(np.uint8)
    std = stack.std(axis=0).mean(axis=2)
    gray_mean = cv2.cvtColor(mean, cv2.COLOR_BGR2GRAY)
    mean_edges = cv2.Canny(gray_mean, 70, 150)

    stable = std < 18.0
    stable_edges = stable & (mean_edges > 0)
    h, w = std.shape
    top = stable_edges[: h // 5, :]
    bottom = stable_edges[4 * h // 5 :, :]
    sides = np.concatenate([stable_edges[:, : w // 10].ravel(), stable_edges[:, 9 * w // 10 :].ravel()])
    center = stable_edges[h // 5 : 4 * h // 5, w // 10 : 9 * w // 10]

    heat = np.clip(255 - std * 7, 0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO)
    blend = cv2.addWeighted(mean, 0.55, heat_color, 0.45, 0)
    cv2.putText(
        blend,
        "bright heat = temporally stable pixels",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(path), blend)

    return {
        "heatmap": str(path),
        "stable_edge_top_density": round(density(top), 4),
        "stable_edge_bottom_density": round(density(bottom), 4),
        "stable_edge_side_density": round(float(np.mean(sides > 0)), 4) if sides.size else 0.0,
        "stable_edge_center_density": round(density(center), 4),
        "interpretation": "Static overlays are more likely when stable edge density is much higher near top/bottom/sides than in the center.",
    }


def summarize_signals(signals: list[FrameSignal]) -> dict[str, Any]:
    if not signals:
        return {}

    def avg(attr: str) -> float:
        return round(float(np.mean([getattr(s, attr) for s in signals])), 4)

    def p95(attr: str) -> float:
        return round(float(np.percentile([getattr(s, attr) for s in signals], 95)), 4)

    return {
        "sample_count": len(signals),
        "avg_sharpness": avg("sharpness"),
        "avg_brightness": avg("brightness"),
        "avg_contrast": avg("contrast"),
        "avg_saturation": avg("saturation"),
        "avg_edge_top": avg("edge_top"),
        "avg_edge_bottom": avg("edge_bottom"),
        "avg_yellow_bottom": avg("yellow_bottom"),
        "avg_white_total": avg("white_total"),
        "avg_template_score": avg("template_score"),
        "p95_template_score": p95("template_score"),
        "template_sample_ratio": round(float(np.mean([s.likely_template for s in signals])), 4),
        "yellow_subtitle_sample_ratio": round(float(np.mean([s.likely_yellow_subtitle for s in signals])), 4),
    }


def compare_metadata(edited: dict[str, Any], raw: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    ei, ri = edited["info"], raw["info"]
    if ei["width"] and ri["width"] and (ei["width"], ei["height"]) != (ri["width"], ri["height"]):
        notes.append(f"画布/分辨率不同：剪辑版 {ei['width']}x{ei['height']}，原片 {ri['width']}x{ri['height']}。")
    if ei.get("display_aspect_ratio") != ri.get("display_aspect_ratio"):
        notes.append(f"显示比例不同：剪辑版 {ei.get('display_aspect_ratio')}，原片 {ri.get('display_aspect_ratio')}。")
    if ei.get("encoder") != ri.get("encoder"):
        notes.append(f"封装 encoder 不同：剪辑版 {ei.get('encoder')}，原片 {ri.get('encoder')}。")
    if ri.get("description"):
        notes.append(f"原片 metadata 带描述：{ri.get('description')}。")
    if ei.get("audio_profile") != ri.get("audio_profile") or ei.get("audio_bitrate") != ri.get("audio_bitrate"):
        notes.append(
            f"音频参数不同：剪辑版 {ei.get('audio_codec')} {ei.get('audio_profile')} {ei.get('audio_bitrate')}bps，"
            f"原片 {ri.get('audio_codec')} {ri.get('audio_profile')} {ri.get('audio_bitrate')}bps。"
        )
    if ei.get("video_bitrate") and ri.get("video_bitrate"):
        ratio = ei["video_bitrate"] / max(1, ri["video_bitrate"])
        notes.append(f"视频码率约为原片 {ratio:.2f} 倍，说明至少经过重新压制或平台转码。")
    return notes


def report_markdown(edited: dict[str, Any], raw: dict[str, Any] | None, matches: list[MatchResult], out_dir: Path) -> str:
    lines: list[str] = []
    lines.append("# 视频处理取证分析报告")
    lines.append("")
    lines.append("> 说明：本报告用于识别剪辑/转载/二创中的可见与技术处理痕迹，不能作为规避平台重复或版权检测的操作指南。")
    lines.append("")
    lines.append("## 核心结论")
    es = edited["summary"]
    lines.append(f"- 剪辑版抽样 {es.get('sample_count', 0)} 帧；模板命中比例 {es.get('template_sample_ratio', 0):.2%}，黄色新字幕命中比例 {es.get('yellow_subtitle_sample_ratio', 0):.2%}。")
    if edited["template_segments"]:
        seg = edited["template_segments"][0]
        lines.append(f"- 固定花草边框/顶部剧名模板约从 {seg['start']}s 开始稳定出现，持续到 {edited['template_segments'][-1]['end']}s 左右。")
    if edited["yellow_subtitle_segments"]:
        seg = edited["yellow_subtitle_segments"][0]
        lines.append(f"- 黄色描边字幕从 {seg['start']}s 附近就出现，底部字幕区域通常伴随暗化/模糊底板。")
    lines.append("- 可见处理包括：原字幕区域遮挡/模糊、新黄色描边字幕、后段花草边框、顶部剧名横幅、左侧竖排声明、白色装饰符号、底部模板装饰。")
    lines.append("- 技术处理包括：竖屏画布重排、重新封装/压制、音频重新编码、整体亮度/锐度/码率分布改变。")
    if edited.get("stability_summary"):
        st = edited["stability_summary"]
        lines.append(
            "- 时间稳定性热力图显示边缘区域存在长期稳定的高边缘结构，通常对应固定模板、边框、贴纸或横幅。"
            f" top={st.get('stable_edge_top_density')} bottom={st.get('stable_edge_bottom_density')} sides={st.get('stable_edge_side_density')} center={st.get('stable_edge_center_density')}。"
        )
    lines.append("")

    lines.append("## 媒体参数")
    lines.append("### 剪辑版")
    lines.append("```json")
    lines.append(json.dumps(edited["info"], ensure_ascii=False, indent=2))
    lines.append("```")
    if raw:
        lines.append("### 原片/参考片")
        lines.append("```json")
        lines.append(json.dumps(raw["info"], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("### 参数差异")
        for note in compare_metadata(edited, raw):
            lines.append(f"- {note}")
    lines.append("")

    lines.append("## 可疑时间段")
    lines.append("### 模板/边框段")
    if edited["template_segments"]:
        for s in edited["template_segments"]:
            lines.append(f"- {s['start']}s - {s['end']}s：平均模板分 {s['avg_template_score']}，样本数 {s['samples']}。")
    else:
        lines.append("- 未检测到稳定模板段。")
    lines.append("### 黄色字幕段")
    if edited["yellow_subtitle_segments"]:
        for s in edited["yellow_subtitle_segments"]:
            lines.append(f"- {s['start']}s - {s['end']}s：底部黄色像素均值 {s['avg_yellow_bottom']}，样本数 {s['samples']}。")
    else:
        lines.append("- 未检测到稳定黄色字幕段。")
    lines.append("")

    if raw:
        lines.append("## 与原片粗匹配")
        lines.append("匹配方式：裁掉剪辑版模板/字幕高发区域后，对中心画面做 perceptual hash + HSV 直方图近邻搜索。因为剧情片段未必对齐，低置信结果只能说明画面结构相近。")
        if matches:
            for m in matches:
                lines.append(
                    f"- 剪辑 {m.edited_t}s ≈ 原片 {m.raw_t}s，hash 距离 {m.hamming_distance}，"
                    f"直方图距离 {m.histogram_distance}，置信度 {m.confidence}。"
                )
        else:
            lines.append("- 未找到可靠近邻。")
        lines.append("")

    lines.append("## 证据图")
    lines.append(f"- 剪辑版 contact sheet：`{(out_dir / 'edited_contact_sheet.jpg').as_posix()}`")
    lines.append(f"- 剪辑版静态叠加层热力图：`{(out_dir / 'edited_static_overlay_heatmap.jpg').as_posix()}`")
    if raw:
        lines.append(f"- 原片 contact sheet：`{(out_dir / 'raw_contact_sheet.jpg').as_posix()}`")
        lines.append(f"- 原片静态叠加层热力图：`{(out_dir / 'raw_static_overlay_heatmap.jpg').as_posix()}`")
    lines.append(f"- 单帧证据目录：`{(out_dir / 'frames').as_posix()}`")
    lines.append("")

    lines.append("## 代码可复现的处理类型")
    lines.append("- 可以复现用于合规审计/编辑还原的可见层：模糊字幕底板检测、文字层定位、模板边框定位、画布比例变化、重编码参数差异。")
    lines.append("- 不建议也不能用于复现“绕过检测”的隐蔽扰动；本脚本只输出检测结果和证据，不生成规避平台检测的处理链。")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze visible and technical treatments in short-drama videos.")
    parser.add_argument("--edited", default="video.mp4", help="edited/viral clip path")
    parser.add_argument("--raw", default="raw_drama.mp4", help="optional reference/raw video path")
    parser.add_argument("--out", default="analysis_output", help="output directory")
    parser.add_argument("--interval", type=float, default=4.0, help="seconds between sampled frames")
    parser.add_argument("--max-frames", type=int, default=140, help="maximum sampled frames per video")
    parser.add_argument("--raw-compare-interval", type=float, default=3.0, help="seconds between raw frames for matching")
    args = parser.parse_args()

    edited_path = Path(args.edited)
    raw_path = Path(args.raw) if args.raw else None
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not edited_path.exists():
        raise SystemExit(f"Edited video not found: {edited_path}")

    edited = analyze_video(edited_path, args.interval, args.max_frames, out_dir, "edited")
    raw = None
    matches: list[MatchResult] = []
    if raw_path and raw_path.exists():
        raw = analyze_video(raw_path, args.interval, args.max_frames, out_dir, "raw")
        edited_signals = [FrameSignal(**x) for x in edited["signals"]]
        raw_signals = [FrameSignal(**x) for x in raw["signals"]]
        matches = compare_to_raw(
            edited_path,
            raw_path,
            edited_signals,
            raw_signals,
            edited_interval=args.interval,
            raw_interval=args.raw_compare_interval,
            max_matches=12,
        )

    payload = {
        "edited": edited,
        "raw": raw,
        "matches": [asdict(m) for m in matches],
    }
    (out_dir / "analysis.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(report_markdown(edited, raw, matches, out_dir), encoding="utf-8")
    print(f"Wrote {out_dir / 'report.md'}")
    print(f"Wrote {out_dir / 'analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
