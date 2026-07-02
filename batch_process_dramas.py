#!/usr/bin/env python3
"""
Batch process full drama series with deep anti-detection processing.

This script walks through all drama folders under input_videos/,
processes every .mp4 file with deep processing modes only,
and writes outputs to output_videos/ preserving the original
relative path structure (same directory names, same filenames).

Visible clip-style treatments (subtitles, banners, borders, title text,
background images) are disabled since these are complete episodes,
not short clips.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# All deep anti-detection modes
DEEP_MODES = [
    "invisible-watermark-destruction",
    "frame-random-perturbation",
    "audio-fingerprint-evasion",
    "hash-evasion-chain",
    "film-grain-synthesis",
    "color-space-jitter",
    "temporal-filter-jitter",
    "block-boundary-perturbation",
    "container-level-evasion",
]

# Drama folders to process
DRAMA_FOLDERS = [
    "家事破局者",
    "面试官竟是我同学",
    "漏进我家的脏水",
]


def probe_resolution(video_path: Path) -> tuple[int, int]:
    """Return (width, height) of the first video stream via ffprobe."""
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "json",
                str(video_path),
            ],
            text=True,
        )
        data = json.loads(out)
        stream = data["streams"][0]
        return int(stream["width"]), int(stream["height"])
    except Exception as exc:
        raise RuntimeError(f"无法探测视频分辨率: {video_path}") from exc


def process_video(
    input_path: Path,
    output_path: Path,
    *,
    deep_intensity: float = 0.5,
    crf: int = 28,
    preset: str = "medium",
    codec: str = "libx264",
    min_bitrate: str | None = None,
    target_bitrate: str | None = "5M",
    dry_run: bool = False,
) -> bool:
    """Process a single video with deep anti-detection modes only."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Keep original resolution
    orig_w, orig_h = probe_resolution(input_path)

    cmd = [
        sys.executable,
        "make_clip_style_video.py",
        "--input", str(input_path),
        "--output", str(output_path),
        "--disable-style",
        "--subtitle-source", "none",
        "--width", str(orig_w),
        "--height", str(orig_h),
        "--preset", preset,
        "--codec", codec,
        "--deep-intensity", str(deep_intensity),
    ]

    if target_bitrate:
        cmd.extend(["--target-bitrate", target_bitrate])
    elif min_bitrate:
        cmd.extend(["--min-bitrate", min_bitrate])
        cmd.extend(["--crf", str(crf)])
    else:
        cmd.extend(["--crf", str(crf)])

    for mode in DEEP_MODES:
        cmd.extend(["--mode", mode])

    if dry_run:
        cmd.append("--dry-run")

    print(f"\n[PROCESS] {input_path}")
    print(f"  -> {output_path}")
    print(f"  Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            for line in result.stdout.strip().splitlines():
                print(f"  {line}")
        return True
    except subprocess.CalledProcessError as exc:
        print(f"[ERROR] Failed to process {input_path}")
        if exc.stdout:
            print(f"  stdout: {exc.stdout}")
        if exc.stderr:
            print(f"  stderr: {exc.stderr}")
        return False


def find_mp4_files(folder: Path) -> list[Path]:
    """Find all .mp4 files in the given folder, sorted naturally."""
    mp4s = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"]
    # Natural sort by filename
    mp4s.sort(key=lambda p: p.name)
    return mp4s


def main() -> int:
    parser = argparse.ArgumentParser(
        description="批量处理完整剧集：仅深度反检测，无可见剪辑样式。"
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("input_videos"),
        help="输入视频根目录，默认 input_videos",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output_videos"),
        help="输出视频根目录，默认 output_videos",
    )
    parser.add_argument(
        "--deep-intensity",
        type=float,
        default=0.5,
        help="深度处理强度 0.0–1.0，默认 0.5",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=28,
        help="视频质量 CRF，默认 28（越小画质越好、文件越大；越大画质越差、文件越小）",
    )
    parser.add_argument(
        "--preset",
        default="medium",
        help="编码预设，默认 medium",
    )
    parser.add_argument(
        "--codec",
        default="libx264",
        choices=["libx264", "libx265"],
        help="视频编码器，默认 libx264；libx265 同等画质体积更小但兼容性略差",
    )
    parser.add_argument(
        "--min-bitrate",
        default=None,
        help="最低视频码率（CRF 模式下兜底），例如 4M",
    )
    parser.add_argument(
        "--target-bitrate",
        default="5M",
        help="目标平均视频码率，默认 5M；设置后改用 ABR 模式严格锁定码率",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印命令，不实际执行",
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        help="指定要处理的文件夹名称（默认处理全部三部剧）",
    )
    args = parser.parse_args()

    input_root: Path = args.input_root
    output_root: Path = args.output_root
    folders_to_process = args.folders or DRAMA_FOLDERS

    total = 0
    success = 0
    failed: list[Path] = []

    for folder_name in folders_to_process:
        drama_input = input_root / folder_name
        if not drama_input.exists():
            print(f"[SKIP] Folder not found: {drama_input}")
            continue

        mp4_files = find_mp4_files(drama_input)
        if not mp4_files:
            print(f"[SKIP] No .mp4 files found in: {drama_input}")
            continue

        print(f"\n{'='*60}")
        print(f"[FOLDER] {folder_name} ({len(mp4_files)} episodes)")
        print(f"{'='*60}")

        for video_path in mp4_files:
            # Preserve relative path: output_root / folder_name / filename
            rel_path = video_path.relative_to(input_root)
            output_path = output_root / rel_path

            ok = process_video(
                video_path,
                output_path,
                deep_intensity=args.deep_intensity,
                crf=args.crf,
                preset=args.preset,
                codec=args.codec,
                min_bitrate=args.min_bitrate,
                target_bitrate=args.target_bitrate,
                dry_run=args.dry_run,
            )
            total += 1
            if ok:
                success += 1
            else:
                failed.append(video_path)

    print(f"\n{'='*60}")
    print(f"[DONE] Total: {total}, Success: {success}, Failed: {len(failed)}")
    if failed:
        print("[FAILED FILES]")
        for p in failed:
            print(f"  - {p}")
    print(f"{'='*60}")

    return 0 if len(failed) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
