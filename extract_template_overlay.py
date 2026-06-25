#!/usr/bin/env python3
"""
Build transparent background templates from provided images.

This script intentionally does not extract templates from edited videos. It takes
clean background images, resizes/crops them to the target canvas, punches a
transparent content window, and optionally draws a border around that window.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def resize_cover(image: np.ndarray, width: int, height: int) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    scale = max(width / src_w, height / src_h)
    scaled_w = int(round(src_w * scale))
    scaled_h = int(round(src_h * scale))
    resized = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
    x = max(0, (scaled_w - width) // 2)
    y = max(0, (scaled_h - height) // 2)
    return resized[y : y + height, x : x + width]


def punch_transparent_window(
    image: np.ndarray,
    rect: tuple[int, int, int, int],
    *,
    feather: int = 0,
    border_width: int = 3,
    border_color: tuple[int, int, int, int] = (230, 230, 230, 220),
) -> np.ndarray:
    if image.shape[2] == 3:
        rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = 255
    else:
        rgba = image.copy()

    x, y, w, h = rect
    x = max(0, min(rgba.shape[1] - 1, x))
    y = max(0, min(rgba.shape[0] - 1, y))
    w = max(1, min(rgba.shape[1] - x, w))
    h = max(1, min(rgba.shape[0] - y, h))

    alpha = rgba[:, :, 3].copy()
    alpha[y : y + h, x : x + w] = 0
    if feather > 0:
        k = feather * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
        alpha[y + feather : y + h - feather, x + feather : x + w - feather] = 0
    rgba[:, :, 3] = alpha

    if border_width > 0:
        color = tuple(int(v) for v in border_color)
        cv2.rectangle(rgba, (x, y), (x + w - 1, y + h - 1), color, border_width, lineType=cv2.LINE_AA)
    return rgba


def rect_from_ratios(width: int, height: int, ratios: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    x, y, w, h = ratios
    return (int(round(width * x)), int(round(height * y)), int(round(width * w)), int(round(height * h)))


def parse_rect(value: str) -> tuple[int, int, int, int]:
    parts = [int(x.strip()) for x in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("rect 格式应为 x,y,w,h")
    return tuple(parts)  # type: ignore[return-value]


def parse_rect_ratio(value: str) -> tuple[float, float, float, float]:
    parts = [float(x.strip()) for x in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("rect-ratio 格式应为 x,y,w,h")
    return tuple(parts)  # type: ignore[return-value]


def parse_rgba(value: str) -> tuple[int, int, int, int]:
    parts = [int(x.strip()) for x in value.split(",")]
    if len(parts) not in {3, 4}:
        raise argparse.ArgumentTypeError("颜色格式应为 b,g,r 或 b,g,r,a")
    if len(parts) == 3:
        parts.append(255)
    return tuple(parts)  # type: ignore[return-value]


def parse_rgb_or_rgba(value: str) -> tuple[int, int, int, int]:
    parts = [int(x.strip()) for x in value.split(",")]
    if len(parts) not in {3, 4}:
        raise argparse.ArgumentTypeError("颜色格式应为 r,g,b 或 r,g,b,a")
    if len(parts) == 3:
        parts.append(255)
    return tuple(parts)  # type: ignore[return-value]


def load_title_font(font_path: Path | None, font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if font_path:
        candidates.append(font_path)
    candidates.extend(
        Path(p)
        for p in (
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        )
    )
    for candidate in candidates:
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), font_size)
            except OSError:
                continue
    return ImageFont.load_default()


def draw_title(
    overlay: np.ndarray,
    title: str,
    *,
    y: int,
    font_size: int,
    color: tuple[int, int, int, int],
    outline_color: tuple[int, int, int, int],
    outline_width: int,
    band_height: int,
    band_color: tuple[int, int, int, int],
    font_path: Path | None = None,
) -> np.ndarray:
    rgba = cv2.cvtColor(overlay, cv2.COLOR_BGRA2RGBA)
    image = Image.fromarray(rgba)
    draw = ImageDraw.Draw(image, "RGBA")
    if band_height > 0 and band_color[3] > 0:
        draw.rectangle((0, 0, image.width, band_height), fill=band_color)

    font = load_title_font(font_path, font_size)
    bbox = draw.textbbox((0, 0), title, font=font, stroke_width=outline_width)
    text_w = bbox[2] - bbox[0]
    x = max(0, (image.width - text_w) // 2)
    draw.text(
        (x, y),
        title,
        font=font,
        fill=color,
        stroke_width=outline_width,
        stroke_fill=outline_color,
    )
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)


def iter_input_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def build_template(
    input_path: Path,
    output_path: Path,
    *,
    width: int,
    height: int,
    rect: tuple[int, int, int, int],
    feather: int,
    border_width: int,
    border_color: tuple[int, int, int, int],
    title: str | None = None,
    title_y: int = 28,
    title_font_size: int = 44,
    title_color: tuple[int, int, int, int] = (255, 255, 255, 255),
    title_outline_color: tuple[int, int, int, int] = (0, 0, 0, 230),
    title_outline_width: int = 3,
    title_band_height: int = 92,
    title_band_color: tuple[int, int, int, int] = (0, 0, 0, 70),
    title_font: Path | None = None,
) -> None:
    image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise SystemExit(f"无法读取图片：{input_path}")
    canvas = resize_cover(image, width, height)
    overlay = punch_transparent_window(
        canvas,
        rect,
        feather=feather,
        border_width=border_width,
        border_color=border_color,
    )
    if title:
        overlay = draw_title(
            overlay,
            title,
            y=title_y,
            font_size=title_font_size,
            color=title_color,
            outline_color=title_outline_color,
            outline_width=title_outline_width,
            band_height=title_band_height,
            band_color=title_band_color,
            font_path=title_font,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)


def main() -> int:
    parser = argparse.ArgumentParser(description="从提供的背景图片生成中间挖空的透明模板 PNG。")
    parser.add_argument("--input", required=True, type=Path, help="输入图片或图片目录")
    parser.add_argument("--output", required=True, type=Path, help="输出 PNG；输入为目录时作为输出目录")
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=1280)
    parser.add_argument("--rect", type=parse_rect, help="透明窗口 x,y,w,h，像素坐标")
    parser.add_argument("--rect-ratio", type=parse_rect_ratio, default=(0.08, 0.12, 0.84, 0.76), help="透明窗口比例 x,y,w,h")
    parser.add_argument("--feather", type=int, default=2)
    parser.add_argument("--border-width", type=int, default=3)
    parser.add_argument("--border-color", type=parse_rgba, default=(230, 230, 230, 220), help="OpenCV BGRA: b,g,r,a")
    parser.add_argument("--title", help="直接画在模板顶部背景区域的剧名")
    parser.add_argument("--title-y", type=int, default=28)
    parser.add_argument("--title-font-size", type=int, default=44)
    parser.add_argument("--title-color", type=parse_rgb_or_rgba, default=(255, 255, 255, 255), help="PIL RGBA: r,g,b,a")
    parser.add_argument("--title-outline-color", type=parse_rgb_or_rgba, default=(0, 0, 0, 230), help="PIL RGBA: r,g,b,a")
    parser.add_argument("--title-outline-width", type=int, default=3)
    parser.add_argument("--title-band-height", type=int, default=92)
    parser.add_argument("--title-band-color", type=parse_rgb_or_rgba, default=(0, 0, 0, 70), help="PIL RGBA: r,g,b,a")
    parser.add_argument("--title-font", type=Path)
    args = parser.parse_args()

    rect = args.rect or rect_from_ratios(args.width, args.height, args.rect_ratio)
    inputs = iter_input_images(args.input)
    if not inputs:
        raise SystemExit(f"没有找到图片：{args.input}")

    for input_path in inputs:
        output_path = args.output
        if args.input.is_dir():
            output_path = args.output / f"{input_path.stem}_template.png"
        build_template(
            input_path,
            output_path,
            width=args.width,
            height=args.height,
            rect=rect,
            feather=args.feather,
            border_width=args.border_width,
            border_color=args.border_color,
            title=args.title,
            title_y=args.title_y,
            title_font_size=args.title_font_size,
            title_color=args.title_color,
            title_outline_color=args.title_outline_color,
            title_outline_width=args.title_outline_width,
            title_band_height=args.title_band_height,
            title_band_color=args.title_band_color,
            title_font=args.title_font,
        )
        print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
