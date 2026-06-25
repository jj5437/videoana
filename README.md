# Videoana — 视频处理与反检测工具集

用于短剧/视频片段的可见样式处理与深度反检测处理，支持平台重复检测鲁棒性测试。

---

## 环境依赖

```bash
# 核心依赖
python3 -m pip install opencv-python rapidocr-onnxruntime

# 可选：ASR 语音识别（字幕提取）
python3 -m pip install faster-whisper

# 系统依赖
# macOS: brew install ffmpeg tesseract
# Ubuntu: sudo apt install ffmpeg tesseract-ocr
```

---

## 脚本一览

| 脚本 | 用途 |
|------|------|
| `make_clip_style_video.py` | **主处理脚本** — 给原视频加剪辑样式 + 深度反检测处理 |
| `extract_template_overlay.py` | **模板生成** — 从背景图生成带透明窗口的模板 PNG |
| `analyze_video_treatments.py` | **视频分析** — 对比编辑版与原版的画面差异分析 |

---

## 1. make_clip_style_video.py — 视频处理主脚本

### 基础用法（仅可见样式处理）

给原视频加上字幕、横幅、边框等常见剪辑样式：

```bash
python3 make_clip_style_video.py \
  --input input_videos/2.mp4 \
  --output output_videos/2.mp4 \
  --title "安宁玉的前行之路"
```

### 带 OCR 字幕提取 + 模板

从原视频硬字幕区域 OCR 提取字幕，再叠加模板边框：

```bash
python3 make_clip_style_video.py \
  --input input_videos/2.mp4 \
  --output output_videos/2.mp4 \
  --subtitle-source ocr \
  --ocr-engine rapidocr \
  --ocr-sample-interval 1.0 \
  --ocr-crop-y-ratio 0.55 \
  --ocr-crop-height-ratio 0.33 \
  --template-png image_templates/img11_template.png \
  --template-content-rect-ratio 0.055,0.075,0.89,0.84 \
  --title "安宁玉的前行之路"
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| `--subtitle-source ocr` | 从视频画面 OCR 提取原字幕（硬字幕视频适用） |
| `--ocr-engine rapidocr` | OCR 引擎：rapidocr（轻量 ONNX）或 tesseract |
| `--ocr-sample-interval 1.0` | 每 1 秒抽一帧做 OCR |
| `--ocr-crop-y-ratio 0.55` | 字幕区域起始位置占画面高度比例 |
| `--ocr-crop-height-ratio 0.33` | 字幕区域高度占画面高度比例 |
| `--template-png` | 透明模板 PNG 路径，覆盖在最上层 |
| `--template-content-rect-ratio` | 模板透明窗口比例 `x,y,w,h`；原视频画面会被裁剪进这个窗口 |
| `--title` | 顶部横幅剧名 |

### 深度反检测处理

在可见样式基础上，叠加不可见水印破坏、帧级扰动、音频指纹规避、Hash 规避等处理：

```bash
python3 make_clip_style_video.py \
  --input input_videos/2.mp4 \
  --output output_videos/2.mp4 \
  --subtitle-source ocr \
  --ocr-engine rapidocr \
  --ocr-sample-interval 1.0 \
  --ocr-crop-y-ratio 0.55 \
  --ocr-crop-height-ratio 0.33 \
  --template-png image_templates/img11_template.png \
  --template-content-rect-ratio 0.055,0.075,0.89,0.84 \
  --title "安宁玉的前行之路" \
  --mode invisible-watermark-destruction \
  --mode frame-random-perturbation \
  --mode audio-fingerprint-evasion \
  --mode hash-evasion-chain \
  --deep-intensity 0.5
```

**深度模式说明：**

| 模式 | 作用 |
|------|------|
| `invisible-watermark-destruction` | 不可见水印破坏：微旋转 + 缩放扰动 + DCT 域噪声 |
| `frame-random-perturbation` | 帧级随机扰动：每帧亮度/色相/饱和度/对比度抖动 |
| `audio-fingerprint-evasion` | 音频指纹规避：音高偏移 + 时间拉伸 + EQ 切槽 + 相位涂抹 + 动态压缩 |
| `hash-evasion-chain` | Hash 规避：x264 参数随机化 + CRF 微变 + GOP 抖动 + 假元数据 + SEI 注入 |
| `film-grain-synthesis` | 合成胶片颗粒：时域相关噪声，模拟 16mm/35mm 胶片质感 |
| `color-space-jitter` | 色彩空间扰动：BT.601↔BT.709 矩阵转换 + Gamma 微调 + 曲线调整 |
| `temporal-filter-jitter` | 时域滤波抖动：hqdn3d 去噪 + unsharp 锐化交替 |
| `block-boundary-perturbation` | 块边界扰动：非对齐裁剪 + 重填充，打乱宏块网格 |
| `container-level-evasion` | 容器级规避：MKV 转封装消除容器历史，或改变分片设置 |

**强度控制：**

| 参数 | 说明 |
|------|------|
| `--deep-intensity 0.5` | 全局强度 0.0–1.0，默认 0.5 |
| `--watermark-destruction-intensity` | 单独控制水印破坏强度 |
| `--frame-perturbation-intensity` | 单独控制帧扰动强度 |
| `--audio-evasion-intensity` | 单独控制音频规避强度 |
| `--hash-evasion-intensity` | 单独控制 Hash 规避强度 |
| `--disable-style` | 仅做深度处理，跳过所有可见样式（字幕、边框等） |

### 其他常用参数

```bash
# 只处理前 N 秒（快速测试）
--duration 5

# 指定输出分辨率
--width 720 --height 1280

# 指定字体（中文字幕必需）
--fontfile /System/Library/Fonts/PingFang.ttc

# 只打印 FFmpeg 命令，不执行
--dry-run

# 使用内嵌字幕/外挂字幕文件
--subtitle-source embedded      # 提取视频内嵌字幕
--subtitle-source sidecar       # 使用同目录 .srt/.ass
--subtitle-file subtitles.srt   # 指定字幕文件
```

---

## 2. extract_template_overlay.py — 模板生成

从提供的背景图生成中间挖空的透明模板 PNG，用于 `make_clip_style_video.py` 的 `--template-png`。

### 基础用法

```bash
python3 extract_template_overlay.py \
  --input analysis_output/clean_backgrounds_720x1280 \
  --output image_templates/ \
  --width 720 --height 1280 \
  --rect-ratio 0.055,0.075,0.89,0.84 \
  --feather 2 \
  --border-width 3 \
  --border-color 230,230,230,230
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| `--input` | 输入图片或图片目录（支持批量处理） |
| `--output` | 输出 PNG；输入为目录时作为输出目录 |
| `--width` / `--height` | 输出模板分辨率 |
| `--rect-ratio` | 透明窗口比例 `x,y,w,h`（0–1），例如 `0.055,0.075,0.89,0.84` |
| `--rect` | 透明窗口像素坐标 `x,y,w,h`（与 `--rect-ratio` 二选一） |
| `--feather` | 透明窗口边缘羽化像素数 |
| `--border-width` | 窗口边框宽度（像素） |
| `--border-color` | 边框颜色，OpenCV BGRA 格式：`b,g,r,a` |
| `--title` | 在模板顶部背景区域画剧名 |
| `--title-y` | 剧名 Y 坐标 |
| `--title-font-size` | 剧名字号 |
| `--title-color` | 剧名颜色，PIL RGBA：`r,g,b,a` |

### 带剧名的模板

```bash
python3 extract_template_overlay.py \
  --input background.png \
  --output template_with_title.png \
  --width 720 --height 1280 \
  --rect-ratio 0.055,0.075,0.89,0.84 \
  --title "安宁玉的前行之路" \
  --title-y 40 \
  --title-font-size 36 \
  --title-color 255,255,255,255 \
  --title-outline-color 0,0,0,255 \
  --title-outline-width 2
```

---

## 3. analyze_video_treatments.py — 视频分析

对比编辑版（ viral 剪辑）与原版的画面差异，输出分析报告和对比图。

### 基础用法

```bash
python3 analyze_video_treatments.py \
  --edited output_videos/2.mp4 \
  --raw input_videos/2.mp4 \
  --out analysis_output/
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| `--edited` | 编辑版/剪辑版视频路径 |
| `--raw` | 原版/原始视频路径（可选） |
| `--out` | 输出目录，存放分析图、对比图、report.md |
| `--interval` | 采样间隔秒数（默认 1.0） |
| `--max-frames` | 每视频最大采样帧数 |
| `--raw-compare-interval` | 原视频采样间隔（用于帧匹配） |

### 输出文件

| 文件 | 说明 |
|------|------|
| `analysis.json` | 结构化分析数据 |
| `report.md` | 文字报告 |
| `raw_contact_sheet.jpg` / `edited_contact_sheet.jpg` | 帧缩略图拼接 |
| `raw_static_overlay_heatmap.jpg` | 静态元素热力图 |
| `edited_static_overlay_heatmap.jpg` | 编辑版静态元素热力图 |
| `extracted_template_from_video.png` | 从编辑版提取的模板 overlay |

---

## 典型工作流

```bash
# 1. 分析编辑版与原版的差异
python3 analyze_video_treatments.py \
  --edited viral_clip.mp4 \
  --raw original_drama.mp4 \
  --out analysis_output/

# 2. 从分析结果中提取干净背景，生成模板 PNG
python3 extract_template_overlay.py \
  --input analysis_output/clean_backgrounds_720x1280 \
  --output image_templates/ \
  --width 720 --height 1280 \
  --rect-ratio 0.055,0.075,0.89,0.84

# 3. 用模板处理新视频，叠加深度反检测
python3 make_clip_style_video.py \
  --input input_videos/2.mp4 \
  --output output_videos/2.mp4 \
  --subtitle-source ocr \
  --ocr-engine rapidocr \
  --template-png image_templates/img11_template.png \
  --template-content-rect-ratio 0.055,0.075,0.89,0.84 \
  --title "安宁玉的前行之路" \
  --mode invisible-watermark-destruction \
  --mode frame-random-perturbation \
  --mode audio-fingerprint-evasion \
  --mode hash-evasion-chain \
  --deep-intensity 0.5
```

---

## Git 仓库

**https://github.com/jj5437/videoana**

> 注意：`images/` 和 `image_templates/` 目录下的 PNG 资源文件体积较大（~85MB），已排除在版本控制外。使用时请自行准备或从其他渠道获取。
