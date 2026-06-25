import unittest
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import make_clip_style_video as clip


class TreatmentPlanTests(unittest.TestCase):
    def test_import_does_not_load_cv2_for_asr_path(self):
        self.assertNotIn("cv2", sys.modules)

    def test_builds_visible_clip_style_filter_graph(self):
        plan = clip.TreatmentPlan(
            source=Path("raw_video.mp4"),
            output=Path("out.mp4"),
            width=720,
            height=1280,
            title="测试剧名",
            vertical_text="平台测试",
            captions=["第一句测试字幕", "第二句测试字幕"],
            seed=7,
            caption_total_seconds=12.0,
        )

        graph = clip.build_filter_graph(plan)

        self.assertIn("scale=", graph)
        self.assertIn("pad=720:1280", graph)
        self.assertIn("boxblur=", graph)
        self.assertIn("drawtext=", graph)
        self.assertIn("fontcolor=yellow", graph)
        self.assertIn("borderw=4", graph)
        self.assertIn("测试剧名", graph)
        self.assertIn("平台测试", graph)
        self.assertIn("crop=720:192:0:780", graph)
        self.assertIn("between(t,8.70,12.30)", graph)

    def test_template_png_is_added_as_overlay_input(self):
        plan = clip.TreatmentPlan(
            source=Path("raw_video.mp4"),
            output=Path("out.mp4"),
            template_png=Path("template.png"),
        )

        cmd = clip.build_ffmpeg_command(plan, validate_paths=False)
        filter_complex = cmd[cmd.index("-filter_complex") + 1]

        self.assertIn("-i", cmd)
        self.assertIn("template.png", cmd)
        self.assertIn("[1:v]scale=720:1280[template]", filter_complex)
        self.assertIn("[base][template]overlay=0:0", filter_complex)

    def test_template_content_rect_scales_video_into_window(self):
        plan = clip.TreatmentPlan(
            source=Path("raw_video.mp4"),
            output=Path("out.mp4"),
            template_png=Path("template.png"),
            title="运行时剧名",
            template_content_rect_ratio=(0.055, 0.075, 0.89, 0.84),
        )

        cmd = clip.build_ffmpeg_command(plan, validate_paths=False)
        filter_complex = cmd[cmd.index("-filter_complex") + 1]

        self.assertIn("scale=641:1075:force_original_aspect_ratio=increase", filter_complex)
        self.assertIn("crop=641:1075:(iw-641)/2:(ih-1075)/2", filter_complex)
        self.assertIn("pad=720:1280:40:96:color=black", filter_complex)
        self.assertIn("[1:v]scale=720:1280[template]", filter_complex)
        self.assertIn("[base][template]overlay=0:0[templated]", filter_complex)
        self.assertIn("[templated]drawtext=", filter_complex)
        self.assertIn("运行时剧名", filter_complex)

    def test_caption_events_use_original_timing_without_repeating_defaults(self):
        plan = clip.TreatmentPlan(
            source=Path("raw_video.mp4"),
            output=Path("out.mp4"),
            caption_events=[
                clip.CaptionEvent(1.0, 2.6, "原片第一句"),
                clip.CaptionEvent(5.0, 7.2, "原片第二句"),
            ],
            caption_total_seconds=30.0,
        )

        graph = clip.build_filter_graph(plan)

        self.assertIn("原片第一句", graph)
        self.assertIn("原片第二句", graph)
        self.assertIn("between(t,1.00,2.60)", graph)
        self.assertIn("between(t,5.00,7.20)", graph)
        self.assertNotIn("原来真相一直藏在这里", graph)

    def test_caption_ass_file_is_used_instead_of_many_drawtext_filters(self):
        with TemporaryDirectory() as tmp:
            ass_path = Path(tmp) / "captions.ass"
            clip.write_caption_ass(
                ass_path,
                [clip.CaptionEvent(1.0, 2.6, "原片第一句"), clip.CaptionEvent(5.0, 7.2, "原片第二句")],
                width=720,
                height=1280,
                font_name="Arial Unicode MS",
            )
            plan = clip.TreatmentPlan(
                source=Path("raw_video.mp4"),
                output=Path("out.mp4"),
                caption_ass_path=ass_path,
            )

            graph = clip.build_filter_graph(plan)
            ass_text = ass_path.read_text(encoding="utf-8")

        self.assertIn("subtitles=filename=", graph)
        self.assertNotIn("原片第一句", graph)
        self.assertIn("Dialogue: 0,0:00:01.00,0:00:02.60", ass_text)
        self.assertIn("原片第一句", ass_text)

    def test_ocr_samples_are_merged_into_caption_events(self):
        events = clip.merge_ocr_samples(
            [
                (1.0, "  你终于回来了 "),
                (1.8, "你终于回来了"),
                (3.4, ""),
                (4.2, "她没有说话"),
                (5.0, "她没有说话"),
            ],
            sample_interval=0.8,
        )

        self.assertEqual(
            events,
            [
                clip.CaptionEvent(1.0, 2.6, "你终于回来了"),
                clip.CaptionEvent(4.2, 5.8, "她没有说话"),
            ],
        )

    def test_extract_text_from_rapidocr_result(self):
        result = [
            [[[1, 2], [3, 2], [3, 4], [1, 4]], "你怎么少了颗牙", "0.87"],
            [[[1, 6], [3, 6], [3, 8], [1, 8]], "低置信噪声", "0.20"],
        ]

        self.assertEqual(clip.text_from_rapidocr_result(result, min_confidence=0.5), "你怎么少了颗牙")

    def test_parse_srt_subtitle_file(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.srt"
            path.write_text(
                "1\n00:00:01,200 --> 00:00:03,400\n你终于回来了\n\n"
                "2\n00:00:04,000 --> 00:00:05,500\n她没有说话\n",
                encoding="utf-8",
            )

            events = clip.parse_subtitle_file(path)

        self.assertEqual(
            events,
            [
                clip.CaptionEvent(1.2, 3.4, "你终于回来了"),
                clip.CaptionEvent(4.0, 5.5, "她没有说话"),
            ],
        )

    def test_parse_ass_subtitle_file(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.ass"
            path.write_text(
                "[Events]\n"
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
                "Dialogue: 0,0:00:01.20,0:00:03.40,Default,,0,0,0,,{\\bord2}你终于\\N回来了\n",
                encoding="utf-8",
            )

            events = clip.parse_subtitle_file(path)

        self.assertEqual(events, [clip.CaptionEvent(1.2, 3.4, "你终于回来了")])

    def test_auto_source_requires_extraction_instead_of_default_captions(self):
        self.assertTrue(clip.should_attempt_asr("auto", []))
        self.assertTrue(clip.should_attempt_asr("asr", []))
        self.assertFalse(clip.should_attempt_asr("auto", [clip.CaptionEvent(1.0, 2.0, "已有字幕")]))
        self.assertFalse(clip.should_attempt_ocr("auto", False, []))
        self.assertTrue(clip.should_attempt_ocr("ocr", False, []))
        self.assertFalse(clip.should_attempt_ocr("auto", False, [clip.CaptionEvent(1.0, 2.0, "已有字幕")]))
        self.assertFalse(clip.should_attempt_ocr("none", False, []))

    def test_forbidden_evasion_modes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "不支持"):
            clip.validate_requested_modes(["watermark-destruction"])

        with self.assertRaisesRegex(ValueError, "不支持"):
            clip.validate_requested_modes(["audio-fingerprint-evasion"])

        with self.assertRaisesRegex(ValueError, "不支持"):
            clip.validate_requested_modes(["hash-evasion-chain"])


if __name__ == "__main__":
    unittest.main()
