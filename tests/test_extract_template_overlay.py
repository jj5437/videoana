import unittest

import numpy as np

import extract_template_overlay as eto


class TemplateOverlayTests(unittest.TestCase):
    def test_cover_resize_preserves_target_size(self):
        image = np.zeros((200, 100, 3), dtype=np.uint8)

        resized = eto.resize_cover(image, 72, 128)

        self.assertEqual(resized.shape, (128, 72, 3))

    def test_punch_transparent_window_keeps_background_and_hole(self):
        image = np.zeros((128, 72, 3), dtype=np.uint8)
        image[:, :] = (10, 20, 30)

        overlay = eto.punch_transparent_window(image, (10, 20, 30, 40), border_width=0)

        self.assertEqual(overlay.shape, (128, 72, 4))
        self.assertEqual(int(overlay[0, 0, 3]), 255)
        self.assertEqual(int(overlay[30, 20, 3]), 0)

    def test_punch_can_draw_inner_border(self):
        image = np.zeros((128, 72, 3), dtype=np.uint8)

        overlay = eto.punch_transparent_window(
            image,
            (10, 20, 30, 40),
            border_width=2,
            border_color=(1, 2, 3, 255),
        )

        self.assertEqual(tuple(int(x) for x in overlay[20, 10]), (1, 2, 3, 255))
        self.assertEqual(int(overlay[30, 20, 3]), 0)

    def test_draw_title_keeps_window_transparent(self):
        image = np.zeros((128, 72, 4), dtype=np.uint8)
        image[:, :] = (10, 20, 30, 255)
        image[20:100, 8:64, 3] = 0

        titled = eto.draw_title(
            image,
            "标题",
            y=4,
            font_size=18,
            color=(255, 255, 255, 255),
            outline_color=(0, 0, 0, 255),
            outline_width=1,
            band_height=20,
            band_color=(0, 0, 0, 120),
        )

        self.assertEqual(titled.shape, (128, 72, 4))
        self.assertGreater(int(titled[8, 36, 3]), 0)
        self.assertEqual(int(titled[60, 36, 3]), 0)


if __name__ == "__main__":
    unittest.main()
