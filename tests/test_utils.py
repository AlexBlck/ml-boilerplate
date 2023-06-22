import os
import sys

path = os.path.realpath(os.path.dirname(__file__))
sys.path.append(os.path.join(path, "../src"))
import unittest

from PIL import Image

from src.model.utils import concat_h, concat_v, resize_img


class TestImageUtils(unittest.TestCase):
    def test_concat_h(self):
        im1 = Image.new("RGB", (100, 100))
        im2 = Image.new("RGB", (100, 100))
        im3 = concat_h(im1, im2)
        self.assertEqual(im3.size, (200, 100))

    def test_concat_v(self):
        im1 = Image.new("RGB", (100, 100))
        im2 = Image.new("RGB", (100, 100))
        im3 = concat_v(im1, im2)
        self.assertEqual(im3.size, (100, 200))

    def test_resize_img(self):
        im1 = Image.new("RGB", (100, 100))
        im2 = resize_img(im1, 2.0)
        im3 = resize_img(im1, 0.5)
        self.assertEqual(im2.size, (200, 200))
        self.assertEqual(im3.size, (50, 50))


if __name__ == "__main__":
    unittest.main()
