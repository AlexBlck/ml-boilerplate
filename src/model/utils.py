from PIL import Image, ImageDraw


def text_on_img(img, text, font, color, position):
    img = img.copy()
    draw = ImageDraw.Draw(img)
    draw.text(position, text, font=font, fill=color)
    return img


def resize_img(img, percent):
    width, height = img.size
    new_width = int(width * percent)
    new_height = int(height * percent)
    return img.resize((new_width, new_height))


def concat_h(im1, im2, mode=Image.BICUBIC):
    r = im1.height / im2.height
    im2 = im2.resize((int(r * im2.width), im1.height), mode)
    dst = Image.new("RGB", (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def concat_v(im1, im2, mode=Image.BICUBIC):
    r = im1.width / im2.width
    im2 = im2.resize((im1.width, int(r * im2.height)), mode)
    dst = Image.new("RGB", (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst
