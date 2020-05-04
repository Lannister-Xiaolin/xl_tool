#!usr/bin/env python3
# -*- coding: UTF-8 -*-
import os
import shutil

from PIL import Image
from cv2 import namedWindow, imshow, waitKey, WINDOW_FREERATIO, destroyAllWindows
from tqdm import tqdm
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np
from random import uniform
import imgaug.augmenters as iaa
from xl_tool.xl_io import time_analysis_wrapper


def read_with_rgb(image_file):
    """将非rgb格式图片转成rgb"""
    img = Image.open(image_file)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def grey_world(img_array):
    """白平衡处理函数
    Args:
        img_array:输入三通道图像数组
    """
    R = img_array[:, :, 0].mean()
    G = img_array[:, :, 1].mean()
    B = img_array[:, :, 2].mean()
    avg = (B + G + R) / 3
    img_array[:, :, 0] = np.minimum(img_array[:, :, 0] * (avg / R), 255)
    img_array[:, :, 1] = np.minimum(img_array[:, :, 1] * (avg / G), 255)
    img_array[:, :, 2] = np.minimum(img_array[:, :, 2] * (avg / B), 255)
    return img_array.astype(np.uint8)


def linear_contrast(img_array, offset=0.3):
    """对比度：线性变换"""
    value = offset * uniform(-1, 1) + 1
    try:
        seq = iaa.Sequential([
            iaa.LinearContrast(value)
        ])
    except AttributeError:
        np.random.bit_generator = np.random._bit_generator
        seq = iaa.Sequential([
            iaa.LinearContrast(value)
        ])
    return seq(images=[img_array])[0]


def affine_with_rotate_scale(img_array, x_scale=(0.9, 1.1), y_scale=(0.9, 1.1), rotate=(-5, 5)):
    """简单仿射变换，即旋转和缩放"""
    try:
        seq = iaa.Sequential([iaa.Affine(
            scale={"x": x_scale, "y": y_scale},  # scale images to 80-120% of their size, individually per axis
            rotate=rotate,  # rotate by -45 to +45 degrees
            order=[0, 1]
        )])
    except AttributeError:
        np.random.bit_generator = np.random._bit_generator
        seq = iaa.Sequential([iaa.Affine(
            scale={"x": x_scale, "y": y_scale},  # scale images to 80-120% of their size, individually per axis
            rotate=rotate,  # rotate by -45 to +45 degrees
            order=[0, 1]
        )])

    return seq(images=[img_array])[0]


def resize_with_pad(image, target_size):
    """
    resize  with padding
    Args:
        image: 输入图片可以是PIL.Image也可以是numpy.ndarray
        target_size: 目标尺寸

    Returns:
        resized numpy or image( same type as input), scaled size with no padding
    """
    input_type = type(image)
    if type(image) == np.ndarray:
        image = Image.fromarray(image)
    if image.size == target_size:
        return (np.array(image), image.size) if input_type == np.ndarray else (image, image.size)
    else:
        scale = min(target_size[0] / image.size[0], target_size[1] / image.size[1])
        background = Image.new("RGB", target_size)
        image = image.resize([int(scale * i) for i in image.size])
        courdinate = (target_size[0] - image.size[0]) // 2, (target_size[1] - image.size[1]) // 2
        background.paste(image, courdinate)
        return (np.array(background), image.size) if input_type == np.ndarray else (background, image.size)


def cv_show_image(image_file, window_name="图片"):
    namedWindow(window_name, WINDOW_FREERATIO)  # 0表示压缩图片，图片过大无法显示
    imshow(window_name, image_file)
    k = waitKey(0)  # 无限期等待输入，需要有这个否则会死机
    if k == 27:  # 如果输入ESC退出
        destroyAllWindows()


#
# def blending_one_image(blending_method="direct", background_img_file=None, blending_img_file=None,
#                        background_img_array=None, blending_img_array=None,
#                        x=None, y=None, x_proportion=0.6, y_proportion=0.6,
#                        x_shift=(0.1, 1.9), y_shift=(0.5, 1.9), save_img=""):
#     blender = DirectBlending() if blending_method == "direct" else PyramidBlending()
#     blending_result, [x, y, x1, y1] = blender.blending_one_image(background_img_file, blending_img_file,
#                                                                  background_img_array, blending_img_array,
#                                                                  x, y, x_proportion, y_proportion, x_shift, y_shift,
#                                                                  save_img)
#     return blending_result, [x, y, x1, y1]
#
#
# def blending_images(background_img_file, blending_img_files, sp_dis, save_img="", blending_region=None,
#                     blending_method="direct"):
#     blender = DirectBlending() if blending_method == "direct" else PyramidBlending()
#     background_img, image_sizes, positions = blender.blending_images(background_img_file, blending_img_files, sp_dis,
#                                                                      save_img,
#                                                                      blending_region)
#     return background_img, image_sizes, positions


class ImageSimilarity:
    """
        Image similarity algorithm
            ssim:
            histogram similarity
            hash: hash similaritied,
            Todo : hash is note correct
    """

    def __init__(self):
        pass

    def hist_similar(self, image1, image2):
        """直方图计算"""
        lh, rh = image1.histogram(), image2.histogram()
        assert len(lh) == len(rh)
        return sum(1 - (0 if l == r else float(abs(l - r)) / max(l, r)) for l, r in zip(lh, rh)) / len(lh)

    def ssim(self, image1, image2):
        """ssim相似度计算"""
        return compare_ssim(np.array(image1), np.array(image2), multichannel=True)

    def dhash_sim(self, image1, image2):
        """计算差异哈希值算法的相似度"""
        # Todo:差异哈希未经仔细验证，有问题
        image1 = image1.resize((8, 8), Image.ANTIALIAS).convert('L')
        image2 = image2.resize((8, 8), Image.ANTIALIAS).convert('L')
        return self.hamming_distance(self.dHash(np.array(image1)), self.dHash(np.array(image2)))

    # 差异哈希算法
    def dHash(self, array):
        assert array.shape == (8, 8)
        hash = []
        # 每行前一个像素大于后一个像素为1，相反为0，生成哈希
        for i in range(8):
            for j in range(7):
                if array[i, j] > array[i, j + 1]:
                    hash.append(1)
                else:
                    hash.append(0)
        return hash

    # 均值哈希算法
    def aHash(self, array):
        assert array.shape == (8, 8)
        avreage = np.mean(array)
        hash = []
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                if array[i, j] > avreage:
                    hash.append(1)
                else:
                    hash.append(0)
        return hash

    # 计算汉明距离
    def hamming_distance(self, hash1, hash2):
        num = 0
        for index in range(len(hash1)):
            if hash1[index] != hash2[index]:
                num += 1
        return num / len(hash1)

    @time_analysis_wrapper("相似度计算")
    def caculate_similarities(self, files, save=None, hash=False, hstm=False, resize=(48, 48)):
        images = [(os.path.basename(file), Image.open(file).resize(resize), file) for file in files]
        results = set()
        for i in tqdm(list(range(len(images) - 1))):
            for j in (list(range(i + 1, len(images)))):
                img1 = images[i][1]
                img2 = images[j][1]
                hstm = self.hist_similar(img1, img2) if hstm else 0
                ssim = self.ssim(img1, img2)
                dhash = self.dhash_sim(img1, img2) if hash else 0
                if ssim > 0.6 or hstm > 0.7:
                    print(f"\nsslim: {ssim}\thstm: {hstm}\tdhash:{dhash}\n-----",
                          images[i][0], "\t", images[j][0])
                    results.add(images[i][2])
                    results.add(images[j][2])
        if save:
            os.makedirs(save, exist_ok=True)
            for file in results:
                shutil.copy(file, save + "/" + os.path.basename(file))


def main():
    # embedding_img = Image.open(
    #     r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\3_公开数据集抽取\原始标注文件\hamburger\n07697313_12.JPEG")
    # background_img = Image.open(r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\4_背景底图\1.jpg")
    # aug = blending_one_image(background_img, embedding_img)
    # aug.show()
    image = Image.open(r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\4_背景底图\1.jpg")
    resize_with_pad(image, (256, 256)).show()
    # print(embedding_rescale((1000, 1000), (900, 900), x=None, y=None, x_proportion=0.7, y_proportion=0.7))


if __name__ == '__main__':
    main()
