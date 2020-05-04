#!usr/bin/env python3
# -*- coding: UTF-8 -*-
from copy import deepcopy

from PIL import Image, ImageDraw
from PIL.Image import BILINEAR
import random
import cv2
import numpy as np
from abc import abstractmethod
from xl_tool.data.image.general import resize_with_pad
from xl_tool.xl_general import binary_search_nearest_abs, file_scanning
from .annonation import get_bndbox


# Todo 小目标组合多个排列合成
# Todo 单目标比例范围
# Todo 指定位置合成
# Todo 分割合成边缘模糊之：泊松/金字塔等


class BlendingBase:
    def __init__(self):
        pass

    @abstractmethod
    def blending_one_image(self):
        pass

    @classmethod
    def max_blending_size(cls, background_size, x, y, x_proportion, y_proportion):
        """计算最大允许嵌入图片大小"""
        if x:
            max_x = min(int(background_size[0] * x_proportion), background_size[0] - x)
        else:
            max_x = int(background_size[0] * x_proportion)
        if y:
            max_y = min(int(background_size[1] * y_proportion), background_size[1] - y)
        else:
            max_y = int(background_size[1] * y_proportion)
        return max_x, max_y

    @classmethod
    def calculate_rescale(cls, background_size, blending_size, x, y, x_proportion, y_proportion):
        """计算融合嵌入图片缩放系数
        Args:
            background_size:背景图片尺寸，宽 X 高
            blending_size:背景图片尺寸，宽 X 高
            x:嵌入位置,宽度方向，可为空
            y:嵌入位置，高度方向，可为空
            x_proportion:宽度方向最大占比
            y_proportion:高度方向最大占比
        """
        max_x, max_y = cls.max_blending_size(background_size, x, y, x_proportion, y_proportion)
        x_scale = 1 if max_x > blending_size[0] else max_x / blending_size[0]
        y_scale = 1 if max_y > blending_size[1] else max_y / blending_size[1]
        scale = min(x_scale, y_scale)
        if scale < 0.99:
            pass
            # print("嵌入图像{}过大，需进行缩放，缩放比例：{:.2f}".format(blending_size, scale))
        return scale

    @classmethod
    def save_image(cls, img: Image.Image, save):
        try:
            img.save(save)
        except Exception as e:
            print("增强图片存储失败！:".format(save))
            pass

    @classmethod
    def valid_size_confirm(cls, image_sizes, region_size, sp_dis, min_rescale):
        """
        该函数暂时无用
        Args:
            image_sizes:尺寸列表
            region_size: 嵌入区域大小
            sp_dis ： 组合方式
            min_rescale: 最大缩放区域
        """
        if sp_dis in (1, 5):
            return (sum([size[0] for size in image_sizes]) * min_rescale) < region_size[0]
        elif sp_dis(2, 6):
            return (sum([size[1] for size in image_sizes]) * min_rescale) < region_size[1]
        elif sp_dis in (3, 4):
            for i in range(2, -1, -1):
                one_combine = image_sizes[i]
                valid_x = (one_combine[0] + max([image_sizes[j][0] for j in range(3) if j != i])) < region_size[0]
                valid_y = (one_combine[1] < region_size[1] and sum([image_sizes[j][1] for j in range(3) if
                                                                    j != i])) < region_size[1]
                if valid_x and valid_y:
                    return True
            return False
        else:
            return False
        pass

    @classmethod
    def paralleling_caculate(cls, image_sizes, region_size, sp_dis):
        # print("输入尺寸：", image_sizes, "\t区域大小:", region_size)
        first, second = (0, 1) if sp_dis in (1, 5) else (1, 0)
        first_sum = sum([size[first] for size in image_sizes])
        if first_sum < region_size[first]:
            rescales = [1 if size[second] < region_size[second] else region_size[second] / size[second] for size in
                        image_sizes]
        else:
            # print("图片过大，进行缩放！")
            rescale = region_size[first] * 0.95 / first_sum
            rescales = [rescale if size[second] * rescale < region_size[second] else region_size[second] / size[second]
                        for size in
                        image_sizes]
        image_sizes = [[int(c * rescale) for c in size] for (rescale, size) in zip(rescales, image_sizes)]
        margin_first_max = region_size[first] - sum([size[first] for size in image_sizes])
        margin_seconds = [region_size[second] - size[second] for size in image_sizes]
        positions = []
        margin_first_remain = margin_first_max
        first_start = 0
        for size, margin_second in zip(image_sizes, margin_seconds):
            position = [None, None, None, None]
            margin = int(margin_first_remain * random.random())
            # print(margin)
            margin_first_remain = margin_first_remain - margin
            position_first = margin + first_start
            position_second = int(margin_second * random.random())
            first_start = position_first + size[first]
            position[first], position[second], position[first + 2], position[second + 2] = \
                position_first, position_second, first_start, position_second + size[second]
            positions.append(position)
        # print("缩放比例：", rescales, "\n缩放后大小：", image_sizes)
        return image_sizes, rescales, positions

    @classmethod
    def calculate_mul_blending_positions(cls, image_sizes, region_size, sp_dis):
        """计算嵌入位置
            1，2为并列排列
            3为混合排列，根据尺寸随机切分成上下或者左右2块，然后再做并列排列（暂未完成）
        """
        if sp_dis in (1, 5, 2, 6):
            return cls.paralleling_caculate(image_sizes, region_size, sp_dis)
        else:
            pass

    def rescale_blending(self, blending, background, x, y, x_proportion,
                         y_proportion, x_shift, y_shift, blending_region=None):
        """
        Args:
            blending:
            background:
            x:
            y:
            x_proportion:
            y_proportion:
            x_shift:
            y_shift:

        Returns:
            (x,y,blending),
                x: 合成图片插入横坐标（对应宽度，PIL的size[0], opencv的shape[1]），左上角
                y: 合成图片插入坐标（对应高度，PIL的size[1], opencv的shape[0]），左上角
                blending: resize后的图片
        """
        blending_size = (blending.shape[1], blending.shape[0],)
        background_size = (background.shape[1], background.shape[0])
        if not blending_region:
            blending_region = [0, 0, *background_size]
        region_size = (blending_region[2] - blending_region[0], blending_region[3] - blending_region[1])
        assert region_size[0] > 0, '插入区域宽度小于0'
        assert region_size[1] > 0, '插入区域高度小于0'
        scale_whole = self.calculate_rescale(background_size, blending_size, x, y, x_proportion, y_proportion)
        scale_region = self.calculate_rescale(region_size, blending_size, x, y, 0.99, 0.99)
        scale = min(scale_whole, scale_region)
        new_size_cv = (int(blending_size[0] * scale), int(blending_size[1] * scale))
        blending = blending if 0.99 < scale < 1.01 else cv2.resize(blending, dsize=new_size_cv)
        if not x:
            x = blending_region[0] + int(((region_size[0] - blending.shape[1]) // 2) * random.uniform(*x_shift))
        if not y:
            y = blending_region[1] + int(((region_size[1] - blending.shape[0]) // 2) * random.uniform(*y_shift))
        return x, y, blending

        # blending_size = (blending.shape[1], blending.shape[0],)
        # background_size = (background.shape[1], background.shape[0])
        # scale = self.calculate_rescale(background_size, blending_size, x, y, x_proportion, y_proportion)
        # new_size_cv = (int(blending_size[0] * scale), int(blending_size[1] * scale))
        # blending = blending if 0.99 < scale < 1.01 else cv2.resize(blending, dsize=new_size_cv)
        # if not x:
        #     x = int(((background_size[0] - blending.shape[1]) // 2) * random.uniform(*x_shift))
        # if not y:
        #     y = int(((background_size[1] - blending.shape[0]) // 2) * random.uniform(*y_shift))
        # return x, y, blending


class DirectBlending(BlendingBase):
    """
    直接对图片进行融合嵌入，主要包括单目标融合和多目标融合
    """

    def __init__(self):
        super(DirectBlending).__init__()

    def blending_one_image(self, background_img_file=None, blending_img_file=None,
                           background_img_array=None, blending_img_array=None, x=None, y=None,
                           x_proportion=0.7, y_proportion=0.7, x_shift=(0.5, 1.5), blending_region=None,
                           y_shift=(1, 1.9), save_img="", resample=BILINEAR,test=False):
        """单张图片融合嵌入
        下一版完善功能：
            最小比例占比限制
            长宽比扰动
        Args:
            background_img_file:背景图片
            blending_img_file:嵌入图片
            background_img_array:背景图片数组, background_img_file为空时传递该值可用
            blending_img_array:融合图片数组，blending_img_file为空时传递该值可用
            x: 嵌入位置x坐标（左上角），如为空，根据中心位置计算
            y:同上
            x_shift:针对x中心位置的摆动,取值范围为0-2
            y_shift:针对y中心位置的摆动，取值范围为0-2
            x_proportion: 嵌入图片x长度与背景图片x长度的最大占比
            y_proportion: 嵌入图片y长度与背景图片y长度的最大占比
            resample: 图像插值方式
            save_img:是否保持图片，是则输入路径，否则为空字符
        Returns:
            数组：Image格式的融合图片 + 融合后的bounding box坐标
                blending image,[xmin,ymin,xmax,ymax]
        """
        # background_img = Image.open(background_img_file) if background_img_array is None else \
        #     Image.fromarray(background_img_array)
        # blending_img = Image.open(blending_img_file) if blending_img_array is None else \
        #     Image.fromarray(blending_img_array)
        # if not blending_region:
        #     blending_region = [0,0,*blending_img.size]
        # bg_size = background_img.size
        # region_size = (blending_region[2]-blending_region[0],blending_region[3]-blending_region[1])
        # scale_whole = self.calculate_rescale(bg_size, blending_img.size, x, y, x_proportion, y_proportion)
        # scale_region = self.calculate_rescale(region_size, blending_img.size, x, y, x_proportion, y_proportion)
        # new_size = (int(size[0] * scale), int(size[1] * scale))
        # blending_img = blending_img if 0.99 < scale < 1.01 else blending_img.resize(new_size, resample=resample)
        # if not x:
        #     x = int(((background_img.size[0] - blending_img.size[0]) // 2) * random.uniform(*x_shift))
        # if not y:
        #     y = int(((background_img.size[1] - blending_img.size[1]) // 2) * random.uniform(*y_shift))

        background = cv2.imdecode(np.fromfile(background_img_file, dtype=np.uint8), cv2.IMREAD_UNCHANGED) \
            if background_img_array is None else background_img_array[:, :, ::-1]
        blending = cv2.imdecode(np.fromfile(blending_img_file, dtype=np.uint8), cv2.IMREAD_UNCHANGED) \
            if blending_img_array is None else blending_img_array[:, :, ::-1]
        # print(blending.shape)
        x, y, blending = self.rescale_blending(blending, background, x, y,
                                               x_proportion, y_proportion, x_shift, y_shift,blending_region)

        background_img = Image.fromarray(background[:, :, ::-1])
        blending_img = Image.fromarray(blending[:, :, ::-1])
        background_img.paste(blending_img, (x, y))
        blending_result = background_img
        if save_img:
            self.save_image(blending_result, save_img)
        if test:
            blending_result.show()
        return blending_result, [x, y, x + blending_img.size[0], y + blending_img.size[1]]

    def blending_images(self, background_img_file, blending_img_files, sp_dis, save_img="", blending_region=None):
        """
        多目标融合：
        多目标融合的位置分布主要包括以下几种方式：
            左右并列排列  1
            上下并列排列  2
            混合排列   3
        """
        background_img = Image.open(background_img_file)
        # print("背景图片大小：", background_img.size)
        blending_imgs = [Image.open(blending_img_file) for blending_img_file in blending_img_files]
        image_sizes = [img.size for img in blending_imgs]
        blending_region = blending_region if blending_region else (0, 0, background_img.size[0], background_img.size[1])
        x, y = blending_region[0], blending_region[1]
        region_size = blending_region[2] - blending_region[0], blending_region[3] - blending_region[1]
        image_sizes, rescales, positions = self.calculate_mul_blending_positions(image_sizes, region_size, sp_dis)
        positions = [(position[0] + x, position[1] + y, position[2] + x, position[3] + y) for position in positions]
        # print("嵌入坐标位置：", positions)
        blending_imgs = [blending_img.resize(size) for (blending_img, size) in zip(blending_imgs, image_sizes)]
        for blending_img, position in zip(blending_imgs, positions):
            # print(blending_img.size, position)
            background_img.paste(blending_img, (position[0], position[1]))
        if save_img:
            self.save_image(background_img, save_img)
        # background_img.show()
        return background_img, image_sizes, positions


class PyramidBlending(BlendingBase):
    """
    Args:
        num_pyramid: 层数，默认为6
        min_size； 金字塔最后一层最小尺寸，默认为6
    """

    def __init__(self, num_pyramid=6, min_size=4):
        super(PyramidBlending).__init__()
        self.num_pyramid = num_pyramid
        self.min_size = min_size
        pass

    def generate_gaussian_pyramid(self, image_array):
        """计算高斯金字塔"""
        array = image_array.copy()
        sizes = []
        if min(image_array.shape[:2]) ** (1 / (2 ** self.num_pyramid)) < (self.min_size - 1):
            base_exp = (min(image_array.shape[:2]) / self.min_size) ** (1 / self.num_pyramid)
            denominators = [base_exp ** i for i in range(1, self.num_pyramid + 1)]
            sizes = [(int(image_array.shape[0] / i), int(image_array.shape[1] / i)) for i in denominators]
        gaussian_pyramid = [array]
        for i in range(self.num_pyramid):
            array = cv2.pyrDown(array, sizes[i]) if sizes else cv2.pyrDown(array)
            gaussian_pyramid.append(array)
        return gaussian_pyramid

    def generate_laplacian_pyramid(self, gaussian_pyramid):
        """计算拉普拉斯金字塔"""
        laplacian_pyramid = [gaussian_pyramid[self.num_pyramid - 1]]
        for i in range(self.num_pyramid - 1, 0, -1):
            ge = cv2.pyrUp((gaussian_pyramid[i]),
                           dstsize=(gaussian_pyramid[i - 1].shape[1], gaussian_pyramid[i - 1].shape[0]))
            laplacian = cv2.subtract(gaussian_pyramid[i - 1], ge)
            laplacian_pyramid.append(laplacian)
        return laplacian_pyramid

    def surround_reconstruct(self, lp_background, lp_blending, normalize_x, normalize_y):
        ls = []
        for la, lb in zip(lp_background, lp_blending):
            temp = deepcopy(la)
            x, y = int(normalize_x * la.shape[1]), int(normalize_y * la.shape[0])
            temp[y:y + lb.shape[0], x:x + lb.shape[1]] = lb
            ls.append(temp)
        ls_array = ls[0]
        for i in range(1, self.num_pyramid):
            ls_array = cv2.pyrUp(ls_array, dstsize=(ls[i].shape[1], ls[i].shape[0]))
            ls_array = cv2.add(ls_array, ls[i])
        return ls_array

    def blending_one_image(self, background_img_file=None, blending_img_file=None,
                           background_img_array=None, blending_img_array=None,
                           x=None, y=None, x_proportion=0.6, y_proportion=0.6, blending_region=None,
                           x_shift=(0.5, 1.5), y_shift=(1.0, 1.9), save_img="",test=False):
        """基于opencv的金字塔融合"""
        background = cv2.imdecode(np.fromfile(background_img_file, dtype=np.uint8), cv2.IMREAD_UNCHANGED) \
            if background_img_array is None else background_img_array[:, :, ::-1]
        blending = cv2.imdecode(np.fromfile(blending_img_file, dtype=np.uint8), cv2.IMREAD_UNCHANGED) \
            if blending_img_array is None else blending_img_array[:, :, ::-1]
        # print(blending.shape)
        x, y, blending = self.rescale_blending(blending, background, x, y,
                                               x_proportion, y_proportion, x_shift, y_shift,blending_region=blending_region)

        g_background = self.generate_gaussian_pyramid(background)
        g_blending = self.generate_gaussian_pyramid(blending)
        lp_background = self.generate_laplacian_pyramid(g_background)
        lp_blending = self.generate_laplacian_pyramid(g_blending)
        blending_result = Image.fromarray(
            self.surround_reconstruct(lp_background, lp_blending,
                                      x / background.shape[1], y / background.shape[0])[:, :, ::-1])
        if save_img:
            self.save_image(blending_result, save_img)
        if test:
            blending_result.show()
        return blending_result, [x, y, x + blending.shape[1], y + blending.shape[0]]


class PoissonBlending(BlendingBase):

    def __init__(self):
        super(PoissonBlending).__init__()
        pass

    def blending_one_image(self, background_img_file=None, blending_img_file=None,
                           background_img_array=None, blending_img_array=None,
                           x=None, y=None, x_proportion=0.6, y_proportion=0.6, blending_region=None,
                           x_shift=(0.5, 1.5), y_shift=(1.0, 1.9), save_img="",test=False):
        """基于opencv的金字塔融合"""
        background = cv2.imdecode(np.fromfile(background_img_file, dtype=np.uint8),
                                  cv2.IMREAD_UNCHANGED) \
            if background_img_array is None else background_img_array[:, :, ::-1]
        blending = cv2.imdecode(np.fromfile(blending_img_file, dtype=np.uint8), cv2.IMREAD_UNCHANGED) \
            if blending_img_array is None else blending_img_array[:, :, ::-1]
        # print(blending.shape)
        x, y, blending = self.rescale_blending(blending, background, x, y,
                                               x_proportion, y_proportion, x_shift, y_shift,blending_region=blending_region)
        # print(blending.shape)
        blending_result = Image.fromarray(
            cv2.seamlessClone(blending, background, np.ones(blending.shape, dtype=np.uint8) * 255,
                              (x + blending.shape[1] // 2, y + blending.shape[0] // 2),
                              flags=cv2.NORMAL_CLONE)[:, :, ::-1])
        if save_img:
            self.save_image(blending_result, save_img)
        if test:
            blending_result.show()
        return blending_result, [x, y, x + blending.shape[1], y + blending.shape[0]]


class ObjectReplaceBlend:
    """目标框替换"""

    def __init__(self):
        pass

    def blending_one_image(self, image_file, object_images, object_aspects, xml_file, classes=None,
                           random_choice=False, aspect_jump=0):
        """ """
        boxes = get_bndbox(xml_file)
        if classes:
            boxes = [i for i in boxes if i["name"] if classes]
        image = Image.open(image_file)
        aug_boxes = []
        for box in boxes:
            center_x = (box["coordinates"][2] + box["coordinates"][0]) / 2
            center_y = (box["coordinates"][3] + box["coordinates"][1]) / 2
            width = box["coordinates"][2] - box["coordinates"][0]
            height = box["coordinates"][3] - box["coordinates"][1]
            aspect = (width) / (height)
            if aspect_jump:
                aspect = aspect + random.uniform(-aspect_jump * 0.8, aspect_jump)
            best_aspect_index = binary_search_nearest_abs(object_aspects, aspect, 0, len(object_aspects) - 1)
            if random_choice:
                paste_image = random.choice(object_images)
            else:
                paste_image = object_images[best_aspect_index]
            paste_image, size = resize_with_pad(paste_image, target_size=(
                box["coordinates"][2] - box["coordinates"][0], box["coordinates"][3] - box["coordinates"][1]))
            image.paste(paste_image, box["coordinates"])
            aug_boxes.append([int(center_x - size[0] / 2),
                              int(center_y - size[1] / 2),
                              int(center_x + size[0] / 2),
                              int(center_y + size[1] / 2)])
        return image, aug_boxes


class SegBlend(BlendingBase):
    """
    分割图片合成
    """

    def __init__(self):
        super(SegBlend).__init__()
        pass

    def blending_one_image(self, background_img_file=None, blending_img_file=None, mask_img_file=None,
                           background_img_array=None, blending_img_array=None, mask_img_array=None,
                           x=None, y=None, x_proportion=0.6, y_proportion=0.6, poisson=False, blending_region=None,
                           x_shift=(0.5, 1.5), y_shift=(1.0, 1.9), save_img="", resample=BILINEAR):
        background_img = Image.open(background_img_file) if background_img_array is None else \
            Image.fromarray(background_img_array)
        blending_img = Image.open(blending_img_file) if blending_img_array is None else \
            Image.fromarray(blending_img_array)
        mask_img = Image.open(mask_img_file) if mask_img_array is None else \
            Image.fromarray(mask_img_array)
        size = blending_img.size
        scale = self.calculate_rescale(background_img.size, blending_img.size, x, y, x_proportion, y_proportion)
        new_size = (int(size[0] * scale), int(size[1] * scale))
        blending_img = blending_img if 0.99 < scale < 1.01 else blending_img.resize(new_size, resample=resample)
        mask_img = mask_img if 0.99 < scale < 1.01 else mask_img.resize(new_size, resample=resample)
        if not x:
            x = int(((background_img.size[0] - blending_img.size[0]) // 2) * random.uniform(*x_shift))
        if not y:
            y = int(((background_img.size[1] - blending_img.size[1]) // 2) * random.uniform(*y_shift))
        if not poisson:
            background_img.paste(blending_img, (x, y), mask_img)
            blending_result = background_img
        else:
            blending = np.array(blending_img)[:, :, ::-1]
            background = np.array(background_img)[:, :, ::-1]
            mask = np.array(mask_img)[:, :, ::-1] if mask_img.mode not in ["P", "L", "1"] else np.array(mask_img)
            mask = mask.astype(np.uint8) * 255
            blending_result = Image.fromarray(
                cv2.seamlessClone(blending, background, mask,
                                  (x + blending.shape[1] // 2, y + blending.shape[0] // 2),
                                  flags=cv2.NORMAL_CLONE)[:, :, ::-1])
        if save_img:
            self.save_image(blending_result, save_img)
        blending_result.show()
        return blending_result, [x, y, x + blending_img.size[0], y + blending_img.size[1]]

    def segbox_extract(self):
        """open image segmatation area extract"""
        pass


def test_blending_pyramid():
    background_img_file = r"D:\1.jpg"
    blending_img_file = r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\public_dataset\food101\food\pizza" \
                        r"\702165.jpg"
    blender = PyramidBlending(6)
    blender.blending_one_image(background_img_file, blending_img_file)


def test_blending_direct():
    background_img_file = r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\4_背景底图" \
                        r"\4.jpg"
    blending_img_file = r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\4_背景底图" \
                        r"\55555.jpg"
    blender = DirectBlending()
    blender1 = PoissonBlending()
    blender2 = PyramidBlending()
    # for i in range(3):
    blender.blending_one_image(background_img_file, blending_img_file,test=True)
    blender1.blending_one_image(background_img_file, blending_img_file, blending_region=[100, 200, 540, 480],test=True)
    blender2.blending_one_image(background_img_file, blending_img_file, blending_region=[100, 200, 540, 480],test=True)

def test_mul_blending():
    a = DirectBlending()
    backgroud_file = r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\4_背景底图\2.jpg"
    blending_files = [
        r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\5_抽取目标\broccoli\000000000009.jpg",
        r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\5_抽取目标\broccoli\000000000009.jpg",
        r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\5_抽取目标\orange\000000000009.jpg"]
    print(a.blending_images(backgroud_file, blending_files, 2, (20, 80, 620, 440)))


def test_mask_speed():
    import numpy as np
    import time
    b = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    m = np.random.randint(0, 2, (224, 224, 3), dtype=np.uint8)
    b_img = Image.fromarray(b)
    m_img = Image.fromarray(m[:, :, 0], mode="1")
    st = time.time()
    for i in range(1000):
        b_img.paste(m_img, mask=m_img)
    print("paste时间：", time.time() - st)
    st = time.time()
    for i in range(1000):
        t = b * m
    print("numpy点乘时间：", time.time() - st)
    st = time.time()
    for i in range(1000):
        np.putmask(b, m, m)
        # t = b*m
    print("numpy掩码：", time.time() - st)


def test_seg_mask():
    blender = SegBlend()
    background_img_file = r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\4_背景底图\1.jpg"
    blending_img_file = r"F:\Large_dataset\segmentation\mask\crop_00c1d2697deeaea1_m014j1m.jpg"
    mask_img_file = r"F:\Large_dataset\segmentation\image\crop_00bb5720a7ba062e_m014j1m_b26b27f6.png"
    blender.blending_one_image(background_img_file, blending_img_file, mask_img_file, poisson=True, x_proportion=0.2,
                               save_img=r"F:\Large_dataset\segmentation\segmentation_blend\crop_00bb5720a7ba062e_m014j1m_b26b27f6.jpg")


def test_ObjectReplaceBlend():
    for i in range(10):
        blender = ObjectReplaceBlend()
        base = r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\1_真实场景\0_已标框\bacon"
        image_file = f"{base}/0a634ce3-4f09-5bd4-99a0-238331e88fad.jpg"
        xml_file = f"{base}/0a634ce3-4f09-5bd4-99a0-238331e88fad.xml"
        object_path = r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\5_抽取目标\混合\bacon"
        object_images = [Image.open(i) for i in file_scanning(object_path, "jpg")]
        aspects = [i.size[0] / i.size[1] for i in object_images]
        object_images, aspects = zip(*sorted(zip(object_images, aspects), key=lambda x: x[1]))
        im, boxes = blender.blending_one_image(image_file, object_images, aspects, xml_file, random_choice=False,
                                               aspect_jump=1.0)
        draw = ImageDraw.Draw(im)
        for box in boxes:
            draw.rectangle(box)
        im.show()


def main():
    # blender = ObjectReplaceBlend()

    # blender = PoissonBlending()

    # background_img_file = r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\4_背景底图\1.jpg"
    # blending_img_file = r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集" \
    #                     r"\5_抽取目标\混合\sweet_potato\baidu_sweet_potato_1027493905_3011920384.jpg"
    # a = blender.blending_one_image(background_img_file, blending_img_file, x=100, y=100)[0]
    # a.show()
    # print(a)
    # test_blending_pyramid()
    # test_blending_direct()
    # test_mul_blending()
    # test_seg_mask()
    test_ObjectReplaceBlend()


if __name__ == '__main__':
    main()
