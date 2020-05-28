"""
常用批量图片增强函数
"""
from ..blending import ObjectReplaceBlend, DirectBlending, PyramidBlending, PoissonBlending
from ..annonation import Text2XML
import logging
import os
from tqdm import tqdm
from PIL import Image
from xl_tool.xl_io import file_scanning
from xl_tool.data.image.annonation import get_bndbox
from xl_tool.xl_io import save_to_json, read_json
from random import choice
import numpy as np


def update_bg_config_file(background_config_path):
    """
    背景配置文件，voc标注（标注用于合成的区域）格式文件和图片放置同一文件夹即可
    """
    files = file_scanning(background_config_path, "xml", sub_scan=False)
    configs = []
    for file in files:
        temp = {}
        temp["file"] = file.replace("xml", "jpg")
        box = get_bndbox(file)[0]
        temp["region"] = [int(i) for i in box["coordinates"]]
        configs.append(temp)
    save_to_json(configs, f"{background_config_path}/background_config.json", indent=4)


def batch_object_replace(labeled_data, object_files, object_classes, image_save_path, xml_save_path=None,
                         aspect_jump=0.5, aspects=None, replace_classes=None):
    """
    批量替换数据增强
    Args:
        labeled_datas: [(image_file,xml_file), ...]
        object_files: 目标框文件列表
        object_classes： 目标类别列表，应该与目标框文件列表长度一致
        image_save_path： 增强图片保存路径
        xml_save_path： xml保存路径
        aspect_jump: 是否对长宽比进行扰动，最终用于匹配的长宽比为以下范围采样：
                原始长宽比-aspect_jump ， 原始长宽比+aspect_jump
        aspects: 目标框长宽比，None,则会自动读取图片生成
        replace_classes: 替换的类别列表，None表示替换所有类别
    """
    blender = ObjectReplaceBlend()
    object_images = [Image.open(i) for i in object_files]
    aspects = [i.size[0] / i.size[1] for i in object_images] if not aspects else aspects
    xml_save_path = xml_save_path if xml_save_path else image_save_path
    pbar = tqdm(list(labeled_data))

    assert len(aspects) == len(object_images), "目标长宽比数量与图片数量不一致"
    assert len(object_classes) == len(object_images), "目标类别数量与图片数量不一致"

    object_images, aspects = zip(*sorted(zip(object_images, aspects), key=lambda x: x[1]))
    for image_file, xml_file in pbar:
        try:
            # assert os.path.basename(xml_file).split(".")[0] == os.path.basename(image_file).split(".")[0], "图片与标注文件无法对应"
            xml_folder = r"Dataset"
            xml_source = r'Dataset'
            aug_image, boxes, aug_object_indexes = blender.blending_one_image(image_file, object_images, aspects,
                                                                              xml_file,
                                                                              random_choice=False,
                                                                              aspect_jump=aspect_jump,
                                                                              replace_classes=replace_classes)
            save_img = f"{image_save_path}/{'replace_aug_' + os.path.basename(image_file)}"
            aug_image.save(save_img)
            text2xml = Text2XML()
            filename = os.path.basename(xml_file)
            save_xml = f"{xml_save_path}/{'replace_aug_' + os.path.basename(xml_file)}"
            objects_info = [[object_classes[index]] + coordinate for coordinate, index in
                            zip(boxes, aug_object_indexes)]
            xml = text2xml.get_xml(xml_folder, filename, filename, xml_source, aug_image.size, objects_info)
            with open(save_xml, "w") as f:
                f.write(xml)
        except Exception as e:
            logging.warning("数据替换异常！！！！\n" + str(e) + "\n" + str(xml_file))
        pbar.set_description("替换增强进度：")


def batch_mul_object_blend(object_path, background_config_path, image_save_path, xml_save_path=None,
                           blending_method="direct", categories=None, target_number=1000, num_per_images=(1, 2, 3, 4),
                           num_distribute=(0.1, 0.35, 0.4, 0.15), space_distribute=(0.5, 0.5),
                           folder="Dataset",
                           source='Dataset', ):
    """
    Args:
        object_path： 目标框位置
        background_config_path: 背景配置文件地址
        image_save_path: 图片保存位置
        xml_save_path: xml 文件保存位置
        categories: 用于合成的类别
        space_distribute: 左右排列与上下排列的比例

    """
    # todo 目前多目标只支持直接融合，不支持泊松融合和金字塔融合
    assert sum(num_distribute) == 1.0, "概率分布不为1"
    assert sum(space_distribute) == 1.0, "概率分布不为1"
    blender = DirectBlending()
    update_bg_config_file(background_config_path)
    background_config_file = f"{background_config_path}/background_config.json"
    xml_save_path = xml_save_path if xml_save_path else image_save_path
    os.makedirs(xml_save_path, exist_ok=True)
    os.makedirs(image_save_path, exist_ok=True)
    categories = os.listdir(object_path) if not categories else categories
    assert len(categories) >= 1, f"目标文件夹下未找到类别:{object_path}"

    # print(background_config_file)
    background_configs = read_json(background_config_file)
    logging.info(f"有效背景配置文件数量：{len(background_configs)}")
    cats_files = {cat: file_scanning(f"{object_path}/{cat}", file_format="jpg|JPG|jpeg|JPEG") for cat in categories}
    cats_files = {k: v for k, v in cats_files.items() if len(v) > 0}
    categories = list(cats_files.keys())
    logging.info("有效融合类别与数量分布：\n" + str({k: len(v) for k, v in cats_files.items()}))
    count = 0
    cats_count = {cat: 0 for cat in categories}
    num_state = 0
    number = num_per_images[num_state]
    sp_dis = 1 if number == 1 else np.random.choice([1, 2], space_distribute)
    choose_files = []
    while count < target_number:
        # 目前只支持并排排列，完全随机暂未实现（涉及比例和空间规划组合）
        try:
            ratio = count / target_number
            if ratio <= sum(num_distribute[:num_state + 1]):
                pass
            else:
                num_state += 1
                number = num_per_images[num_state]
                sp_dis = 1 if number == 1 else np.random.choice([1, 2], p=space_distribute)
            # 随机选择用于合成的类别，均布

            choose_cats = [choice(categories) for _ in range(number)]
            choose_files = [choice(cats_files[cat]) for cat in choose_cats]
            background_config = choice(background_configs)
            background_file, region = background_config["file"], background_config['region']
            image_file_name = f"aug_{number}_{count}.jpg"
            save_file = f"{image_save_path}/{image_file_name}"

            background_img, image_sizes, positions = blender.blending_images(background_file, choose_files,
                                                                             sp_dis,
                                                                             save_file,
                                                                             region)

            objects_info = [[cat, ] + list(positions[i]) for i, cat in enumerate(choose_cats)]
            text2xml = Text2XML()
            xml = text2xml.get_xml(folder, image_file_name, image_file_name, source, background_img.size, objects_info)
            xml_save_file = f"{xml_save_path}/{image_file_name.replace('jpg', 'xml')}"
            with open(xml_save_file, "w") as f:
                f.write(xml)
            for c in choose_cats:
                cats_count[c] += 1
            count += 1
            logging.info("\r合成数量：" + str(count))
        except Exception as e:
            logging.warning("合成异常！！！：\n" + str(e) + str(choose_files))
