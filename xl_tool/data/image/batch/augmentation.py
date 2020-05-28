"""
常用批量图片增强函数
"""
from ..blending import ObjectReplaceBlend
from ..annonation import Text2XML
import logging
import os
from tqdm import tqdm
from PIL import Image

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
    os.makedirs(xml_save_path, exist_ok=True)
    os.makedirs(image_save_path, exist_ok=True)
    object_images, aspects = zip(*sorted(zip(object_images, aspects), key=lambda x: x[1]))
    for  image_file,xml_file in pbar:
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
            logging.warning("数据替换异常！！！！\n" + str(e)+"\n"+str(xml_file))
        pbar.set_description("替换增强进度：")
