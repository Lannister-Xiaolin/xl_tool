#!usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
目标抽取python object_extract.py --xml=F:\Dataset\IMAGE\华为—生活垃圾\dataset\trainval\VOC2007\val --save=F:\Dataset\IMAGE\华为—生活垃圾\dataset\trainval\VOC2007\val_ob

"""

import os
from tqdm import tqdm
from xl_tool.data.image.annonation import xml_object_extract
from xl_tool.xl_io import file_scanning
from absl import flags, app, logging

flags.DEFINE_string("xml", "", "xml path")
flags.DEFINE_string("img", "", "image path, if image and xml in the same dir set it to:F:\Dataset\IMAGE\华为—生活垃圾\dataset\trainval\VOC2007 '' ")
flags.DEFINE_string("save", "", "path to save object")
Flags = flags.FLAGS


def main(_):
    assert Flags.save != "", "请输入保存路径"
    assert Flags.xml != "", "请输入xml文件路径"
    xml_files = file_scanning(Flags.xml, file_format='xml', sub_scan=True)
    image_files = [os.path.join(Flags.img, os.path.basename(file).replace("xml", "jpg")) for file in
                   xml_files] if Flags.img else [
        file.replace("xml", "jpg") for file in xml_files]
    valid_xml_files = []
    valid_image_files = []
    for i in range(len(xml_files)):
        if os.path.exists(image_files[i]):
            valid_xml_files.append(xml_files[i])
            valid_image_files.append(image_files[i])
    logging.info(f"扫描到有效文件： {len(valid_image_files)}")
    os.makedirs(Flags.save, exist_ok=True)
    pbar = tqdm(list(range(len(valid_image_files))))
    for i in pbar:
        xml_object_extract(valid_xml_files[i], valid_image_files[i], Flags.save,
                           min_size_sum=40, w_h_limits=(10, 0.1))
        pbar.set_description("抽取进度： ")

if __name__ == '__main__':
    app.run(main)
