#!usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
根据目标框分布划分目标检测数据集
python detection_validation_split.py --xml_path= \
F:\Dataset\IMAGE\华为—生活垃圾\dataset\trainval\VOC2007\Annotations \
--train=F:\Dataset\IMAGE\华为—生活垃圾\dataset\trainval\VOC2007\trainm\
 --val=F:\Dataset\IMAGE\华为—生活垃圾\dataset\trainval\VOC2007\valm\
  --val_split=0.1 \
  --save_result=F:\Dataset\IMAGE\华为—生活垃圾\dataset\trainval\VOC2007\labels_distribute.json
"""

import os

from xl_tool.data.image.annonation import get_bndbox
from xl_tool.xl_io import file_scanning, save_to_json
from collections import Counter
from random import shuffle
import shutil
from tqdm import tqdm
from absl import app
from absl import flags
from absl import logging

flags.DEFINE_string('xml_path', "", 'Path to xml_path')
flags.DEFINE_string('train', "", 'train path to save')
flags.DEFINE_string('val', "", 'val path to save')
flags.DEFINE_string('save_result', "", 'result to save label distribute')
flags.DEFINE_string('image_path', "", 'image_path to copy if xml and jpg in the same directory set it "same"')
flags.DEFINE_float('val_split', 0.2, 'val split')
FLAGS = flags.FLAGS


def validation_split(xml_path, train, val, image_path=None, val_split=0.2, save_result=""):
    xml_files = file_scanning(xml_path, "xml", sub_scan=True)
    shuffle(xml_files)
    xml_labels = []
    labels_all = []
    os.makedirs(train, exist_ok=True)
    os.makedirs(val, exist_ok=True)
    for file in tqdm(xml_files):
        labels = [i["name"] for i in get_bndbox(file)]
        labels_all.extend(labels)
        xml_labels.append(labels)
    labels_count = dict(Counter(labels_all))
    print(labels_count)
    labels_val_spilt_index = {k: int(v * val_split) for k, v in labels_count.items()}
    labels_count_val = {k: 0 for k, v in labels_count.items()}
    for i in tqdm(list(range(len(xml_files)))):
        basename = os.path.basename(xml_files[i])
        image_basename = os.path.basename(xml_files[i]).replace("xml", "jpg")
        if all([(labels_count_val[name] + 1) > labels_val_spilt_index[name] for name in xml_labels[i]]) and xml_labels[
            i]:
            shutil.copy(xml_files[i], os.path.join(train, basename))
            if image_path:
                image_file = os.path.join(image_path, image_basename) if image_path != "same" else xml_files[i].replace(
                    "xml", "jpg")
                shutil.copy(image_file, os.path.join(train, image_basename))
        else:
            shutil.copy(xml_files[i], os.path.join(val, basename))
            if image_path:
                image_file = os.path.join(image_path, image_basename) if image_path != "same" else xml_files[i].replace(
                    "xml", "jpg")
                shutil.copy(image_file, os.path.join(val, image_basename))
            for name in xml_labels[i]:
                labels_count_val[name] += 1
    print("validation distribute: ", labels_count_val)
    if save_result:
        save_to_json({"all_labels":
                          dict(labels_count_val),
                      "val_labels": labels_count_val}, save_result)


def main(_):
    validation_split(FLAGS.xml_path, FLAGS.train, FLAGS.val, FLAGS.image_path, val_split=FLAGS.val_split,
                     save_result=FLAGS.save_result)


if __name__ == '__main__':
    app.run(main)

# validation_split(r"F:\Dataset\IMAGE\华为—生活垃圾\dataset\trainval\VOC2007\Annotations",
#                  r"F:\Dataset\IMAGE\华为—生活垃圾\dataset\trainval\VOC2007\train",
#                  r"F:\Dataset\IMAGE\华为—生活垃圾\dataset\trainval\VOC2007\val", image_path=None, val_split=0.2)
