#!usr/bin/env python3
# -*- coding: UTF-8 -*-
import os

from xl_tool.data.image.batch.augmentation import batch_object_replace,batch_mul_object_blend
import glob
from xl_tool.xl_io import file_scanning
import logging
logging.basicConfig(level=logging.DEBUG)
def test_replace():
    datas = glob.glob(r"F:\Dataset\IMAGE\华为—生活垃圾\dataset\trainval\VOC2007\val\*.jpg")
    labeled_datas = zip(datas,map(lambda x:x.replace("jpg","xml"), datas))
    ob = file_scanning(r"F:\Dataset\IMAGE\华为—生活垃圾\dataset\trainval\VOC2007\val_ob","jpg",sub_scan=True)
    ob_classes = list(map(lambda x:os.path.dirname(x),ob))
    batch_object_replace(labeled_datas, ob,ob_classes,
                         r"F:\Dataset\IMAGE\华为—生活垃圾\dataset\trainval\VOC2007\新建文件夹",aspect_jump=0.0)
def test_mul_object():
    object_path = r"F:\Dataset\IMAGE\华为—生活垃圾\dataset\trainval\VOC2007\val_ob"
    background_config_path =r"E:\Programming\Python\8_Ganlanz\food_recognition\dataset\自建数据集\4_背景底图"
    batch_mul_object_blend(object_path,background_config_path,
                           image_save_path=r"F:\Dataset\IMAGE\华为—生活垃圾\dataset\trainval\VOC2007\blend")
test_mul_object()