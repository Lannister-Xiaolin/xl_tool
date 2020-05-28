#!usr/bin/env python3
# -*- coding: UTF-8 -*-
import os

from xl_tool.data.image.batch.augmentation import batch_object_replace
import glob
from xl_tool.xl_io import file_scanning
datas = glob.glob(r"F:\Dataset\IMAGE\华为—生活垃圾\dataset\trainval\VOC2007\val\*.jpg")
labeled_datas = zip(datas,map(lambda x:x.replace("jpg","xml"), datas))
ob = file_scanning(r"F:\Dataset\IMAGE\华为—生活垃圾\dataset\trainval\VOC2007\val_ob","jpg",sub_scan=True)
ob_classes = list(map(lambda x:os.path.dirname(x),ob))
batch_object_replace(labeled_data, ob,ob_classes,
                     r"F:\Dataset\IMAGE\华为—生活垃圾\dataset\trainval\VOC2007\新建文件夹",aspect_jump=0.0)
# os.path.pathsep