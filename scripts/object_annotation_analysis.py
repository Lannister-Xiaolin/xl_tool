#!usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
分析目标检测数据集，使用示例：
python object_annotation_analysis.py --xm_path=/VOC2007/Annotations --result_path=/VOC2007/result.json

"""

from xl_tool.data.image.analysis import VocAnalysis
from absl import app
from absl import flags
from absl import logging

logging.info('Interesting Stuff')

# flags.DEFINE

flags.DEFINE_string('xml_path', '', 'Path to xml_path.')
flags.DEFINE_string('result_path', None,
                    'Path to result json file with a dictionary.')
flags.DEFINE_integer('num_thread', 1, 'number_thread')
FLAGS = flags.FLAGS


def main(_):
    analyzer = VocAnalysis()
    analyzer.dataset_analysis(FLAGS.xml_path, number_thread=FLAGS.num_thread,json_path=FLAGS.result_path)


if __name__ == '__main__':
    """"
    ddddddddddddddddd
    """
    app.run(main)
