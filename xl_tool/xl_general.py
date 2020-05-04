"""通用预处理模块"""
import numpy as np
from xl_tool.xl_io import file_scanning
import os
import shutil
from random import shuffle


def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.

    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def auto_split_val(origin_data, train_path=None, val_path=None, split=0.2, file_format="jpg"):
    """自动划分数据集
    Args:
        origin_data :原始数据路径位置
        val_path: 验证集位置，如为空，则会在当前目录上级创建val名的文件夹
        split: 划分比例
        train_path: 训练集位置，默认为origin_data
    """
    if not train_path or origin_data == train_path:
        train_path = origin_data
    if not val_path:
        val_path = os.path.dirname(train_path) + "/val"
    cat_dirs = [d for d in os.listdir(origin_data) if os.path.isdir(origin_data + "/" + d)]
    print("搜索到以下类别：", "\t".join(cat_dirs))
    cat_num = [len(file_scanning(origin_data + "/" + dir_, file_format=file_format, sub_scan=True)) for dir_ in
               cat_dirs]
    cat_val = [int(split * num) for num in cat_num]
    print((sorted(zip(cat_dirs, cat_num), key=lambda x: x[1])))
    for i, dir_ in enumerate(cat_dirs):
        files = file_scanning(origin_data + "/" + dir_, file_format=file_format, full_path=True, sub_scan=True)
        shuffle(files)
        val_cat_path = val_path + "/" + dir_
        train_cat_path = train_path + "/" + dir_
        os.makedirs(val_cat_path, exist_ok=True)
        os.makedirs(train_cat_path, exist_ok=True)
        for j in range(cat_num[i]):
            src = files[j]
            if j < cat_val[i]:
                dst = val_cat_path + "/" + os.path.basename(src)
                shutil.move(src, dst)
            else:
                dst = train_cat_path + "/" + os.path.basename(src)
                shutil.move(src, dst)


def count_files(path, file_format=""):
    """统计path文件夹下面各个子文件包含指定格式的文件数量
    Args:
        path: 路径
        file_format: 文件格式，正则
    """
    cat_dirs = os.listdir(path)
    cat_dirs = [d for d in cat_dirs if os.path.isdir(path + "/" + d)]
    print(f"{path}：")
    cat_num = [len(file_scanning(path + "/" + dir_, file_format=file_format, sub_scan=True)) for dir_ in cat_dirs]
    count_result = sorted(zip(cat_dirs, cat_num), key=lambda x: x[1])
    print("\n".join([": ".join([str(j) for j in i]) for i in count_result]))
    return count_result


def binary_search_nearest_abs(sorted_arr, value, start, end):
    """查找与某个值距离最近（绝对值）的值"""
    if value >= sorted_arr[end]:
        return end
    if value <= sorted_arr[start]:
        return start
    if start == end:
        return start
    mid = (start + end) // 2
    left = abs(value - sorted_arr[mid])
    right = abs(value - sorted_arr[mid + 1])
    if left < right:
        start = start
        end = mid
        return binary_search_nearest_abs(sorted_arr, value, start, end)
    elif right < left:
        start = mid + 1
        end = end
        return binary_search_nearest_abs(sorted_arr, value, start, end)
    else:
        return mid

