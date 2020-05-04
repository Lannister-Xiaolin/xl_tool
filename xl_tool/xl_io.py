#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
    常用io读取函数
"""

import os
import re
import chardet
import random
# from aip import AipNlp
import time
import requests
import base64
import json
import numpy as np
from functools import wraps
from math import ceil
import logging
import zipfile


def ali_ocr_api(image_path, url="",
                appcode="",
                prob=False, charinfo=False, rotate=False,
                table=False, sortpage=False):
    """阿里巴巴图片识别API"""
    with open(image_path, "rb") as f:
        img = base64.b64encode(f.read()).decode()
    body = {
        "img": img,
        "prob": prob,  # 是否需要识别结果中每一行的置信度，默认不需要。 true：需要 false：不需要
        "charInfo": charinfo,  # 是否需要单字识别功能，默认不需要。 true：需要 false：不需要
        "rotate": rotate,  # 是否需要自动旋转功能，默认不需要。 true：需要 false：不需要
        "table": table,  # 是否需要表格识别功能，默认不需要。 true：需要 false：不需要
        "sortPage": sortpage  # 字块返回顺序，false表示从左往右，从上到下的顺序，true表示从上到下，从左往右的顺序，默认false
    }
    headers = {"Content-Type": "application/octet-stream; charset=utf-8", "Authorization": "APPCODE {}".format(appcode)}
    try:
        response = requests.post(url, json.dumps(body).encode(encoding='UTF8'), headers=headers)
        if response.status_code == 200:
            print("成功解析图片！！！")
            return response.json()
        else:
            print("访问错误：", response.status_code)
            return False
    except Exception as e:
        print("发生错误！！！")
        print(e)
        return False
def dict_transpose(dict_):
    """键值对对换"""
    return dict([j,i] for i,j in dict_.items())

def zip(path, zip_filename,compression=zipfile.ZIP_DEFLATED):
    """ compress file or directory"""
    z = zipfile.ZipFile(zip_filename, "w",compression)
    if os.path.isfile(path):
        z.write(path)
    else:
        for root, dirs, files in os.walk(path, topdown=True):
            if files:
                for file in files:
                    z.write(f"{root}/{file}")
            else:
                z.write(root)
    z.close()

def file_scanning(path, file_format=r".txt$", full_path=True, sub_scan=False):
    """
        scanning directory and return file paths with specified format
        :param path: directory to scan
        :param file_format:  file format to return ,regular patterns
        :param full_path: whether to return the full path
        :param sub_scan: whether to sanning the subfolder
        :return:file paths
        """
    if os.path.exists(path):
        file_paths = []
        for root, dirs, files in os.walk(path, topdown=True):
            paths = [file for file in files if re.search(file_format, file)]
            if full_path:
                paths = [os.path.join(root, file) for file in paths]
            file_paths.extend(paths)
            if not sub_scan:
                break
        if not file_paths:
            print("File with specified format not find")
            return []
    else:
        print("Invalid path!")
        return []
    return file_paths


def read_txt(file, return_list=False, remove_linebreak=True):
    """read txt, deal with encoding error(rt-utf-8-chardet)
    Args:
        file: full file path
        return_list: read by line and return a list
        remove_linebreak: only valid when return_list=True, remove line break for every line text
    Returns:
        text or list
        """
    if not os.path.exists(file):
        print("file not find!", file)
        return False
    try:
        with open(file, "rt") as f:
            if return_list:
                text = f.readlines()
                if remove_linebreak:
                    text = [t.rstrip("\n\r") for t in text]
            else:
                text = f.read()
    except (UnicodeDecodeError, UnicodeEncodeError):
        # print("encoding error, try encoding with utf-8")
        try:
            with open(file, "rt", encoding="utf-8") as f:
                if return_list:
                    text = f.readlines()
                    if remove_linebreak:
                        text = [t.rstrip("\n\r") for t in text]
                else:
                    text = f.read()
        except UnicodeDecodeError:
            # print("encoding error, try encoding with chardet, and ignore error")
            with open(file, "rb") as f:
                text_byte = f.read()
                encoding = chardet.detect(text_byte)["encoding"]
                text = text_byte.decode(encoding, errors="ignore")
                if return_list:
                    text = [i for i in text.split("\n")]
                    if remove_linebreak:
                        text = [t.rstrip("\n\r") for t in text]
        except PermissionError:
            print("not a valid path, please confirm")
            return False
    return text



# def baidu_nlp_client(APP_ID,API_KEY,SECRET_KEY):
    # """ 你的 APPID AK SK """
    # client = AipNlp(APP_ID, API_KEY, SECRET_KEY)
    # return client


# def ner_detect(text, net_type=("PER",), remove_blank=True):
    # """
    # Args:
    # :param text:检测文本
    # :param net_type: 实体检测类型，["ORG", "LOC", "PER"，"TIME"]
    # :param remove_blank:是否去除无用字符，空格代替
    # :return:
    # """
    # ner = {ner_name: [] for ner_name in net_type}  # set类型去重
    # if remove_blank:
        # text = re.sub(r"\W|\d", " ", text)  # 去掉数字、特殊字符、空字符等非汉字英文字符
    # if not text:
        # print("实体检测模块：未发现有效文本")
        # return ner
    # # print("发送实体检测文本：", text)
    # # time_s = datetime.now()
    # client = baidu_nlp_client()
    # try:
        # ner_result = client.lexer(text)
    # except requests.exceptions.ConnectionError:
        # time.sleep(0.2)
        # print("！！！百度实体检测接口连接失败尝试，尝试再次连接")
        # try:
            # ner_result = client.lexer(text)
        # except requests.exceptions.ConnectionError:
            # print("！！！百度实体检测接口连接失败尝试，返回空字典")
            # return ner
    # # print(ner_result)
    # try:
        # for item in ner_result["items"]:
            # if item["ne"] in net_type:
                # ner[item["ne"]].append(item["item"])
                # print("检测到实体: ", item["ne"], item["item"])
    # except KeyError:
        # print("未检测到有效文本")
    return ner


def time_analysis_wrapper(name="默认", show_time=True):
    """时间分析装饰器"""

    def time_analysis(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            st = time.time()
            result = func(*args, **kwargs)
            et = time.time()
            if show_time:
                print("———程序：{}———运行时间(s)：{:.4f}".format(name, (et - st)))
            return result

        return wrapper

    return time_analysis


@time_analysis_wrapper('实体识别')
def ner_detect_client(text, client, ner_type=("PER",), remove_blank=True):
    """百度实体识别封装接口
    Args:
        text:检测文本
        client: api客户端实例
        ner_type: 实体检测类型，["ORG", "LOC", "PER"，"TIME"]
        remove_blank: 是否去除非中文和英文字符
    Returns:
        字典形式{ner type:ner text}
    """
    ner2pos = {'PER': 'nr', 'LOC': 'ns', 'ORG': 'nt'}
    pos2ner = {'nr': 'PER', 'ns': 'LOC', 'nt': 'ORG'}
    ner = {ner_name: [] for ner_name in ner_type}
    ner_result = dict()
    pos_type = [ner2pos[i] for i in ner_type]
    if remove_blank:
        text = re.sub(r"\W|\d", " ", text)  # 去掉数字、特殊字符、空字符等非汉字英文字符
    if not text:
        print("实体检测模块：未发现有效文本")
        return ner
    for i in range(2):
        try:
            ner_result = client.lexer(text)
            break
        except requests.exceptions.ConnectionError:
            time.sleep(0.2)
            print("！！！百度实体检测接口连接失败尝试，尝试再次连接")
        # 特殊字符编码问题，直接替换掉特殊字符
        except UnicodeEncodeError:
            for ch in list(text):
                try:
                    ch.encode("gbk")
                except UnicodeEncodeError:
                    text = text.replace(ch, "")
            try:
                ner_result = client.lexer(text)
                break
            except Exception as e:
                print("出现未知错误:", e)
                return ner
    try:
        for item in ner_result["items"]:
            if item["ne"] in ner_type or item['pos'] in pos_type:
                ner_name = item["ne"] if item["ne"] else pos2ner[item["pos"]]
                ner[ner_name].append(item["item"])
                print("检测到实体: ", item["ne"], item["item"])
    except KeyError:
        print("未检测到有效文本")
    return ner




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


def random_phone(mul_form=True):
    """随机生成11位有效的手机号，包含四种格式连接符
    Args:
        mul_form: 是否随机选择连接符如xxx-xxxx-xxxx的格式
    """
    full_num = '0123456789'
    # 手机号头三个数字有效性要求
    head_1 = ['13', '14', '15', '17', '18', '16', '19']
    head_2 = [full_num, '57', '012356789', '01235678', full_num, '2567', '189']
    index = random.randint(0, 6)
    head = head_1[index] + random.choice(head_2[index])
    middle = "".join([random.choice(full_num) for i in range(4)])
    tail = "".join([random.choice(full_num) for i in range(4)])
    # 连接格式
    random_form = ["-", "", " ", "—"]
    if mul_form:
        phone = random.choice(random_form).join([head, middle, tail])
    else:
        phone = head + middle + tail
    return phone


def random_email():
    """随机生成邮件地址"""
    dns = ["163.com", "126.com", "139.com", "qq.com", "sohu.com", "gmail.com", "foxmail.com", "outlook.com", "sina.com",
           "vip.163.com",
           "sina.cn", "qiye.aliyun.com", "21cn.com", "china.com", "bit.edu.cn", "ncu.edu.cn", "ustc.edu.cn",
           "corpease.net", "263.net", "shdongchang.com", "gmail.kenfor.net", "cumark.com.cn"]
    full = "0123456789qwertyuiopasdfghjklzxcvbnm"
    head_form = ["纯字母数字", "2段", "3段"]
    form = random.choice(head_form)
    if form == "纯数字字母":
        head = "".join([random.choice(full) for i in range(random.randint(6, 18))])
    elif form == "2段":
        head = "".join([random.choice(full[10:]) for i in range(random.randint(3, 8))])
        head = head + "_" + "".join([random.choice(full) for i in range(random.randint(3, 8))])
    else:
        head = "".join([random.choice(full) for i in range(random.randint(3, 8))])
        head = head + "_" + "".join([random.choice(full[10:]) for i in range(random.randint(3, 8))])
        head = head + "_" + "".join([random.choice(full) for i in range(random.randint(3, 8))])
    tail = random.choice(dns)
    email = head + "@" + tail
    return email


def read_json(filename):
    """读取json格式的字典与ID对应表
    Args:
        filename:文件名
    Returns:
        json转换python数据类型的数据，无数据或路径错误返回False"""
    try:
        if os.path.basename(filename) == filename:
            filename = os.getcwd() + '/' + filename
        with open(filename, mode="r", encoding="utf-8", buffering=1000000000) as f:
            data = json.load(f)
            if data:
                return data
            else:
                print("未读取到有效json数据！")
                return False
    except (FileExistsError, FileNotFoundError):
        print("未找到文件或者文件路径不合法！")
        return False


def read_with_base64(file):
    with open(file, 'rb') as f:
        file_base64 = base64.b64encode(f.read()).decode("utf-8")
    return file_base64


def save_to_json(data, filename, indent=None):
    """保存到json文件中，indent是否缩进
    Args:
        filename: 文件名，推荐绝对路径，如只有文件名则保存当前路径下
        data:保存数据
        indent:json格式缩进"""
    try:
        if os.path.basename(filename) == filename:
            filename = os.getcwd() + '/' + filename
        with open(filename, mode="w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)

    except (FileExistsError, FileNotFoundError):
        print("未找到文件或者文件路径不合法！")


