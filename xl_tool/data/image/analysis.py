#!usr/bin/env python3
# -*- coding: UTF-8 -*-
import os
import xml
from abc import abstractmethod
from .annonation import coordinate_name
from xl_tool.xl_io import file_scanning, time_analysis_wrapper, save_to_json
from xl_tool.xl_concurrence import MyThread
from math import ceil
from tqdm import tqdm
from copy import deepcopy
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter

plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）


class ObjectAnalysis:
    def __init__(self):
        pass

    @staticmethod
    def get_aspect_ratio(x1, y1, x2, y2, w=None, h=None, ignore_zero=True):
        """
        get aspect ratio(w/h)
        """
        try:
            w = w if w else x2 - x1
            h = h if h else y2 - y1
            # assert w > 0, "width can not be zero"
            # assert h > 0, "height can not be zero"
            aspect_ratio = w / h
        except ZeroDivisionError:
            if ignore_zero:
                return 0
            else:
                raise ZeroDivisionError
        return aspect_ratio

    @staticmethod
    def get_box_info(box_coordinate: tuple, image_w, image_h, image_area=None, coordinate_format="xxyy",
                     ignore_zero=True):
        """get box width, height, area ratio relative to whole image
        Args:
            box_coordinate: tuple,(x1,y1,x2,y2) whene coordinate_format is xxyy else (x,y,w,h)
            image_w: int or float
            image_h: int or float
            image_area: int or float or None, default None
            coordinate_format: string ,default xxyy, if not equal xxyy, meaning x,y,w,h
            ignore_zero: bool, whether to ignore zero divide, if True, return zero when zero divide happen
        """
        if coordinate_format == "xxyy":
            x, y = box_coordinate[:2]
            w, h = box_coordinate[2] - box_coordinate[0], box_coordinate[3] - box_coordinate[1]
        else:
            x, y, w, h = box_coordinate
        try:
            # w_ratio = w / image_w
            # h_ratio = h / image_h
            image_area = image_area if image_area else (image_h * image_w)
            box_area = w * h
            area_ratio = box_area / image_area
            box_aspect = w / h
            return w, h, box_area, area_ratio, box_aspect
        except ZeroDivisionError:
            if ignore_zero:
                return 0, 0, 0, 0, 0
            else:
                raise ZeroDivisionError

    @abstractmethod
    def dataset_analysis(self, annotation_path, number_thread=2):
        pass


class VocAnalysis(ObjectAnalysis):
    """VOC dataset analysis"""

    def __init__(self):
        super(VocAnalysis, self).__init__()
        self.results_template = {
            "images": {"aspects": [], "areas": [], "box_per_images": []},
            "boxes": {"aspects": [], "areas": [], "labels": [], "ratios": []}
        }
        self.results = None

    def extract_annotation(self, xml_file):
        dom = xml.dom.minidom.parse(xml_file)
        doc = dom.documentElement
        bndboxes = []
        for object_node in doc.getElementsByTagName('object'):
            name = object_node.getElementsByTagName('name')[0].firstChild.data
            coordinates = [int(object_node.getElementsByTagName(i)[0].firstChild.data) for i in coordinate_name]
            bndboxes.append({"name": name, "coordinates": coordinates})
        temp_node = doc.getElementsByTagName('size')[0]
        w = float(temp_node.getElementsByTagName("width")[0].firstChild.data)
        h = float(temp_node.getElementsByTagName("height")[0].firstChild.data)
        return w, h, bndboxes

    def single_annotation_analysis(self, xml_file):
        w, h, boxes = self.extract_annotation(xml_file)
        image_area = w * h
        image_aspect = w / h
        box_per_image = len(boxes)
        _, _, box_areas, area_ratios, box_aspects = zip(
            *[self.get_box_info(box["coordinates"], None, None, image_area) for box in boxes])
        labels = [i["name"] for i in boxes]
        return {"image": (image_aspect, image_area, box_per_image),
                "box": (labels, box_aspects, box_areas, area_ratios)}

    def batch_annotation_analysis(self, xml_files):
        batch_results = {
            "images": {"aspects": [], "areas": [], "box_per_images": []},
            "boxes": {"aspects": [], "areas": [], "labels": [], "ratios": []}
        }
        pbar = tqdm(xml_files)
        for xml_file in pbar:
            try:
                results = self.single_annotation_analysis(xml_file)
            except ValueError:
                print("NonAnnotation", xml_file)
                continue
            except ZeroDivisionError:
                print("ZeroDivisionError", xml_file)
                continue
            batch_results["images"]["aspects"].append(results["image"][0])
            batch_results["images"]["areas"].append(results["image"][1])
            batch_results["images"]["box_per_images"].append(results["image"][2])
            # box
            batch_results["boxes"]["aspects"].extend(results["box"][1])
            batch_results["boxes"]["areas"].extend(results["box"][2])
            batch_results["boxes"]["labels"].extend(results["box"][0])
            batch_results["boxes"]["ratios"].extend(results["box"][3])
            pbar.set_description("analysis progress: ")
        return batch_results

    @time_analysis_wrapper("分析耗时：")
    def dataset_analysis(self, annotation_path, number_thread=1, json_path=None, plt_path=None):
        """
        Args:
            annotation_path: 标注文件位置
            number_thread: 线程数量
            json_path : 用于保存分析结果文件名
            plt_path ：用于保存分析结果图片的路径
        Returns:
            Dict
        # Todo 图片结果未作分析，意义不大
        """
        self.results = deepcopy(self.results_template)
        xml_files = file_scanning(annotation_path, "xml", sub_scan=True)
        if number_thread == 1:
            self.results = self.batch_annotation_analysis(xml_files)
        else:
            threads = list(range(number_thread))
            handle_num = ceil(len(xml_files) / number_thread)
            for i in range(number_thread):
                args = (xml_files[i * handle_num:(i + 1) * handle_num],)
                threads[i] = MyThread(target=self.batch_annotation_analysis,
                                      args=args)
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            for thread in threads:
                results = thread.get_result()
                self.results["images"]["aspects"].extend(results["images"]["aspects"])
                self.results["images"]["areas"].extend(results["images"]["areas"])
                self.results["images"]["box_per_images"].extend(results["images"]["box_per_images"])
                # box
                self.results["boxes"]["aspects"].extend(results["boxes"]["aspects"])
                self.results["boxes"]["areas"].extend(results["boxes"]["areas"])
                self.results["boxes"]["labels"].extend(results["boxes"]["labels"])
                self.results["boxes"]["ratios"].extend(results["boxes"]["ratios"])
        self.results["boxes"]["areas"] = [i // 100 for i in self.results["boxes"]["areas"]]
        self.results["images"]["areas"] = [i // 100 for i in self.results["images"]["areas"]]

        labels = Counter(self.results["boxes"]["labels"])
        if plt_path:
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(1, 1, 1)
            ax.bar(*zip(*labels.items()))

            # plt.bar(*zip(*labels.items()))
            ax.set_xlabel("label")
            ax.set_ylabel("number")
            plt.xticks(rotation=45)
            # plt.show()
            plt.savefig(os.path.join(plt_path, "labels.png"))
        plt.cla()
        box_aspects = plt.hist(self.results["boxes"]["aspects"], bins=20, range=(0.01, 5))
        box_aspects = list(box_aspects[0]), list(box_aspects[1][1:])
        if plt_path:
            ax = plt.gca()
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax.xaxis.set_major_locator(MultipleLocator(0.500))
            plt.xlabel("box_aspects")
            plt.ylabel("number")
            plt.savefig(os.path.join(plt_path, "box_aspects.png"))
        plt.cla()
        box_areas = plt.hist(self.results["boxes"]["areas"], bins=20)
        box_areas = list(box_areas[0]), list(box_areas[1][1:])
        if plt_path:
            plt.xlabel("box_areas")
            plt.ylabel("number")
            ax = plt.gca()
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            plt.savefig(os.path.join(plt_path, "box_areas.png"))
        plt.cla()
        box_ratios = plt.hist(self.results["boxes"]["ratios"], bins=20)
        box_ratios = list(box_ratios[0]), list(box_ratios[1][1:])
        if plt_path:
            plt.xlabel("box_areas_ratio")
            plt.ylabel("number")
            ax = plt.gca()
            ax.xaxis.set_major_locator(MultipleLocator(0.0500))
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            plt.savefig(os.path.join(plt_path, "box_areas_ratio.png"))

        analysis_result = {"labels": labels, "box_aspects": box_aspects,
                           "box_areas": box_areas, "box_ratios": box_ratios}
        if json_path:
            save_to_json(analysis_result, json_path)
        return analysis_result
