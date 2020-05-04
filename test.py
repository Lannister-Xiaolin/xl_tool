#!usr/bin/env python3
# -*- coding: UTF-8 -*-
from xl_tool.data.image.analysis import VocAnalysis
def analysis_test():
    anaer = VocAnalysis()
    (anaer.dataset_analysis(r"F:\Large_dataset\VOC\VOCtest_06-Nov-2007\VOCdevkit"
                            r"\VOC2007\Annotations",1,r"F:\1.json",r"F:/"))
    temp1 = anaer.results["boxes"]["labels"]
    # # print(anaer.results["boxes"]["labels"])
    # (anaer.dataset_analysis(r"F:\Large_dataset\VOC\VOCtest_06-Nov-2007\VOCdevkit"
    #                         r"\VOC2007\Annotations",1))
    # print(temp1 == anaer.results["boxes"]["labels"])
    # (anaer.dataset_analysis(r"F:\Large_dataset\VOC\VOCtest_06-Nov-2007\VOCdevkit"
    #                         r"\VOC2007\Annotations",2))
    # print(temp1 == anaer.results["boxes"]["labels"])
    # print(anaer.results)
analysis_test()