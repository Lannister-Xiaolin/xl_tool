#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
    多线程模块api
"""

import os
from functools import wraps
import threading
from math import ceil
import logging


class MyThread(threading.Thread):
    """带返回值的多线程类"""

    def __init__(self, target, args=(), kwargs=None):
        super(MyThread, self).__init__()
        self.target = target
        self.args = args
        self.kwargs = kwargs if kwargs else dict()
        self.result = ""

    # 重新定义带返回值的线程类
    def run(self):
        try:
            if self.target:
                self.result = self.target(*self.args, **self.kwargs)
                logging.info("成功获取返回结果")
        finally:
            # Avoid a refcycle if the thread is running a function with
            # an argument that has a member that points to the thread.
            del self.target, self.args, self.kwargs
    def get_result(self):
        try:
            return self.result
        except Exception as e:
            logging.warning(str(e))
            return None


def mul_thread(function, thread_param, share_param=(), thread_num=8):
    threads = list(range(thread_num))
    handle_num = ceil(len(thread_param) / thread_num)
    for i in range(thread_num):
        args = [thread_param[i * handle_num:(i + 1) * handle_num]]
        args.extend(list(share_param))
        threads[i] = MyThread(target=function,
                              args=tuple(args))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
