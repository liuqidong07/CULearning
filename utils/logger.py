# -*- encoding: utf-8 -*-
'''
@File    :   logger.py
@Time    :   2020/10/21 11:09:28
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
import logging
import os

if not os.path.exists(r'./log/'):
    os.mkdir(r'./log/')

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s-%(name)s-%(levelname)s:%(message)s',
    filename=r'./log/test.txt', filemode='w')
logger = logging.getLogger("logger")
logger.info("Start print log.")
logger.warning("Warning! Nuclear missile launch!")
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s-%(name)s-%(levelname)s:%(message)s')
logger.info("Mission Complete")

def Log():
    ''''''
    
    pass








