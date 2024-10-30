from HslCommunication import SiemensS7Net
from HslCommunication import SiemensPLCS
import time
import threading
from threading import Thread

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime

from logconfig import *
from matplotlib.ticker import MultipleLocator




def str_time_to_int(time_str):
   # 假设 time_str 的格式为 "HH:MM:SS.mmm"
    datetime_obj = datetime.strptime(time_str, "%H:%M:%S.%f")
    timestamp = datetime_obj.timestamp()
    timestamp_in_ms = int(timestamp * 1000)
    return timestamp_in_ms
def get_time_with_ms_format():
    # 获取当前时间
    now = datetime.now()
    # 将datetime对象转换为字符串，包含毫秒
    timestamp = now.strftime('%Y-%m-%d %H:%M:%S.%f')
    timestamp = timestamp[:-3]  # 去掉末尾的3个0，只保留毫秒(ms)
    return timestamp

def printReadResult(prefix,result):
    if result.IsSuccess:
        logger.info(prefix + str(int(result.Content)))
        #print(prefix + result.Content)
    else:
        logger.warning("read plc failed "+ result.Message)
def printWriteResult(prefix,result):
    if result.IsSuccess:
        logger.info(prefix + " write plc success")
    else:
         logger.warning(prefix + " write plc falied  " + result.Message)


plc_valid = False
roller_forward_code = 0
roller_backward_code = 1
roller_stop_code = 2

#锯前辊道控制 只能写  
# M666.0 正
# M666.1 反

#M666.2 锯前到位

roller_before_saw_forward_address = "M666.0"
roller_before_saw_backward_address = "M666.1"

roller_finish_address = "M666.2"

#锯后辊道控制 只能写 
#M666.3 正
#M666.4 反

roller_after_saw_forward_address = "M666.3"
roller_after_saw_backward_address = "M666.4"

siemens = SiemensS7Net(SiemensPLCS.S1500, "192.168.0.20")



##M666.0 666.1   666.2
###正转 反转  到位

if siemens.ConnectServer().IsSuccess == True:
	logger.info('')
	logger.info('PLC connect success!')
	logger.info('')
	plc_valid = True
else:
	logger.info('')
	logger.error('PLC connect fail!')
	logger.info('')
	exit()


