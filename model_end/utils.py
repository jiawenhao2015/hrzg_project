import cv2
import os
import time
import numpy as np
from HslCommunication import SiemensS7Net
from HslCommunication import SiemensPLCS
from logconfig import *
from config import *

# 计算两点间的垂直距离
def point_to_line_distance(p, a, b):
    # 向量ab和向量pa
    ab = np.array(b) - np.array(a)
    pa = np.array(p) - np.array(a)
    
    # 计算斜率
    slope = ab[1] / ab[0] if ab[0] != 0 else np.inf
    
    # 计算垂直距离
    distance = abs(pa[1] - (slope * pa[0])) / np.sqrt(slope ** 2 + 1)
    
    return distance
 
# 计算交点
def find_intersection(p, a, b):
    # 直线方程的参数形式：y = mx + c
    m = (b[1] - a[1]) / (b[0] - a[0]) if b[0] - a[0] != 0 else np.inf
    c = a[1] - m * a[0]
    
    # 交点坐标
    intersection = [
        (c - p[1]) / (m - 0 if m - 0 != 0 else np.inf),
        m * (c - p[1]) / (m - 0 if m - 0 != 0 else np.inf) + p[1]
    ]
    
    return intersection
 
def line_coefficients(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    
    return A, B, C
 

def point_to_line_distance_and_intersection(point, line):
    # 点坐标
    x0, y0 = point
    # 直线系数 Ax + By + C = 0
    A, B, C = line

    # 计算垂直距离
    distance = abs(A * x0 + B * y0 + C) / np.sqrt(A**2 + B**2)

    # 计算交点
    x_inter = (B * (B * x0 - A * y0) - A * C) / (A**2 + B**2)
    y_inter = (A * (-B * x0 + A * y0) - B * C) / (A**2 + B**2)
    intersection = (x_inter, y_inter)

    return distance, intersection 

def read_enable_control(file_path):
    if not os.path.exists(file_path):
        return 0
    with open(file_path, 'r') as f:
        flag = f.readline().strip()
        print('read_enable_control ',flag)
        return int(flag)


def write_enable_control(file_path,flag):
    
    with open(file_path, 'w') as f:
        f.write(flag)
        print('write flag  ',flag)
        


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

siemens = None
plc_valid = False

ROLLER_BEFORE_SAW = 1
ROLLER_AFTER_SAW = 2

FORWARD_DIRECTION = 1
BACKWARD_DIRECTION = 2

flag_file_path = '/home/user/hrzg/work/project/front_end/checkEnableControl.txt'


siemens = SiemensS7Net(SiemensPLCS.S1500, plc_ip)
if siemens.ConnectServer().IsSuccess == True:
    print ('\n plc  siemens s1500连接成功 !\n ')
    plc_valid = True
else:
    print ('\n error plc  siemens s1500连接失败!\n ')


def printReadResult(result):
    if result.IsSuccess:
        print(result.Content)
    else:
        print("read plc failed "+result.Message)
def printWriteResult(result):
    if result.IsSuccess:
        print("write plc success")
    else:
        print("write plc falied  " + result.Message)


def control_roller(roller_id, direction, duration):
    if not plc_valid:
        logger.warning("control_roller failed! plc init failed!")
        return False
    if not read_enable_control(flag_file_path):
        logger.warning("control_roller failed! enable_control == false!, change it...")
        return False

    if roller_id == ROLLER_BEFORE_SAW:
        if direction == FORWARD_DIRECTION:
            siemens.WriteBool(roller_before_saw_backward_address, 0)
            siemens.WriteBool(roller_before_saw_forward_address, 1)
            time.sleep(duration)
            siemens.WriteBool(roller_before_saw_forward_address, 0)
            return True
        elif direction == BACKWARD_DIRECTION:
            siemens.WriteBool(roller_before_saw_forward_address, 0)
            siemens.WriteBool(roller_before_saw_backward_address, 1)
            time.sleep(duration)
            siemens.WriteBool(roller_before_saw_backward_address, 0)
            return True
        else:
            logger.warning(" ROLLER_BEFORE_SAW direction not support , change it...")
            return False
    
    elif roller_id == ROLLER_AFTER_SAW:
        if direction == FORWARD_DIRECTION:
            siemens.WriteBool(roller_after_saw_backward_address, 0)
            siemens.WriteBool(roller_after_saw_forward_address, 1)
            time.sleep(duration)
            siemens.WriteBool(roller_after_saw_forward_address, 0)
            return True

        elif direction == BACKWARD_DIRECTION:
            siemens.WriteBool(roller_after_saw_forward_address, 0)
            siemens.WriteBool(roller_after_saw_backward_address, 1)
            time.sleep(duration)
            siemens.WriteBool(roller_after_saw_backward_address, 0)
            return True
        else:
            logger.warning(" ROLLER_BEFORE_SAW direction not support , change it...")
            return False

    else:
        logger.warning("roller_id not support , change it...")
        return False

    

# #尾部相机画面锯切位置标定 a(x,y) b(x,y)  w,h

tail_a = (1,450)
tail_b = (495,1)  

roller_a = (280,666)
roller_b = (789,102)



head_a = (1,458)
head_b = (930,1)  