import cv2
import queue
from logconfig import *
import time

from HslCommunication import SiemensS7Net
from HslCommunication import SiemensPLCS


siemens = SiemensS7Net(SiemensPLCS.S1500, "192.168.0.20")

connect_plc = False
# if self.siemens.ConnectServer().IsSuccess == True:
#     logging.info('plc connect success!!! 连接成功!')
#     self.connect_plc = True
# else:
#     logging.info('plc connect failed!!! 连接失败!')
#     logging.info(self.siemens.ConnectServer().ToMessageShowString())
    
# write1 = self.siemens.WriteBool("M666.0",0)
# print("step1 write1:",write1.ToMessageShowString())



head_queue = queue.Queue(25)
tail_queue = queue.Queue(25)
stop_threads = False

def process_head():

    while(1):
        global stop_threads
        if stop_threads:
            logger.info('process_head stop_threads ... exit')
            break
        if head_queue.empty():
            time.sleep(2)
            continue

        qsize = head_queue.qsize()
        #logger.info('head_queue:%d',qsize)

        if qsize > 2:
            break

    pass



def process_tail():

    while(1):
        global stop_threads
        if stop_threads:
            logger.info('process_tail stop_threads ... exit')
            break
        if tail_queue.empty():
            time.sleep(2)
            continue

        qsize = tail_queue.qsize()
        #logger.info('tail_queue:%d',qsize)

        if qsize > 2:
            break

    pass



