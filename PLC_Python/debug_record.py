from HslCommunication import SiemensS7Net
from HslCommunication import SiemensPLCS
import time
import threading
from threading import Thread

from logconfig import *


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


siemens = SiemensS7Net(SiemensPLCS.S1500, "192.168.0.20")

##M666.0 666.1   666.2
###正转 反转  到位

if siemens.ConnectServer().IsSuccess == True:
	print ('')
	print ('PLC 连接成功 !')
	print ('')
	
else:
	print ('')
	print ('连接失败 !')
	print ('')
	exit ()




def step1():
       while 1:
            #666.4  666.5锯前 4正 5反 666.6 666.7锯后 6 正 7反
            r4 = siemens.ReadBool("M666.4")
            printReadResult('M666.4 ----- ', r4)

            r5 = siemens.ReadBool("M666.5")
            printReadResult('M666.5 ----- ', r5)
            
            r6 = siemens.ReadBool("M666.6")
            printReadResult('M666.6 ----- ', r6)

            r7 = siemens.ReadBool("M666.7")
            printReadResult('M666.7 ----- ', r7)
            
            # w4 = siemens.WriteBool("M666.4",1)
            # printWriteResult('M666.4',w4)
            
            # w5 = siemens.WriteBool("M666.5",1)
            # printWriteResult('M666.4',w5)

            # w6 = siemens.WriteBool("M666.6",1)
            # printWriteResult('M666.6',w6)

            # w7 = siemens.WriteBool("M666.7",1)
            # printWriteResult('M666.7',w7)
            
            time.sleep(0.01)


            
		

if __name__ == '__main__':
    #exit()
    Thread(target = step1).start()
	