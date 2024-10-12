from HslCommunication import SiemensS7Net
from HslCommunication import SiemensPLCS
import time
import threading
from threading import Thread

siemens = SiemensS7Net(SiemensPLCS.S1500, "192.168.0.20")

##666.0 666.1   666.2
###正转 反转  到位

if siemens.ConnectServer().IsSuccess == True:
	print ('')
	print ('连接成功 !')
	print ('')
	print ('请按启动')
else:
	print ('')
	print ('连接失败 !')
	print ('')
	exit ()

def step1():
	while 1:
		read1 = siemens.ReadBool("I0.1")#I0.1 左启动
		if read1.Content == True:
			time.sleep(1)
			siemens.WriteBool("Q0.5",True)#Q0.5 侧推气缸
			time.sleep(1)
			siemens.WriteBool("Q0.5",False)
			time.sleep(1)
			siemens.WriteBool("Q0.5",True)
			print ('===============================')
			print ('###侧推气缸到位###')
			time.sleep(1)
			siemens.WriteBool("Q0.6",True)#Q0.6 下压气缸
			time.sleep(1)
			print ('###下压气缸到位###')
			time.sleep(1)
			siemens.WriteBool("Q0.1",True)#Q0.1 计时器开始
			print ('计时器开始计时...')



			read1 = siemens.ReadBool("M666.0")#I0.1 左启动
			print("step1 read1:",read1)
			write1 = siemens.WriteBool("M666.0",0)
			print("step1 write1:",write1.ToMessageShowString())

			read2 = siemens.ReadByte("M666.1")#I0.1 左启动
			print("step1 read2:",read2.Message)
			write2 = siemens.WriteBool("M666.1",0)
			print("step1 write1:",write1.ToMessageShowString())

			read3 = siemens.ReadByte("M666.2")#I0.1 左启动
			print("step1 read3:",read3.Message)
			write1 = siemens.WriteBool("M666.2",0)
			print("step1 write3:",write1.ToMessageShowString())
			print(write2.IsSuccess)


def step2():
	while 1:
		read2 = siemens.ReadBool("I0.3")#I0.3 计时完成
		if read2.Content == True:
			print ('###计时完成###')
			time.sleep(1)
			siemens.WriteBool("Q0.6",False)
			print ('###下压气缸复位###')
			siemens.WriteBool("Q0.1",False)
			time.sleep(1)
			siemens.WriteBool("Q0.5",False)
			print ('###侧推气缸复位###')
			print ('===============================')
			print ('请再次启动')

if __name__ == '__main__':
	Thread(target = step1).start()
	Thread(target = step2).start()