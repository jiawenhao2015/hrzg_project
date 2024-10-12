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

#matplotlib.use('Agg')


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

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)

    ax.set_xlabel('Time')
    ax.set_ylabel('op')
    ax.set_title('plc signal')
   

    line_4 = None
    line_5 = None
    line_6 = None
    line_7 = None

    plt.grid(True) #添加网格
    plt.ion()  #interactive mode on
    obsX = []
    M4 = []
    M5 = []
    M6 = []
    M7 = []

    
    t0 = int(time.time() * 1000)

    while 1:
        #666.4  666.5锯前 4正 5反 666.6 666.7锯后 6 正 7反
        r4 = siemens.ReadBool("M666.4")
        r5 = siemens.ReadBool("M666.5")
        r6 = siemens.ReadBool("M666.6")
        r7 = siemens.ReadBool("M666.7")
        
        state = "M666.4 {} M666.5 {} M666.6 {} M666.7 {}".format(int(r4.Content), int(r5.Content), int(r6.Content), int(r7.Content))
        
        logger.info(state)

        t = (int(time.time() * 1000) - t0)//10
        obsX.append(t)
        M4.append(int(r4.Content))
        M5.append(int(r5.Content))
        M6.append(int(r6.Content))
        M7.append(int(r7.Content))

        if len(obsX) > 5000:
            obsX = obsX[2500:]
            M4 = M4[2500:]
            M5 = M5[2500:]
            M6 = M6[2500:]
            M7 = M7[2500:]
        
        if line_4 is None:
            line_4 = ax.plot(obsX,M4,color='green',label='M4')[0]
            line_5 = ax.plot(obsX,M4,color='red',label='M5')[0]
            line_6 = ax.plot(obsX,M4,color='c',label='M6')[0]
            line_7 = ax.plot(obsX,M4,color='b',label='M7')[0]

        line_4.set_xdata(obsX)
        line_4.set_ydata(M4)
        line_5.set_xdata(obsX)
        line_5.set_ydata(M5)
        line_6.set_xdata(obsX)
        line_6.set_ydata(M6)
        line_7.set_xdata(obsX)
        line_7.set_ydata(M7)

        ax.set_xlim([t-5000,t+100])
        ax.set_ylim([-0.2,1.2])
        ax.legend()

        plt.pause(0.001)
        #time.sleep(0.001)

#new plot 增加拐点时刻
def step2():

    #fig=plt.figure()
    fig = plt.figure(figsize=(19.2, 4.8), dpi=100)
    ax=fig.add_subplot(1,1,1)

    #ax.set_xlabel('Time') 
    #ax.set_title('plc signal')
   
    line_4 = None
    line_5 = None
    line_6 = None
    line_7 = None

    plt.grid(True) #添加网格
    #plt.ion()  #interactive mode on
    plt.ioff() # 禁用交互模式
    #plt.axis('off')
    #plt.tick_params(axis='x', which='major',pad=20, width=1)
    plt.gca().xaxis.set_major_locator(MultipleLocator(30))

    for tick in plt.gca().xaxis.get_major_ticks():
        tick.label1.set_rotation(90)

    obsX = []
    M4 = []
    M5 = []
    M6 = []
    M7 = []

    obsX_ = []
    M4_ = []
    M5_ = []
    M6_ = []
    M7_ = []

    
    t0 = int(time.time() * 1000)
    guandian_zuobiao =[]
    guandian_value =[]

    while 1:
        #666.4  666.5锯前 4正 5反 666.6 666.7锯后 6 正 7反
        r4 = siemens.ReadBool("M666.4")
        r5 = siemens.ReadBool("M666.5")
        r6 = siemens.ReadBool("M666.6")
        r7 = siemens.ReadBool("M666.7")
        
        state = "M666.4 {} M666.5 {} M666.6 {} M666.7 {}".format(int(r4.Content), int(r5.Content), int(r6.Content), int(r7.Content))
        
        logger.info(state)

        t = (int(time.time() * 1000) - t0)//10
        
        if t > 6000:
            t0 = t

        ax.set_xlim([t-1900,t+100])
        ax.set_ylim([0.0,1.1])
        ax.legend()

        tik = get_time_with_ms_format().split()[-1].split(':')[-1]
        if len(M4) > 10:
            if M4[-1] != M4[-2] :
                guandian_value.append(tik)
                guandian_zuobiao.append((t,M4[-1]))
            if M5[-1] != M5[-2]:
                guandian_value.append(tik)
                guandian_zuobiao.append((t,M4[-1]))
            if M6[-1] != M6[-2]:
                guandian_value.append(tik)
                guandian_zuobiao.append((t,M4[-1]))
            if M7[-1] != M7[-2]:
                guandian_value.append(tik)
                guandian_zuobiao.append((t,M4[-1]))


        obsX.append(t)
        M4.append(int(r4.Content))
        M5.append(int(r5.Content))
        M6.append(int(r6.Content))
        M7.append(int(r7.Content))

        if len(obsX) > 500:
            obsX = obsX[-200:]
            M4 = M4[-200:]
            M5 = M5[-200:]
            M6 = M6[-200:]
            M7 = M7[-200:]
        
        if len(guandian_zuobiao) > 12:
            guandian_zuobiao = guandian_zuobiao[-6:]
            guandian_value = guandian_value[-6:]

        if line_4 is None:
            line_4 = ax.plot(obsX,M4,color='green',label='M4')[0]
            line_5 = ax.plot(obsX,M5,color='red',label='M5')[0]
            line_6 = ax.plot(obsX,M6,color='c',label='M6')[0]
            line_7 = ax.plot(obsX,M7,color='b',label='M7')[0]



        line_4.set_xdata(obsX)
        line_4.set_ydata(M4)
        line_5.set_xdata(obsX)
        line_5.set_ydata(M5)
        line_6.set_xdata(obsX)
        line_6.set_ydata(M6)
        line_7.set_xdata(obsX)
        line_7.set_ydata(M7)

        # ta = int(time.time() * 1000)
        # for (x, y), v in zip(guandian_zuobiao, guandian_value):
        #     if t - x > 1900:
        #         continue
        #     plt.text(x, float(y), v, ha='center', va='bottom', fontsize=9, rotation=90)

        # tb = int(time.time() * 1000)
        # print('plt text time:', tb-ta)

        

        plt.pause(0.01)
        #time.sleep(0.001)

           
		

if __name__ == '__main__':
    
   step2()
   pass
	