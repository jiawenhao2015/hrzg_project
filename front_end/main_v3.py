import sys,threading,os,cv2,time
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from  PyQt5.QtCore import QDateTime,QTimer
from PyQt5.QtGui import QImage, QPixmap
from window_layout_v3 import *
from PyQt5.QtCore import Qt

import ffmpeg
import numpy as np
from logconfig import *
from datetime import datetime
import traceback

from postprocess import *

class MyWindow(QMainWindow, Ui_mainWindow):

    def showtimer(self):
        self.date.setText(QDateTime.currentDateTime().toString('yyyy-MM-dd hh:mm:ss dddd'))


    def checkEnableControl(self):
        if self.enablesystemBox.isChecked():
            logger.info('enablesystemBox click!!!!!!!!')
            # with open('checkEnableControl.txt','w') as f:
            #     f.write('1')
    
    def checkEnableControl2(self, state):
        if state == Qt.Unchecked:
            print("Checkbox is unchecked")
            with open('checkEnableControl.txt','w') as f:
                f.write('0')
        else:
            print("Checkbox is checked!!!")
            with open('checkEnableControl.txt','w') as f:
                f.write('1')

    def forwardButtonClick(self):
        if not self.enablemanulBox.isChecked():
            QMessageBox.information(self,"警告","请先勾选手动控制辊道！",QMessageBox.Yes | QMessageBox.No)
            return 
        if self.forwardButton.isChecked():
            logger.info('forwardButton click!!!!!!!!')

            #control_roller(roller_forward_address)
            siemens.WriteBool(roller_before_saw_backward_address,0)
            siemens.WriteBool(roller_before_saw_forward_address,1)
            
            time.sleep(0.8)
            siemens.WriteBool(roller_before_saw_forward_address,0)
            
            #printReadResult('forwardButtonClick---',siemens.ReadBool(roller_forward_address), siemens.ReadBool(roller_backward_address))



    def backwardButtonClick(self):
        if not self.enablemanulBox.isChecked():
            QMessageBox.information(self,"警告","请先勾选手动控制辊道！",QMessageBox.Yes | QMessageBox.No)
            return 
        if self.backwardButton.isChecked():
            logger.info('backwardButton click!!!!!!!!')
            # control_roller(roller_backward_address)
            siemens.WriteBool(roller_before_saw_forward_address,0)
            siemens.WriteBool(roller_before_saw_backward_address,1)
            
            time.sleep(0.8)
            siemens.WriteBool(roller_before_saw_backward_address,0)
           # printReadResult('backwardButtonClick---',siemens.ReadBool(roller_forward_address), siemens.ReadBool(roller_backward_address))



    def stopButtonClick(self):
        if not self.enablemanulBox.isChecked():
            QMessageBox.information(self,"警告","请先勾选手动控制辊道！",QMessageBox.Yes | QMessageBox.No)
            return 
        if self.stopButton.isChecked():
            logger.info('stopButtonClick click!!!!!!!!')
            # control_roller(roller_stop_address)
            siemens.WriteBool(roller_before_saw_forward_address,0)
            siemens.WriteBool(roller_before_saw_backward_address,0)
            #printReadResult('stopButtonClick---',siemens.ReadBool(roller_forward_address), siemens.ReadBool(roller_backward_address))



    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)
        
        self.enablesystemBox.clicked.connect(self.checkEnableControl)

        self.enablesystemBox.stateChanged.connect(self.checkEnableControl2)

        
        self.forwardButton.clicked.connect(self.forwardButtonClick)
        self.backwardButton.clicked.connect(self.backwardButtonClick)
        self.stopButton.clicked.connect(self.stopButtonClick)

        self.Timer = QTimer()  # 自定义QTimer
        self.Timer.start(1000)  #间隔1s
        self.Timer.timeout.connect(self.showtimer)


        self.init_camera()
        logger.info('init_camera done...')
        
        #self.init_postprocess()
        #logger.info('init_postprocess done...')

       
    def init_postprocess(self):

        self.t_process_head = threading.Thread(target=Result(self.head_result,self.tail_result).show)
        #self.t_process_tail = threading.Thread(target=Result('tail_result',self.tail_result).show)
        
        self.t_process_head.setDaemon(True)
        #self.t_process_tail.setDaemon(True)

        self.t_process_head.start()
        #self.t_process_tail.start()

    def init_camera(self):

        self.t1 = threading.Thread(target=Camera('rtsp://admin:hrzg2024@192.168.66.101:554/Streaming/Channels/101', self.head_ori,'head').display)
        self.t2 = threading.Thread(target=Camera('rtsp://admin:rb123456@192.168.66.102:554/Streaming/Channels/101', self.tail_ori,'tail').display)
        
        self.t1.setDaemon(True)
        self.t2.setDaemon(True)
       
        self.t1.start()
        self.t2.start()
     

    
  
class Camera:

        def __init__(self, url, out_label, name):
            self.init = False
            self.url = url
            self.outLabel = out_label
            self.name = name
            
            bg_pic = cv2.imread('nosignal.png')

            self.bg_pic = cv2.resize(bg_pic, (640, 360)) 
            self.bg_qimg = QImage(self.bg_pic.data, self.bg_pic.shape[1], self.bg_pic.shape[0], QImage.Format_RGB888)

            self.args = {
                "rtsp_transport": "tcp",
                "fflags": "nobuffer",
                "flags": "low_delay",
                "loglevel": "quiet"
            }    # 添加参数
           
            
            self.width = 1280           # 获取视频流的宽度
            self.height = 720         # 获取视频流的高度
           
            
            self.process1 = (ffmpeg.input(url, **self.args).output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .overwrite_output().run_async(pipe_stdout=True))
            
            in_bytes = self.process1.stdout.read(self.width * self.height * 3)     # 读取图片
            if in_bytes:
               self.init = True
        

        def display(self):
            
            while True:

                if self.init == False:
                   
                    time.sleep(1)

                    self.process1 = (ffmpeg.input(self.url, **self.args).output('pipe:', format='rawvideo', pix_fmt='rgb24')
                        .overwrite_output().run_async(pipe_stdout=True))
                    
                    in_bytes = self.process1.stdout.read(self.width * self.height * 3)     # 读取图片
                    if not in_bytes:
                        self.outLabel.setPixmap(QPixmap.fromImage(self.bg_qimg))
                    else:
                        self.init = True

                    continue

                in_bytes = self.process1.stdout.read(self.width * self.height * 3)     # 读取图片
                if not in_bytes:
                    self.outLabel.setPixmap(QPixmap.fromImage(self.bg_qimg))
                    time.sleep(1)
                    self.init = False
                    continue
                
                in_frame = (np.frombuffer(in_bytes, np.uint8).reshape([self.height, self.width, 3]))
               
                frame = cv2.cvtColor(in_frame, cv2.COLOR_RGB2BGR)  # 转成BGR

                frame = cv2.resize(in_frame, (640, 360)) 
                img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                self.outLabel.setPixmap(QPixmap.fromImage(img))
                # if self.url.find('.102') > 0:

                #     time.sleep(1)
                #     print('time.sleep(1)')
                
            self.process1.kill()             # 关闭


import asyncio
import websockets
import base64
import json
class Result:

    def __init__(self, head_label, tail_label):
        self.init = False
        
        self.tailLabel = tail_label
        self.headLabel = head_label
        
        bg_pic = cv2.imread('nosignal.png')

        self.bg_pic = cv2.resize(bg_pic, (640, 360)) 
        
    async def start_async(self,uri):
        while 1:
            try:
                async with websockets.connect(uri) as websocket:
            
                    # while True:
                    #     img_data = await websocket.recv()  # 接收消息
                    #     img_buffer_numpy = np.frombuffer(img_data, dtype=np.uint8) 
                    #     frame = cv2.imdecode(img_buffer_numpy, 1) 
                    #     if frame is None or frame.size == 0:
                    #         continue
                    #     frame = cv2.resize(frame, (640, 360)) 
                    #     img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                    #     self.outLabel.setPixmap(QPixmap.fromImage(img))
                    while True:
                        response = await websocket.recv()  # 接收消息
                        
                        data_map = json.loads(response)
                        
                        logger.info(data_map['head_or_tail'] +' recv and parse done')

                        image_data = base64.b64decode(data_map['image'])
                        img_buffer_numpy = np.frombuffer(image_data, dtype=np.uint8) 
                        frame = cv2.imdecode(img_buffer_numpy, 1) 
                        if frame is None or frame.size == 0:
                            continue
                        frame = cv2.resize(frame, (640, 360)) 
                        img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                        if data_map['head_or_tail'] == 'head':
                            self.headLabel.setPixmap(QPixmap.fromImage(img))
                        else:
                            self.tailLabel.setPixmap(QPixmap.fromImage(img))

                        logger.info(data_map['head_or_tail'] + ' show done')

            except Exception as e:
                
                self.bg_qimg = QImage(self.bg_pic.data, self.bg_pic.shape[1], self.bg_pic.shape[0], QImage.Format_RGB888)
                self.headLabel.setPixmap(QPixmap.fromImage(self.bg_qimg))
                self.tailLabel.setPixmap(QPixmap.fromImage(self.bg_qimg))

                logger.error(str(e))
                traceback.print_exc()
                time.sleep(1)

    def show(self):
       
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.start_async('ws://127.0.0.1:6666'))
        
        logger.info('run in show...end')
        

if __name__ == "__main__":

    
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())
   

