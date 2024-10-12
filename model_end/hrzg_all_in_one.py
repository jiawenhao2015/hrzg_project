import sys
sys.path.append('/home/user/hrzg/work/mmsegmentation/mmseg/')


from mmseg.apis import inference_model, init_model, show_result_pyplot
import mmcv
import torch
import os
from tqdm import tqdm
import threading
import cv2
import numpy as np
import subprocess as sp

from HslCommunication import SiemensS7Net
from HslCommunication import SiemensPLCS
import time
from datetime import datetime
import traceback

import ffmpeg

Lock_tou = threading.Lock()
Lock_wei = threading.Lock()
import copy

from config import *
from logconfig import *

siemens = None
plc_valid = False

picture_tou = None
picture_wei = None

model_tou = None
model_wei = None


class RTSCapture(cv2.VideoCapture):
    """Real Time Streaming Capture.
    这个类必须使用 RTSCapture.create 方法创建，请不要直接实例化
    """

    _cur_frame = None
    _reading = False
    schemes = ["rtsp://", "rtmp://"] #用于识别实时流

    @staticmethod
    def create(url, *schemes):
        """实例化&初始化
        rtscap = RTSCapture.create("rtsp://example.com/live/1")
        or
        rtscap = RTSCapture.create("http://example.com/live/1.m3u8", "http://")
        """
        rtscap = RTSCapture(url)
        rtscap.frame_receiver = threading.Thread(target=rtscap.recv_frame, daemon=True)
        rtscap.schemes.extend(schemes)
        if isinstance(url, str) and url.startswith(tuple(rtscap.schemes)):
            rtscap._reading = True
        elif isinstance(url, int):
            # 这里可能是本机设备
            pass

        return rtscap

    def isStarted(self):
        """替代 VideoCapture.isOpened() """
        ok = self.isOpened()
        if ok and self._reading:
            ok = self.frame_receiver.is_alive()
        return ok

    def recv_frame(self):
        """子线程读取最新视频帧方法"""
        while self._reading and self.isOpened():
            ok, frame = self.read()
            if not ok: break
            self._cur_frame = frame
        self._reading = False

    def read2(self):
        """读取最新视频帧
        返回结果格式与 VideoCapture.read() 一样
        """
        frame = self._cur_frame
        self._cur_frame = None
        return frame is not None, frame

    def start_read(self):
        """启动子线程读取视频帧"""
        self.frame_receiver.start()
        self.read_latest_frame = self.read2 if self._reading else self.read

    def stop_read(self):
        """退出子线程方法"""
        self._reading = False
        if self.frame_receiver.is_alive(): self.frame_receiver.join()


def generate_pseudo_masks(config_file, checkpoint_file, dir_save_pseudo_masks, list_images):
    model = init_model(config_file, checkpoint_file, device='cuda:0')
    PALETTE = []
    for i in range(150):
        PALETTE.append([i, i, i])
    model.PALETTE = PALETTE

    if not os.path.exists(dir_save_pseudo_masks):
        os.mkdir(dir_save_pseudo_masks)

    for image_name in tqdm(list_images):
        img = mmcv.imread(image_name)
        result = inference_model(model, img)
        #vis_result = show_result_pyplot(model, img, result, with_labels=False, show=False,opacity=0.2)
        # mmcv.imwrite(vis_result, os.path.join(dir_save_pseudo_masks, image_name.split('/')[-1]))
 

def init_model_hrzg(config_file, checkpoint_file):
    print('init_model...')
    model = init_model(config_file, checkpoint_file, device='cuda:0')
    PALETTE = []
    for i in range(150):
        PALETTE.append([i, i, i])
    model.PALETTE = PALETTE

    print('init model success', checkpoint_file)


    img = mmcv.imread('test.jpg')
    for i in range(5):
        inference_model(model, img)
    print('warm up done...')
    return model


def get_rtsp_wei(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)
    print(datetime.now().strftime('%Y%m%d%H%M%S%f'), "get_rtsp_wei|cap.isOpened():", cap.isOpened())

    global picture_wei
    cnt = 0
    while True:
        try:
            ret, frame = cap.read()
            if ret == 0:
                print(datetime.now().strftime('%Y%m%d%H%M%S%f'), "get_rtsp_wei|read error")
                Lock_wei.acquire()
                picture_wei = None
                Lock_wei.release()
                cap.release()
                cap = cv2.VideoCapture(rtsp_url)
                continue
            Lock_wei.acquire()
            
            picture_wei = copy.deepcopy(frame)
            # cv2.imwrite("/mnt/nfs/jwh/code/debug/tmp/"+str(cnt)+'.jpg',picture)
            # print('count--',cnt)
            # cnt +=1
            Lock_wei.release()
            # count += 1
            # print("get_rtsp|count: ",count)
        except Exception as e:
            print(datetime.now().strftime('%Y%m%d%H%M%S%f'), 'get_rtsp_wei|exception:', e)
            traceback.print_exc()
            Lock_wei.acquire()
            picture_wei = None
            Lock_wei.release()

            cap.release()
            cap = cv2.VideoCapture(rtsp_url)


def get_rtsp_tou(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)
    print(datetime.now().strftime('%Y%m%d%H%M%S%f'), "get_rtsp_tou|cap.isOpened():", cap.isOpened())

    global picture_tou
    cnt = 0
    while True:
        try:
            ret, frame = cap.read()
            if ret == 0:
                print(datetime.now().strftime('%Y%m%d%H%M%S%f'), "get_rtsp_tou|read error")
                Lock_tou.acquire()
                picture_tou = None
                Lock_tou.release()
                cap.release()
                cap = cv2.VideoCapture(rtsp_url)
                continue
            Lock_tou.acquire()
            
            picture_tou = copy.deepcopy(frame)
            # cv2.imwrite("/mnt/nfs/jwh/code/debug/tmp/"+str(cnt)+'.jpg',picture)
            # print('count--',cnt)
            # cnt +=1
            Lock_tou.release()
            # count += 1
            # print("get_rtsp|count: ",count)
        except Exception as e:
            print(datetime.now().strftime('%Y%m%d%H%M%S%f'), 'get_rtsp_tou|exception:', e)
            traceback.print_exc()
            Lock_tou.acquire()
            picture_tou = None
            Lock_tou.release()

            cap.release()
            cap = cv2.VideoCapture(rtsp_url)


def init_PLC(plc_ip):
    print('init PLC...')
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

def control_roller(order):
    if not plc_valid:
        logger.warning("control_roller failed! plc init failed!")
        return
    if not enable_control:
        logger.warning("control_roller failed! enable_control == false!, change it...")
        return
        
    if order == roller_forward_code:
        siemens.WriteBool(roller_forward_address,1)
        printReadResult(siemens.ReadBool(roller_forward_address))
    elif order == roller_backward_code:
        siemens.WriteBool(roller_backward_address,1)
        printReadResult(siemens.ReadBool(roller_backward_address))
    elif order == roller_stop_code:
        siemens.WriteBool(roller_stop_address,1)
        printReadResult(siemens.ReadBool(roller_stop_address))
    else:
         logger.warning("control_roller failed, error order code " + order)


def predict_tou_bak(model_tou):

    img = copy.deepcopy(picture_tou)
    image_mmcv_from_opencv = mmcv.array_from_image(img)

    model_result = inference_model(model_tou, image_mmcv_from_opencv)
    
    sem_seg = model_result.pred_sem_seg
    sem_seg = sem_seg.cpu().data
    ids = np.unique(sem_seg)[::-1]
    legal_indices = ids < 2
    ids = ids[legal_indices]
    labels = np.array(ids, dtype=np.int64)

    mask = np.zeros_like(img, dtype=np.uint8)
    palette = [(0,0,0),(255,255,255)]
    colors = [palette[label] for label in labels]
    for label, color in zip(labels, colors):
            mask[sem_seg[0] == label, :] = color


def predict_tou(picture_tou):


    img = copy.deepcopy(picture_tou)
    # image_mmcv_from_opencv = mmcv.array_from_image(img)
    image_mmcv_from_opencv = img

    model_result = inference_model(model_tou, image_mmcv_from_opencv)
    
    sem_seg = model_result.pred_sem_seg
    sem_seg = sem_seg.cpu().data
    
    mask_ = sem_seg[0].numpy().astype(np.uint8)*255
    mask =np.expand_dims(mask_,axis=2)

    mask = np.concatenate((mask, mask, mask), axis=-1)

    combine = cv2.hconcat([picture_tou, mask])
    cv2.imwrite('tmp_mask.png', combine)

    pipe = sp.Popen(command_head, stdin=sp.PIPE) #,shell=False
    pipe.stdin.write(mask.tobytes())  # 存入管道用于直播

    logger.info('predict tou done 1')


    # sem_seg = model_result.pred_sem_seg
    # sem_seg = sem_seg.cpu().data
    # ids = np.unique(sem_seg)[::-1]
    # legal_indices = ids < 2
    # ids = ids[legal_indices]
    # labels = np.array(ids, dtype=np.int64)

    # mask = np.zeros_like(img, dtype=np.uint8)
    # palette = [(0,0,0),(255,255,255)]
    # colors = [palette[label] for label in labels]
    # for label, color in zip(labels, colors):
    #         mask[sem_seg[0] == label, :] = color


def predict_wei(picture_wei):

    img = copy.deepcopy(picture_wei)
    # image_mmcv_from_opencv = mmcv.array_from_image(img)
    image_mmcv_from_opencv = img

    model_result = inference_model(model_wei, image_mmcv_from_opencv)
    
    sem_seg = model_result.pred_sem_seg
    sem_seg = sem_seg.cpu().data
    
    mask_ = sem_seg[0].numpy().astype(np.uint8)*255
    mask =np.expand_dims(mask_,axis=2)

    mask = np.concatenate((mask, mask, mask), axis=-1)

    # combine = cv2.hconcat([picture_tou, mask])
    # cv2.imwrite('tmp_mask.png', combine)

    pipe = sp.Popen(command_tail, stdin=sp.PIPE) #,shell=False
    pipe.stdin.write(mask.tobytes())  # 存入管道用于直播

    # sem_seg = model_result.pred_sem_seg
    # sem_seg = sem_seg.cpu().data
    # ids = np.unique(sem_seg)[::-1]
    # legal_indices = ids < 2
    # ids = ids[legal_indices]
    # labels = np.array(ids, dtype=np.int64)

    # mask = np.zeros_like(img, dtype=np.uint8)
    # palette = [(0,0,0),(255,255,255)]
    # colors = [palette[label] for label in labels]
    # for label, color in zip(labels, colors):
    #         mask[sem_seg[0] == label, :] = color




class Camera:

        def __init__(self, url):
            self.init = False
            self.url = url
          
            self.args = {
                "rtsp_transport": "tcp",
                "fflags": "nobuffer",
                "flags": "low_delay",
                "loglevel": "quiet"
            }    # 添加参数
          
            self.width = 1280
            self.height = 720
           
            self.process1 = (ffmpeg.input(url, **self.args).output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .overwrite_output().run_async(pipe_stdout=True))
            
            in_bytes = self.process1.stdout.read(self.width * self.height * 3)     # 读取图片
            if not in_bytes:
                logger.warning('self.process1.stdout.read is none...')
            else:
               self.init = True
        

        def display(self):
            
            while True:

                if self.init == False:
                    
                    time.sleep(2)

                    self.process1 = (ffmpeg.input(self.url, **self.args).output('pipe:', format='rawvideo', pix_fmt='rgb24')
                        .overwrite_output().run_async(pipe_stdout=True))
                    
                    in_bytes = self.process1.stdout.read(self.width * self.height * 3)     # 读取图片
                    if not in_bytes:
                        logger.warning('self.process1.stdout.read is none...')
                    else:
                        self.init = True

                    continue

                in_bytes = self.process1.stdout.read(self.width * self.height * 3)     # 读取图片
                if not in_bytes:
                    logger.warning('self.process1.stdout.read is none...')
                    time.sleep(2)
                    self.init = False
                    continue
                # 转成ndarray
                in_frame = (np.frombuffer(in_bytes, np.uint8).reshape([self.height, self.width, 3]))
                frame = cv2.cvtColor(in_frame, cv2.COLOR_RGB2BGR)  # 转成BGR
                
                if self.url.find('.101') > 0:
                    #predict tou
                    predict_tou(frame)
                    
                else:
                   predict_wei(frame)
               
                
                
            self.process1.kill()             


    

if __name__ == '__main__':
    
   
    logger.info('project start....')
    

    model_tou = init_model(config_file, checkpoint_file_wei)
    model_wei = init_model(config_file, checkpoint_file_wei)



    logger.info('init model done....')

    
    t1 = threading.Thread(target=Camera(rtsp_tou).display)
    t2 = threading.Thread(target=Camera(rtsp_wei).display)


    logger.info('rtsp_tou wei thread done....')



    t1.start()
    t2.start()

    
    t1.join()
    t2.join()





