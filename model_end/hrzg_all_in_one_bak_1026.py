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
import base64
import json
import asyncio
import websockets

from config import *
from logconfig import *
from utils import *


last_time_head = int(time.time() * 1000)
last_time_tail = int(time.time() * 1000)


picture_tou = None
picture_wei = None

model_tou = None
model_wei = None

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


import queue
to_send_que = queue.Queue()

async def post_to_frontend(websocket):
    while 1:
        try:
            while not to_send_que.empty():
                logger.info(to_send_que.qsize())
                item = to_send_que.get()
                await websocket.send(json.dumps(item))

        except Exception as e:

            logger.error(str(e))
            traceback.print_exc()
            time.sleep(0.1)



def parse_head_pic(mask):

    time.sleep(1)
    return
    
cur_dis = 9999
last_dis = 9999
    
def parse_tail_pic(combine):

    h,w,c = combine.shape
    ori = combine[:,0:w//2,:].copy()
    gray = combine[:,w//2:,0].copy()
    mask = combine[:,w//2:,:].copy()
    ori_bak = ori.copy()
    mask_bak = mask.copy()
    
    cv2.line(ori, tail_a, tail_b, (0,0,255), 3)
    cv2.line(ori, roller_a, roller_b, (0,255,255), 3)
    cv2.line(mask, tail_a, tail_b, (0,0,255), 3)
    cv2.line(mask, roller_a, roller_b, (0,255,255), 3)

    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    dis_list = []

    for con in contours:
        
        area = cv2.contourArea(con)
        if area < 5000:
            continue
        print('area:',area)
        rect = cv2.minAreaRect(con)
        
        box = cv2.boxPoints(rect)
        # 这一步不影响后面的画图，但是可以保证四个角点坐标为顺时针
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box,4-startidx,0)
        # 在原图上画出预测的外接矩形
        box = box.reshape((-1,1,2)).astype(np.int32)
        #print(box)
        # for i in range(4):
        #     cv2.circle(ori, box[i][0], 3,(0,255,0))
        #     cv2.putText(ori,str(i), box[i][0],0,1,(255,0,0),3)
        
        cv2.polylines(mask,[box],True,(0,255,0),10)
        # cv2.drawContours(mask, [con], 0, (0,0,255), 3)

        # 计算点到直线的垂直距离
        point = (int(box[3][0][0]), int(box[3][0][1]))
        #print('point:',point)
        distance, intersection = point_to_line_distance_and_intersection(point, line_coefficients(tail_a, tail_b))
        print(f"垂直距离: {distance}, 交点: {intersection}")
        logger.info(f"垂直距离: {distance}, 交点: {intersection}")
        dis_list.append(distance)
        cv2.line(mask, point, (int(intersection[0]),int(intersection[1])), (255,0,0), 3)
        cv2.line(ori, point, (int(intersection[0]),int(intersection[1])), (255,0,0), 3)
        cv2.putText(ori,str(int(distance)), point,0, 1,(255,0,0),3)


    combine = cv2.hconcat([ori, mask])
    lin2 = cv2.hconcat([ori_bak, mask_bak])
    combine = cv2.vconcat([lin2, combine])

    cv2.imwrite(os.path.join('/home/user/hrzg/data/1026_online/tail/',datetime.now().strftime('%Y%m%d%H%M%S%f')+'.jpg'), combine)

    ###control
    if len(dis_list):
        
        dis_list = sorted(dis_list)
        distance = dis_list[0]
        

        # step = int(distance // 100)
        

        # if step > 0:
        #     step = step + 1
        
        # if step == 0:

        #     # siemens.WriteBool(roller_before_saw_backward_address, 0)
        #     # siemens.WriteBool(roller_before_saw_forward_address, 1)
        #     # time.sleep(duration)
        #     siemens.WriteBool(roller_before_saw_forward_address, 0)
        # # else:
        # #     siemens.WriteBool(roller_before_saw_backward_address, 0)
        # #     siemens.WriteBool(roller_before_saw_forward_address, 1)
        # #     time.sleep(0.7)


        # logger.info('control....dis:' + str(distance) + ' step:' + str(step))

        # for i in range(step):
        #     duration = 0.8
        #     logger.info('!!!!!!!!control...' + str(i) + ':' + str(step)+' duration:'+ str(duration))
        #     control_roller(ROLLER_BEFORE_SAW, FORWARD_DIRECTION, duration)
        #     time.sleep(0.1)
        global last_dis
        if last_dis < 250 and last_dis > 60 and distance == last_dis:
            #说明钢坯停早了,
            siemens.WriteBool(roller_before_saw_forward_address, 1)
            time.sleep(1)
            siemens.WriteBool(roller_before_saw_forward_address, 0)
            

        if distance <  250:
            logger.info('!!!!!!!!control...stop' )
            
            write_enable_control(flag_file_path, '0')

            siemens.WriteBool(roller_before_saw_forward_address, 0)
            
            ###plc有一个现象,转完第一次,再转第二次会很短
        last_dis = distance
        

def predict(picture, head_or_tail):

    img = copy.deepcopy(picture)
   
    model_result = None
    if head_or_tail == 'head':
        model_result = inference_model(model_tou, img)
    else:
        model_result = inference_model(model_wei, img)

    


    sem_seg = model_result.pred_sem_seg
    sem_seg = sem_seg.cpu().data
    
    mask_ = sem_seg[0].numpy().astype(np.uint8)*255
    mask = np.expand_dims(mask_,axis=2)
    

    mask = np.concatenate((mask, mask, mask), axis=-1)    
    combine = cv2.hconcat([picture, mask])

    cv2.imwrite('save/combine_'+ datetime.now().strftime('%Y%m%d%H%M%S%f')+'.jpg',combine)
    logger.info('combine save done.......')

    if head_or_tail == 'head':
        parse_head_pic(combine)
    else:
        parse_tail_pic(combine)



    # if DEBUG:
   
    
    
    # current_time = int(time.time() * 1000)
    
    # global last_time_head
    # global last_time_tail
    # print("=====current_time - last_time_head ", current_time - last_time_head , 'head or tail :', head_or_tail)
    # # if (current_time - last_time_head > 1000) and head_or_tail == 'head':
    # cv2.imwrite(os.path.join('/home/user/hrzg/data/1022_online/head/',datetime.now().strftime('%Y%m%d%H%M%S%f')+'.jpg'), combine)
    # last_time_head = current_time
    # # elif ((current_time - last_time_tail > 1000)  and head_or_tail == 'tail' ):
    # # cv2.imwrite(os.path.join('/home/user/hrzg/data/1022_online/tail/',datetime.now().strftime('%Y%m%d%H%M%S%f')+'.jpg'), combine)
    # # last_time_tail = current_time
    
    






    
    # _, buffer = cv2.imencode('.png', mask)
    # image_str = base64.b64encode(buffer).decode('utf-8')
    
    # data_map = {
    #     'image': image_str,
    #     'head_or_tail': head_or_tail
    # }
    # to_send_que.put(data_map)


    # combine = cv2.hconcat([picture, mask])
    # cv2.imwrite('tmp_mask.png', combine)

    
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





class RTSCapture(cv2.VideoCapture):
    """Real Time Streaming Capture.
    这个类必须使用 RTSCapture.create 方法创建，请不要直接实例化
    """

    _cur_frame = None
    _reading = False
    schemes = ["rtsp://", "rtmp://"] #用于识别实时流
    head_or_tail = 'head'
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

        if url.find('.102') > 0:
            rtscap.head_or_tail = 'tail'
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



    def fetch_rtsp_loop(self):
        self.start_read()
        while 1:
            flag , frame = self.read2()
            if flag:
                # print(datetime.now().strftime('%Y%m%d%H%M%S%f'), frame.shape)
                # cv2.imwrite("save/vc" + datetime.now().strftime('%Y%m%d%H%M%S%f')+'.jpg', frame)
                logger.info(self.head_or_tail + ' predict start')
                predict(frame, self.head_or_tail)
                
                
                logger.info(self.head_or_tail + ' predict done')
            #time.sleep(2)    



class Camera:

        def __init__(self, url):
            self.init = False
            self.url = url
            self.head_or_tail = 'head'
            self.args = {
                "rtsp_transport": "tcp",
                "fflags": "nobuffer",
                "flags": "low_delay",
                "loglevel": "quiet"
            }    # 添加参数

            if self.url.find('.102') > 0:
                self.head_or_tail = 'tail'
            
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
                    
                    time.sleep(1)

                    self.process1 = (ffmpeg.input(self.url, **self.args).output('pipe:', format='rawvideo', pix_fmt='rgb24')
                        .overwrite_output().run_async(pipe_stdout=True))
                    
                    in_bytes = self.process1.stdout.read(self.width * self.height * 3)     # 读取图片
                    if not in_bytes:
                        logger.warning('self.process1.stdout.read is none...')
                    else:
                        self.init = True

                    continue
                
                # self.process1 = (ffmpeg.input(self.url, **self.args).output('pipe:', format='rawvideo', pix_fmt='rgb24')
                #         .overwrite_output().run_async(pipe_stdout=True))


                in_bytes = self.process1.stdout.read(self.width * self.height * 3)     # 读取图片
                if not in_bytes:
                    logger.warning('self.process1.stdout.read is none...')
                    time.sleep(1)
                    self.init = False
                    continue
                # 转成ndarray
                in_frame = (np.frombuffer(in_bytes, np.uint8).reshape([self.height, self.width, 3]))
                frame = cv2.cvtColor(in_frame, cv2.COLOR_RGB2BGR)  # 转成BGR
                


                if self.head_or_tail  == 'head':
                    time.sleep(1)
                    continue
                


                cv2.imwrite('save/frame_'+ datetime.now().strftime('%Y%m%d%H%M%S%f')+'.jpg',frame)
                logger.info('frame save done.......')
                predict(frame, self.head_or_tail)
                
                
                logger.info(self.head_or_tail + ' predict done')
                    
                
                
            self.process1.kill()             


    

if __name__ == '__main__':
    
   
    logger.info('project start....')
    

    model_tou = init_model(config_file_head, checkpoint_file_head)
    model_wei = init_model(config_file_tail, checkpoint_file_tail)



    logger.info('init model done....')

    
   # t1 = threading.Thread(target=Camera(rtsp_tou).display)
    # t2 = threading.Thread(target=Camera(rtsp_wei).display)


    rtscap_tail = RTSCapture.create(rtsp_wei)
    t2 = threading.Thread(target=rtscap_tail.fetch_rtsp_loop)


    logger.info('rtsp_tou wei thread done....')



    #t1.start()
    t2.start()

    
    # start_server = websockets.serve(post_to_frontend,'127.0.0.1',6666) 
    # asyncio.get_event_loop().run_until_complete(start_server)
    # logger.info('websockets server done...')
    # asyncio.get_event_loop().run_forever()

    #t1.join()
    t2.join()





