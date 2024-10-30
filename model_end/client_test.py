import asyncio
import websockets
import numpy as np
import cv2
import os
import json
import base64
from logconfig import *
async def hello(uri):
    async with websockets.connect(uri) as websocket:
        os.makedirs('./save',exist_ok=True)

        cnt = 0
        
        while True:
            response = await websocket.recv()
            data_map = json.loads(response)
            
            image_str = data_map['image']
            image_data = base64.b64decode(image_str)
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            cv2.imwrite('./save/'+data_map['head_or_tail'] + str(cnt)+'.jpg', image)
            print('已成功接收',data_map['head_or_tail'] + str(cnt)+'.jpg')
            cnt =  cnt + 1
            logger.info('done.1')
          

if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(hello('ws://127.0.0.1:6666')) # 改为你自己的地址

    print('another ...')