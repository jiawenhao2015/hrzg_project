
config_file = '/home/user/hrzg/work/mmsegmentation/configs/ddrnet/ddrnet_23_in1k-pre_2xb6-120k_cityscapes-1024x1024-hrzg.py'

checkpoint_file_wei = '/home/user/hrzg/work/mmsegmentation/work_dir/0924_ddrnet_v_0/iter_72000.pth'
checkpoint_file_tou = '/home/user/hrzg/work/mmsegmentation/work_dir/0924_ddrnet_v_0/iter_72000.pth'

dir_save_pseudo_masks = '/home/user/hrzg/work/mmsegmentation/work_dir/0924_ddrnet_v_0/test_show_72000'


rtsp_tou = 'rtsp://admin:hrzg2024@192.168.66.101:554/Streaming/Channels/101'
rtsp_wei = 'rtsp://admin:rb123456@192.168.66.102:554/Streaming/Channels/101'



rtsp_head_result = 'rtsp://127.0.0.1:8554/head_result' #这里改成本地ip，端口号不变，文件夹自定义
rtsp_tail_resuslt = 'rtsp://127.0.0.1:8554/tail_result' #这里改成本地ip，端口号不变，文件夹自定义



command_head= [
    'ffmpeg',
    # 're',#
    # '-y', # 无需询问即可覆盖输出文件
    '-f', 'rawvideo', # 强制输入或输出文件格式
    '-vcodec','rawvideo', # 设置视频编解码器。这是-codec:v的别名
    '-pix_fmt', 'bgr24', # 设置像素格式
    '-s', '1280x720', # 设置图像大小
    '-r', 25, # 设置帧率
    '-i', '-', # 输入
    '-c:v', 'libx264',
    '-pix_fmt', 'yuv420p',
    '-preset', 'ultrafast',
    '-f', 'rtsp',# 强制输入或输出文件格式
    rtsp_head_result]


command_tail= [
    'ffmpeg',
    # 're',#
    # '-y', # 无需询问即可覆盖输出文件
    '-f', 'rawvideo', # 强制输入或输出文件格式
    '-vcodec','rawvideo', # 设置视频编解码器。这是-codec:v的别名
    '-pix_fmt', 'bgr24', # 设置像素格式
    '-s', '1280x720', # 设置图像大小
    '-r', 25, # 设置帧率
    '-i', '-', # 输入
    '-c:v', 'libx264',
    '-pix_fmt', 'yuv420p',
    '-preset', 'ultrafast',
    '-f', 'rtsp',# 强制输入或输出文件格式
    rtsp_tail_resuslt]






enable_control = False

plc_ip = '192.168.0.20'
plc_port = 502
#local_ip = '192.168.0.100' 主机ip固定为 100 才被允许连接plc


roller_forward_code = 0
roller_backward_code = 1
roller_stop_code = 2

roller_forward_address = "M666.0"
roller_backward_address = "M666.1"
roller_stop_address = "M666.2"

#尾部相机画面锯切位置标定 a(x,y) b(x,y)  w,h
wei_a = (1,420)
wei_b = (309,110)
