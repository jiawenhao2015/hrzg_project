from utils import *
import argparse






def main():
    # 创建解析器
    parser = argparse.ArgumentParser(description="control roller")

    # 添加参数
    parser.add_argument("saw", type=int, help="第一个参数")

    parser.add_argument("direction", type=int, help="第一个参数")
    parser.add_argument("duration", type=float, help="第二个参数，整数类型")
    
    # 解析参数
    args = parser.parse_args()

    # 打印解析后的参数
    print("param1:", args.saw)
    print("param2:", args.direction)
    print("duration:", args.duration)

    control_roller(args.saw, args.direction, args.duration)


if __name__ == "__main__":
   
    #main()

    siemens.WriteBool(roller_before_saw_forward_address, 0)
    
    cnt = 0
    while 1:

        duration = 1.5
        control_roller(ROLLER_BEFORE_SAW, FORWARD_DIRECTION, duration)
        time.sleep(1.5)
        cnt  = cnt + 1
        print('control cnt:',cnt)
        if cnt > 100:
            break
        
    
    # time.sleep(1)
    # control_roller(ROLLER_BEFORE_SAW, FORWARD_DIRECTION, duration)
    #control_roller(ROLLER_BEFORE_SAW, BACKWARD_DIRECTION, duration)


#control_roller(ROLLER_AFTER_SAW, FORWARD_DIRECTION, duration)
#control_roller(ROLLER_AFTER_SAW, BACKWARD_DIRECTION, duration)

# write1 = siemens.WriteBool("M666.0",0)
# print("step1 write1:",write1.Message)


# read1 = siemens.ReadBool("M666.0")#
# print("step1 read1:",read1.Content)
