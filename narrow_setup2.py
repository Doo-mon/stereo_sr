import os
import sys
import time
import argparse


# train.py
cmd = "python -m torch.distributed.launch --nproc_per_node=1 --master_port=29443  \
       ~/stereo_sr/train.py \
       -opt ./options/train_hab_rcsb_t.yml"


# test.py
cmd1 = "python -m torch.distributed.launch --nproc_per_node=1 --master_port=29444  \
        ~/stereo_sr/test.py \
        -opt ./options/test_hab_rcsb_t.yml"




def parse_setting():
    parser = argparse.ArgumentParser(description='narrow setup')

    parser.add_argument("--total_gpu", default=4, type=int, help="number of gpu")
    parser.add_argument("--need_gpu", default=1, type=int, help="number of gpu")
    parser.add_argument("--interval", default=2, type=int, help="interval time for checking gpu status")

    return parser.parse_args()

# 获取某个GPU的信息
def gpu_info(gpu_index=0):
    gpu_status = os.popen('nvidia-smi | grep %').read().split('\n')[gpu_index].split('|')
    gpu_memory = int(gpu_status[2].split('/')[0].split('M')[0].strip())
    gpu_power = int(gpu_status[1].split('   ')[-1].split('/')[0].split('W')[0].strip())
    return gpu_power, gpu_memory

# 旧的函数 只能对一个GPU进行监控
def narrow_setup_old(command, interval=2):
    gpu_power, gpu_memory = gpu_info()
    i = 0
    while gpu_memory > 1000 or gpu_power > 20:  # set waiting condition
        gpu_power, gpu_memory = gpu_info()
        i = i % 5
        symbol = 'monitoring: ' + '>' * i + ' ' * (10 - i - 1) + '|'
        gpu_power_str = 'gpu power:%d W |' % gpu_power
        gpu_memory_str = 'gpu memory:%d MiB |' % gpu_memory
        sys.stdout.write('\r' + gpu_memory_str + ' ' + gpu_power_str + ' ' + symbol)
        sys.stdout.flush()
        time.sleep(interval)
        i += 1
    print('\n' + command)
    os.system(command)

# 对多个GPU进行循环监控
def narrow_setup_new(command, interval=2 , total_gpu = 8):
    i = 0
    selected_gpu = 0
    find_gpu = False
    while not find_gpu:
        i = i % 5
        for n in range(total_gpu):
            gpu_power, gpu_memory = gpu_info(gpu_index = n)
            if gpu_memory < 1000 and gpu_power < 20:
                selected_gpu = n
                find_gpu = True
                break
            symbol = 'monitoring: ' + 'GPU :'+ str(n) +' >' * i + ' ' * (10 - i - 1) + '|'
            gpu_power_str = 'gpu power:%d W |' % gpu_power
            gpu_memory_str = 'gpu memory:%d MiB |' % gpu_memory
            sys.stdout.write('\r' + gpu_memory_str + ' ' + gpu_power_str + ' ' + symbol)
            sys.stdout.flush()
        time.sleep(interval)
        i += 1
    command = "CUDA_VISIBLE_DEVICES=" + str(selected_gpu) + " " + command # 指定环境变量
    print('\n' + command)
    os.system(command)


def narrow_setup_multi_gpu(command, interval = 2, total_gpu = 8, need_gpu = 8):
    i = 0
    selected_gpu = []
    mark_list =  [0] * total_gpu
    count = 0
    while count < need_gpu:
        i = i % 5
        for n in range(total_gpu): # 循环获取所有gpu的信息再进行判断
            gpu_power, gpu_memory = gpu_info(gpu_index = n)
            if gpu_memory < 1000 and gpu_power < 25 and mark_list[n] == 0:
                    selected_gpu.append(n)
                    mark_list[n] = 1
                    count += 1    
            elif mark_list[n] == 1: # 不符合 => 重新检查存储表 这个时候一般代表卡又被占了
                    selected_gpu.remove(n)
                    mark_list[n] = 0
                    count -= 1
            symbol = 'monitoring: ' + 'GPU :'+ str(n) +' >' * i + ' ' * (10 - i - 1) + '|'
            gpu_power_str = 'gpu power:%d W |' % gpu_power
            gpu_memory_str = 'gpu memory:%d MiB |' % gpu_memory
            sys.stdout.write('\r' + gpu_memory_str + ' ' + gpu_power_str + ' ' + symbol)
            sys.stdout.flush()
        time.sleep(interval)
        i += 1

    cuda_cmd = "CUDA_VISIBLE_DEVICES="
    for i in range(need_gpu):
        cuda_cmd = cuda_cmd + str(selected_gpu[i])
        if i < need_gpu - 1:
            cuda_cmd = cuda_cmd + ','

    command = cuda_cmd + ' ' + command
    print('\n' + command)
    os.system(command)


if __name__ == '__main__':
    
    args = parse_setting()
    narrow_setup_multi_gpu(command = cmd, interval = args.interval, total_gpu = args.total_gpu, need_gpu = args.need_gpu)
    narrow_setup_multi_gpu(command = cmd1, interval = args.interval, total_gpu = args.total_gpu, need_gpu = args.need_gpu)
    # narrow_setup_new(command = cmd, interval = args.interval, total_gpu = args.total_gpu)