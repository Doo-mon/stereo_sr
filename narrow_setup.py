import os
import sys
import time
import argparse

# 输入命令
# python narrow_setup_new.py --total_gpu 6 --need_gpu 1 --is_only_test False --port 29400 --is_test False --e_block base --f_block scam --size t --x 2 --interval 2


# 可以直接在这里改，但是命令行参数优先级更高
port = 29400 # 每次执行注意要改成不同的端口号

is_test = True # True False # 是否进行测试
is_only_test = False # True False # 是否进行训练 这个优先级更高

e_block = "base"  # base modem hab
f_block = "scam"  # scam skscam mdia rcsb

size =  "t"  # t s b
x = 2 # 2 4

total_gpu = 4 # None
need_gpu = 1 # None
interval = 2 # None

suffix = None  # 后缀(暂时没用)

create_yaml = True
total_iter = 200000
batch_size_per_gpu = 8


def parse_setting():
    parser = argparse.ArgumentParser(description='narrow setup')

    parser.add_argument("--total_gpu", type=int, help="number of gpu")
    parser.add_argument("--need_gpu", type=int, help="number of gpu")
    parser.add_argument("--interval", type=int, help="interval time for checking gpu status")
    parser.add_argument("--port", type=int, help="port number")
    parser.add_argument("--is_test", type=bool, help="is test")
    parser.add_argument("--is_only_test", type=bool, help="is_only_test")

    parser.add_argument("--e_block", type=str, help="e_block")
    parser.add_argument("--f_block", type=str, help="f_block")
    parser.add_argument("--size", type=str, help="size")
    parser.add_argument("--x", type=int, help="x")
    
    parser.add_argument("--create_yaml", type=bool, help="create_yaml")
    

    return parser.parse_args()

# 获取某个GPU的信息
def gpu_info(gpu_index=0):
    gpu_status = os.popen('nvidia-smi | grep %').read().split('\n')[gpu_index].split('|')
    gpu_memory = int(gpu_status[2].split('/')[0].split('M')[0].strip())
    gpu_power = int(gpu_status[1].split('   ')[-1].split('/')[0].split('W')[0].strip())
    return gpu_power, gpu_memory

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

    if args.total_gpu is not None:
        total_gpu = args.total_gpu
    if args.need_gpu is not None:
        need_gpu = args.need_gpu
    if args.interval is not None:
        interval = args.interval
    if args.port is not None:
        port = args.port
    if args.is_test is not None:
        is_test = args.is_test
    if args.e_block is not None:
        e_block = args.e_block
    if args.f_block is not None:
        f_block = args.f_block
    if args.size is not None:
        size = args.size
    if args.x is not None:
        x = args.x
    if args.is_only_test is not None:
        is_only_test = args.is_only_test
    if args.create_yaml is not None:
        create_yaml = args.create_yaml


    if x == 2:
        name = f"{e_block}_{f_block}_{size}_x2"
    else:
        name = f"{e_block}_{f_block}_{size}"

    if suffix is not None:
        file_name = f"{name}_{suffix}.yml"
    else:
        file_name = f"{name}.yml"

    if create_yaml:
        create_cmd = f"python ~/stereo_sr/write_yaml.py --name {name} --train_num_gpu {need_gpu} \
            --total_iter {total_iter} --batch_size_per_gpu {batch_size_per_gpu}"
        print('\n' + create_cmd)
        os.system(create_cmd)

    
    cmd_train = f"python -m torch.distributed.launch --nproc_per_node={need_gpu} --master_port={port}  \
            ~/stereo_sr/train.py \
            -opt ./options/x{str(x)}/train_{file_name}"
    cmd_test = f"python -m torch.distributed.launch --nproc_per_node=1 --master_port={port + 1}  \
                ~/stereo_sr/test.py \
                -opt ./options/x{str(x)}/test_{file_name}"
   

    if is_only_test:
        narrow_setup_multi_gpu(command = cmd_test, interval = interval, total_gpu = total_gpu, need_gpu = 1)

    else:
        narrow_setup_multi_gpu(command = cmd_train, interval = interval, total_gpu = total_gpu, need_gpu = need_gpu)
        if is_test:
            narrow_setup_multi_gpu(command = cmd_test, interval = interval, total_gpu = total_gpu, need_gpu = 1)