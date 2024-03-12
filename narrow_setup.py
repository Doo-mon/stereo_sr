import os
import sys
import time


# train.py
#cmd = "python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500  ~/stereo_sr/train.py"

# test.py
cmd = "python -m torch.distributed.launch --nproc_per_node=1 --master_port=29500  ~/stereo_sr/test.py --opt ./options/base_model_test_4x_T.yml"


# 多GPU版本
def gpu_info(gpu_index=0):
    gpu_status = os.popen('nvidia-smi | grep %').read().split('\n')[gpu_index].split('|')
    gpu_memory = int(gpu_status[2].split('/')[0].split('M')[0].strip())
    gpu_power = int(gpu_status[1].split('   ')[-1].split('/')[0].split('W')[0].strip())
    return gpu_power, gpu_memory


def narrow_setup_old(interval=2):
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
    print('\n' + cmd)
    os.system(cmd)


def narrow_setup_new(interval=2 , total_gpu = 6):
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
    cmd = "CUDA_VISIBLE_DEVICES=" + str(selected_gpu) + " " + cmd
    print('\n' + cmd)
    os.system(cmd)

if __name__ == '__main__':
    # narrow_setup_old()
    narrow_setup_new(interval = 2, total_gpu = 6)