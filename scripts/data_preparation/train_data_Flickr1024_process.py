import os
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm

"""

仿照 iPASSR 对Flickr1024数据集的 训练集图像 进行分patch处理 (并通过bicubic插值得到低分辨率图像)
原matlab代码: https://github.com/YingqianWang/iPASSR/blob/main/data/train/GenerateTrainingPatches.m

注 这里只对flickr1024进行了处理 如果需要扩充数据集 可以添加对其他数据集进行类似处理

!!!!! NAFSSR 训练数据中 包含了800对Flickr1024的训练数据 和 60 对 Middlebury 数据集 !!!!!

"""

def parser_setting():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../../datasets", type=str)
    parser.add_argument("--dataset_name", default="Flickr1024", type=str)
    parser.add_argument("--data_type", default="Train", type=str)

    parser.add_argument("--scale", default=2, type=int) # 设置放缩尺寸
    parser.add_argument("--patchid_start_from_0", default=-1, type=int) # 设置patch id 是否从0开始
    parser.add_argument("--start_file_num", default=0, type=int) # 设置保存的文件夹 默认从0开始

    # 设置patch的大小和步长
    parser.add_argument("--patch_width", default=90, type=int) # 横向长度
    parser.add_argument("--patch_height", default=30, type=int) # 纵向长度
    parser.add_argument("--patch_stride", default=20, type=int) # 取patch的步长
    # 开始取patch的点 这个会忽视掉图片左边和上边的一小部分
    parser.add_argument("--patch_start_height", default=3, type=int) 
    parser.add_argument("--patch_start_width", default=3, type=int)
    
    return parser.parse_args()

# 裁剪图片 使其能够被放缩因子整除
def modcrop(image, scale):
    size = np.array(image.size)
    size = size - size % scale
    image = image.crop((0, 0, size[0], size[1]))
    return image

# 通过Bicubic获得低分辨率图像
def process_images(img_path_0, img_path_1, scale):
    img_0 = Image.open(img_path_0)
    img_1 = Image.open(img_path_1)

    img_hr_0 = modcrop(img_0, scale)
    img_hr_1 = modcrop(img_1, scale)
    # print(img_hr_0.size)
    img_lr_0 = img_hr_0.resize((img_hr_0.size[0] // scale, img_hr_0.size[1] // scale), Image.BICUBIC)
    img_lr_1 = img_hr_1.resize((img_hr_1.size[0] // scale, img_hr_1.size[1] // scale), Image.BICUBIC)
    # print(img_lr_0.size)
    img_hr_0 = np.array(img_hr_0)
    img_hr_1 = np.array(img_hr_1)
    img_lr_0 = np.array(img_lr_0)
    img_lr_1 = np.array(img_lr_1)

    return img_hr_0, img_hr_1, img_lr_0, img_lr_1


def main():
    args = parser_setting()
    data_dir = os.path.join(args.data_dir, args.dataset_name) # data_dir = ../../datasets/Flickr1024
    data_dir = os.path.join(data_dir, args.data_type) # data_dir = ../../datasets/Flickr1024/Train

    # 输出保存路径
    output_dir = os.path.join(args.data_dir, 'train_data') # output_dir = ../../datasets/train_data
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 获取图片路径列表
    imgs_list = []
    img_extensions = ["png",]
    for fname in os.listdir(data_dir):
        if any(fname.endswith(ext) for ext in img_extensions):
            imgs_list.append(os.path.join(data_dir, fname))
    imgs_list = sorted(imgs_list)

    # 设置保存的patch id
    if args.patchid_start_from_0 >= 0: # 手动设置
        idx_patch = args.patchid_start_from_0

    else: # 自动设置
        temp_dir = os.path.join(output_dir, f'patches_x{args.scale}_{args.start_file_num}')
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)
        all_items = os.listdir(temp_dir) 
        idx_patch = len(all_items) + 50000 * args.start_file_num

    scale = args.scale
    patch_width = args.patch_width
    patch_height = args.patch_height 
    patch_stride = args.patch_stride

    start_id = idx_patch
    print(f"Generating Flickr1024 train data x{args.scale} ...")
    for i in tqdm(range(0, len(imgs_list), 2)):
        img_hr_0, img_hr_1, img_lr_0, img_lr_1 = process_images(imgs_list[i], imgs_list[i+1], scale = scale)
        
        # 以步数 patch_stride 取出lr对应的小块图像 每块都是 patch_height x patch_width
        start_index_x, start_index_y = args.patch_start_height, args.patch_start_width
        end_index_x = (img_lr_0.shape[0] - (patch_height + start_index_x))
        end_index_y = (img_lr_0.shape[1] - (patch_width + start_index_y))

        for x_lr in range(start_index_x, end_index_x, patch_stride):
            for y_lr in range(start_index_y, end_index_y, patch_stride):
                x_hr = (x_lr - 1) * scale + 1
                y_hr = (y_lr - 1) * scale + 1
                hr_patch_0 = img_hr_0[x_hr - 1:(x_lr + patch_height - 1) * scale, y_hr - 1:(y_lr + patch_width - 1) * scale, :]
                hr_patch_1 = img_hr_1[x_hr - 1:(x_lr + patch_height - 1) * scale, y_hr - 1:(y_lr + patch_width - 1) * scale, :]
                lr_patch_0 = img_lr_0[x_lr - 1:x_lr + patch_height - 1, y_lr - 1:y_lr + patch_width - 1, :]
                lr_patch_1 = img_lr_1[x_lr - 1:x_lr + patch_height - 1, y_lr - 1:y_lr + patch_width - 1, :]
                
                patch_dir = os.path.join(output_dir, f'patches_x{scale}_{int(idx_patch/50000)}/{idx_patch:06d}')
                os.makedirs(patch_dir, exist_ok=True)
                Image.fromarray(np.uint8(hr_patch_0)).save(f'{patch_dir}/hr0.png')
                Image.fromarray(np.uint8(hr_patch_1)).save(f'{patch_dir}/hr1.png')
                Image.fromarray(np.uint8(lr_patch_0)).save(f'{patch_dir}/lr0.png')
                Image.fromarray(np.uint8(lr_patch_1)).save(f'{patch_dir}/lr1.png')
                # print(f'{idx_patch:06d} training samples have been generated...')
                idx_patch += 1
    print(f"finished ... from id-{start_id} to id-{idx_patch}")


if __name__=="__main__":
    main()
    