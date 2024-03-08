import os
import argparse
from PIL import Image
import numpy as np

"""
对 Middlebury 60对训练数据进行处理
受到原始数据集的影响 这个脚本最好根据实际情况进行修改
"""

def parser_setting():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../../datasets", type=str)
    parser.add_argument("--dataset_name", default="Middlebury", type=str)
    parser.add_argument("--data_type", default="Train", type=str)

    parser.add_argument("--scale", default=2, type=int) # 设置放缩尺寸
    parser.add_argument("--patchid_start_from_0", default=-1, type=int) # 设置patch id 是否从0开始
    parser.add_argument("--file_num", default=4, type=int) # 横向长度

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
    data_dir_1 = os.path.join(args.data_dir, args.dataset_name + "_train_im0_im1")
    data_dir_2 = os.path.join(args.data_dir, args.dataset_name + "_train_view1_view5")
    # 保存到训练数据文件夹
    output_dir = os.path.join(args.data_dir, 'train_data')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    all_items_1 = os.listdir(data_dir_1)
    folders_1 = [item for item in all_items_1 if os.path.isdir(os.path.join(data_dir_1, item))]

    all_items_2 = os.listdir(data_dir_2)
    folders_2 = [item for item in all_items_2 if os.path.isdir(os.path.join(data_dir_2, item))]

    imgs_list = []
    
    for folder in folders_1:
        l_data_dir = os.path.join(data_dir_1, folder, 'im0.png')
        r_data_dir = os.path.join(data_dir_1, folder, 'im1.png')
        imgs_list.append(l_data_dir)
        imgs_list.append(r_data_dir)

    for folder in folders_2:
        l_data_dir = os.path.join(data_dir_2, folder, 'view1.png')
        r_data_dir = os.path.join(data_dir_2, folder, 'view5.png')
        imgs_list.append(l_data_dir)
        imgs_list.append(r_data_dir)

    imgs_list = sorted(imgs_list)
    
    if args.patchid_start_from_0 >= 0:
        idx_patch = args.patchid_start_from_0

    else:
        all_items = os.listdir(os.path.join(output_dir, f'patches_x{args.scale}'))
        idx_patch = len(all_items) + 50000

    scale = args.scale
    patch_width = args.patch_width
    patch_height = args.patch_height 
    patch_stride = args.patch_stride

    for i in range(0, len(imgs_list), 2):
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

                patch_dir = os.path.join(output_dir, f'patches_x{scale}_{args.file_num}/{idx_patch:06d}')
                os.makedirs(patch_dir, exist_ok=True)
                Image.fromarray(np.uint8(hr_patch_0)).save(f'{patch_dir}/hr0.png')
                Image.fromarray(np.uint8(hr_patch_1)).save(f'{patch_dir}/hr1.png')
                Image.fromarray(np.uint8(lr_patch_0)).save(f'{patch_dir}/lr0.png')
                Image.fromarray(np.uint8(lr_patch_1)).save(f'{patch_dir}/lr1.png')
                print(f'{idx_patch:06d} training samples have been generated...')
                idx_patch += 1

if __name__=="__main__":
    main()
    