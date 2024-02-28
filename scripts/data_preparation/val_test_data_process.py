import os
import argparse
from PIL import Image
import numpy as np



def parser_setting():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="..\..\datasets", type=str)
    parser.add_argument("--dataset_name", default="Flickr1024", type=str)
    parser.add_argument("--data_type", default="Validation",choices=["Validation","Test"], type=str)
    parser.add_argument("--scale", default=2, type=int) # 设置放缩尺寸

    return parser.parse_args()



# 裁剪图片 使其能够被放缩因子整除
def modcrop(image, scale):
    size = np.array(image.size)
    size = size - size % scale
    image = image.crop((0, 0, size[0], size[1]))
    return image

# 通过Bicubic获得低分辨率图像
def process_images(img_path_0, scale):
    img_0 = Image.open(img_path_0)
    img_hr_0 = modcrop(img_0, scale) 
    img_lr_0 = img_hr_0.resize((img_hr_0.size[0] // scale, img_hr_0.size[1] // scale), Image.BICUBIC)
    img_hr_0 = np.array(img_hr_0)
    img_lr_0 = np.array(img_lr_0)
    return img_hr_0, img_lr_0



def main():
    args = parser_setting()
    data_dir = os.path.join(args.data_dir, args.dataset_name)
    data_dir = os.path.join(data_dir, args.data_type)

    # 保存到训练数据文件夹
    output_dir = os.path.join(args.data_dir, 'val_data' if args.data_type == 'Validation' else 'test_data')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    imgs_list = []
    img_extensions = ["png",]
    for fname in os.listdir(data_dir):
        if any(fname.endswith(ext) for ext in img_extensions):
            imgs_list.append(os.path.join(data_dir, fname))

    imgs_list = sorted(imgs_list)
    scale = args.scale

    hr_dir = os.path.join(output_dir, 'hr')
    lr_dir = os.path.join(output_dir, f'lr_x{scale}')
    os.makedirs(hr_dir, exist_ok=True)
    os.makedirs(lr_dir, exist_ok=True)


    for i in range(0,len(imgs_list)):
        img_hr_0, img_lr_0 = process_images(imgs_list[i], scale = scale)
        img_name = os.path.basename(imgs_list[i])
        img_name = os.path.splitext(img_name)[0]

        hr_img_dir = os.path.join(hr_dir, f'{img_name}.png')
        lr_img_dir = os.path.join(lr_dir, f'{img_name}.png')

        
        Image.fromarray(np.uint8(img_hr_0)).save(hr_img_dir)
        Image.fromarray(np.uint8(img_lr_0)).save(lr_img_dir)
        print(f'{i} {args.data_type} samples have been generated...')




    # for i in range(0, len(imgs_list), 2):
    #     img_hr_0, img_hr_1, img_lr_0, img_lr_1 = process_images(imgs_list[i], imgs_list[i+1], scale = scale)
        
    #     # 以步数 patch_stride 取出lr对应的小块图像 每块都是 patch_height x patch_width
    #     start_index_x, start_index_y = args.patch_start_height, args.patch_start_width
    #     end_index_x = (img_lr_0.shape[0] - (patch_height + start_index_x))
    #     end_index_y = (img_lr_0.shape[1] - (patch_width + start_index_y))

    #     for x_lr in range(start_index_x, end_index_x, patch_stride):
    #         for y_lr in range(start_index_y, end_index_y, patch_stride):
    #             x_hr = (x_lr - 1) * scale + 1
    #             y_hr = (y_lr - 1) * scale + 1
    #             hr_patch_0 = img_hr_0[x_hr - 1:(x_lr + patch_height - 1) * scale, y_hr - 1:(y_lr + patch_width - 1) * scale, :]
    #             hr_patch_1 = img_hr_1[x_hr - 1:(x_lr + patch_height - 1) * scale, y_hr - 1:(y_lr + patch_width - 1) * scale, :]
    #             lr_patch_0 = img_lr_0[x_lr - 1:x_lr + patch_height - 1, y_lr - 1:y_lr + patch_width - 1, :]
    #             lr_patch_1 = img_lr_1[x_lr - 1:x_lr + patch_height - 1, y_lr - 1:y_lr + patch_width - 1, :]

    #             patch_dir = os.path.join(train_dir, f'patches_x{scale}/{idx_patch:06d}')
    #             os.makedirs(patch_dir, exist_ok=True)
    #             Image.fromarray(np.uint8(hr_patch_0)).save(f'{patch_dir}/hr0.png')
    #             Image.fromarray(np.uint8(hr_patch_1)).save(f'{patch_dir}/hr1.png')
    #             Image.fromarray(np.uint8(lr_patch_0)).save(f'{patch_dir}/lr0.png')
    #             Image.fromarray(np.uint8(lr_patch_1)).save(f'{patch_dir}/lr1.png')
    #             print(f'{idx_patch:06d} training samples have been generated...')
    #             idx_patch += 1

if __name__=="__main__":
    main()