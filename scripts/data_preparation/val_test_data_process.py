import os
import argparse
from PIL import Image
import numpy as np


def parser_setting():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../../datasets", type=str)
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
    data_dir = os.path.join(data_dir, args.data_type) # Flickr1024/Validation or Flickr1024/Test

    # 保存文件夹路径
    if args.data_type == 'Test':
        output_dir = os.path.join(args.data_dir, 'test_data')
        output_dir = os.path.join(output_dir, args.dataset_name) # 对于测试集 需要多保存一个数据集名字的文件夹
        
    else:
        output_dir = os.path.join(args.data_dir, 'val_data')

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


    for i in range(0,len(imgs_list),2):
        img_hr_0, img_lr_0 = process_images(imgs_list[i], scale = scale)
        img_hr_1, img_lr_1 = process_images(imgs_list[i+1], scale = scale)

        img_name = os.path.basename(imgs_list[i])
        img_name = os.path.splitext(img_name)[0]

        hr_img_dir = os.path.join(hr_dir, f'{img_name}')
        lr_img_dir = os.path.join(lr_dir, f'{img_name}')

        
        Image.fromarray(np.uint8(img_hr_0)).save(os.path.join(hr_img_dir, 'hr_0.png'))
        Image.fromarray(np.uint8(img_hr_1)).save(os.path.join(hr_img_dir, 'hr_1.png'))
        Image.fromarray(np.uint8(img_lr_0)).save(os.path.join(lr_img_dir, 'lr_0.png'))
        Image.fromarray(np.uint8(img_lr_1)).save(os.path.join(lr_img_dir, 'lr_1.png'))
        print(f'{i} {args.data_type} samples have been generated...')


if __name__=="__main__":
    main()