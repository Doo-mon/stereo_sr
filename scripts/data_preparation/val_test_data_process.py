import os
import argparse
from PIL import Image
import numpy as np
import random
import sys

"""
    NAFSSR 测试数据中  包含了   112对 Flickr1024 测试集 
                               5 对 Middlebury 数据
                               20 对 KITTI2012 数据
                               20 对 KITTI2015 数据
"""


def parser_setting():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../../datasets", type=str)
    parser.add_argument("--dataset_name", default="Flickr1024",choices=["Flickr1024","KITTI2012","KITTI2015","Middlebury_test","Middlebury2021"], type=str)
    parser.add_argument("--data_type", default="Validation",choices=["Validation", "Test"], type=str)
    # parser.add_argument("--scale", default=2, type=int) # 设置放缩尺寸

    return parser.parse_args()


# 裁剪图片 使其能够被放缩因子整除
def modcrop(image, scale):
    size = np.array(image.size)
    size = size - size % scale
    image = image.crop((0, 0, size[0], size[1]))
    return image

def modcrop_x5(image, scale):
    size = np.array(image.size)
    size = size - size % (scale * 5)
    image = image.crop((0, 0, size[0], size[1]))
    return image

# 通过Bicubic获得低分辨率图像
def get_lr_image_by_bicubic(img_path_0, scale):
    img_0 = Image.open(img_path_0)
    img_hr_0 = modcrop(img_0, scale) 
    img_lr_0 = img_hr_0.resize((img_hr_0.size[0] // scale, img_hr_0.size[1] // scale), Image.BICUBIC)
    img_hr_0 = np.array(img_hr_0)
    img_lr_0 = np.array(img_lr_0)
    return img_hr_0, img_lr_0

def get_lr_image_by_bicubic_x5(img_path_0, scale):
    img_0 = Image.open(img_path_0)
    img_hr_0 = modcrop_x5(img_0, scale) 
    img_lr_0 = img_hr_0.resize((img_hr_0.size[0] // scale, img_hr_0.size[1] // scale), Image.BICUBIC)
    img_hr_0 = np.array(img_hr_0)
    img_lr_0 = np.array(img_lr_0)
    return img_hr_0, img_lr_0


def process_Flickr1024(data_dir, data_type, scale, output_dir):
    data_dir = os.path.join(data_dir, data_type) # Flickr1024/Validation or Flickr1024/Test
    
    imgs_list = []
    img_extensions = ["png",]
    for fname in os.listdir(data_dir):
        if any(fname.endswith(ext) for ext in img_extensions):
            imgs_list.append(os.path.join(data_dir, fname))
    imgs_list = sorted(imgs_list)

    hr_dir = os.path.join(output_dir, 'hr')
    lr_dir_x2 = os.path.join(output_dir, 'lr_x2')
    lr_dir_x4 = os.path.join(output_dir,'lr_x4')

    # 成对处理图像
    for i in range(0, len(imgs_list), 2):
        img_hr_0_x4, img_lr_0_x4 = get_lr_image_by_bicubic_x5(imgs_list[i], scale = 4)
        img_hr_1_x4, img_lr_1_x4 = get_lr_image_by_bicubic_x5(imgs_list[i + 1], scale = 4)

        # 这里直接用x4的crop之后的hr图像 避免出现问题
        hr0_size = np.array(img_hr_0_x4.size)
        hr1_size = np.array(img_hr_1_x4.size)
        img_lr_0_x2 = Image.fromarray(np.uint8(img_hr_0_x4)).resize((hr0_size[0]//2, hr0_size[1]//2), Image.BICUBIC)
        img_lr_1_x2 = Image.fromarray(np.uint8(img_hr_1_x4)).resize((hr1_size[0]//2, hr1_size[1]//2), Image.BICUBIC)

        img_name = os.path.basename(imgs_list[i])
        img_name = os.path.splitext(img_name)[0] # 0001_L
        img_name = img_name.split('_')[0] # 0001

        hr_img_dir = os.path.join(hr_dir, f'{img_name}')
        lr_img_dir_x4 = os.path.join(lr_dir_x4, f'{img_name}')
        lr_img_dir_x2 = os.path.join(lr_dir_x2, f'{img_name}')
        os.makedirs(hr_img_dir, exist_ok=True)
        os.makedirs(lr_img_dir_x4, exist_ok=True)
        os.makedirs(lr_img_dir_x2, exist_ok=True)

        Image.fromarray(np.uint8(img_hr_0_x4)).save(os.path.join(hr_img_dir, 'hr0.png'))
        Image.fromarray(np.uint8(img_hr_1_x4)).save(os.path.join(hr_img_dir, 'hr1.png'))
        
        Image.fromarray(np.uint8(img_lr_0_x4)).save(os.path.join(lr_img_dir_x4, 'lr0.png'))
        Image.fromarray(np.uint8(img_lr_1_x4)).save(os.path.join(lr_img_dir_x4, 'lr1.png'))

        Image.fromarray(np.uint8(img_lr_0_x2)).save(os.path.join(lr_img_dir_x2, 'lr0.png'))
        Image.fromarray(np.uint8(img_lr_1_x2)).save(os.path.join(lr_img_dir_x2, 'lr1.png'))
        print(f'{i}--Flickr1024--{data_type} samples have been generated...')


def process_KITTI2012(data_dir, data_type, scale, output_dir, choice_n = 20):
    if data_type == 'Validation':
        print("The KITTI2012 dataset is not used for validation, only for testing.")
        data_type = 'Test'

    data_dir = os.path.join(data_dir, 'testing') # KITTI2012/testing

    img_extensions = ["png",]

    l_data_dir = os.path.join(data_dir, 'colored_0') # 左图
    r_data_dir = os.path.join(data_dir, 'colored_1') # 右图
    l_imgs_list = [] 
    for fname in os.listdir(l_data_dir):
        if any(fname.endswith(ext) for ext in img_extensions):
            l_imgs_list.append(os.path.join(l_data_dir, fname))
    l_imgs_list = sorted(l_imgs_list)

    r_imgs_list = []
    for fname in os.listdir(r_data_dir):
        if any(fname.endswith(ext) for ext in img_extensions):
            r_imgs_list.append(os.path.join(r_data_dir, fname))
    r_imgs_list = sorted(r_imgs_list)


    hr_dir = os.path.join(output_dir, 'hr')
    lr_dir_x2 = os.path.join(output_dir, 'lr_x2')
    lr_dir_x4 = os.path.join(output_dir,'lr_x4')

    # 创建一个列表，包含0到len之间的所有偶数
    even_numbers = [i for i in range(len(l_imgs_list)) if i % 2 == 0]
    # 随机选择20个不同的偶数
    selected_numbers = random.sample(even_numbers, choice_n)

    for i in selected_numbers:
        # 能被4整除的一定能被2整除
        # img_hr_0_x2, img_lr_0_x2 = get_lr_image_by_bicubic_x5(l_imgs_list[i], scale = 2)
        # img_hr_1_x2, img_lr_1_x2 = get_lr_image_by_bicubic_x5(r_imgs_list[i], scale = 2)

        img_hr_0_x4, img_lr_0_x4 = get_lr_image_by_bicubic_x5(l_imgs_list[i], scale = 4)
        img_hr_1_x4, img_lr_1_x4 = get_lr_image_by_bicubic_x5(r_imgs_list[i], scale = 4)

        # 这里直接用x4的crop之后的hr图像 避免出现问题
        hr0_size = np.array(img_hr_0_x4.size)
        hr1_size = np.array(img_hr_1_x4.size)
        img_lr_0_x2 = Image.fromarray(np.uint8(img_hr_0_x4)).resize((hr0_size[0]//2, hr0_size[1]//2), Image.BICUBIC)
        img_lr_1_x2 = Image.fromarray(np.uint8(img_hr_1_x4)).resize((hr1_size[0]//2, hr1_size[1]//2), Image.BICUBIC)

        img_name = os.path.basename(l_imgs_list[i])
        img_name = os.path.splitext(img_name)[0] # 000000_10
        img_name = img_name.split('_')[0] # 000000

        hr_img_dir = os.path.join(hr_dir, f'{img_name}')
        lr_img_dir_x4 = os.path.join(lr_dir_x4, f'{img_name}')
        lr_img_dir_x2 = os.path.join(lr_dir_x2, f'{img_name}')
        os.makedirs(hr_img_dir, exist_ok=True)
        os.makedirs(lr_img_dir_x4, exist_ok=True)
        os.makedirs(lr_img_dir_x2, exist_ok=True)

        Image.fromarray(np.uint8(img_hr_0_x4)).save(os.path.join(hr_img_dir, 'hr0.png'))
        Image.fromarray(np.uint8(img_hr_1_x4)).save(os.path.join(hr_img_dir, 'hr1.png'))
        
        Image.fromarray(np.uint8(img_lr_0_x4)).save(os.path.join(lr_img_dir_x4, 'lr0.png'))
        Image.fromarray(np.uint8(img_lr_1_x4)).save(os.path.join(lr_img_dir_x4, 'lr1.png'))

        Image.fromarray(np.uint8(img_lr_0_x2)).save(os.path.join(lr_img_dir_x2, 'lr0.png'))
        Image.fromarray(np.uint8(img_lr_1_x2)).save(os.path.join(lr_img_dir_x2, 'lr1.png'))


        print(f'{i/2}--KITTI2012--{data_type} samples have been generated...')


def process_KITTI2015(data_dir, data_type, scale, output_dir, choice_n = 20):
    if data_type == 'Validation':
        print("The KITTI2015 dataset is not used for validation, only for testing.")
        data_type = 'Test'
    
    data_dir = os.path.join(data_dir, 'testing') # KITTI2015/testing

    img_extensions = ["png",]

    l_data_dir = os.path.join(data_dir, 'image_2') # 左图
    r_data_dir = os.path.join(data_dir, 'image_3') # 右图
    l_imgs_list = [] 
    for fname in os.listdir(l_data_dir):
        if any(fname.endswith(ext) for ext in img_extensions):
            l_imgs_list.append(os.path.join(l_data_dir, fname))
    l_imgs_list = sorted(l_imgs_list)

    r_imgs_list = []
    for fname in os.listdir(r_data_dir):
        if any(fname.endswith(ext) for ext in img_extensions):
            r_imgs_list.append(os.path.join(r_data_dir, fname))
    r_imgs_list = sorted(r_imgs_list)


    hr_dir = os.path.join(output_dir, 'hr')
    lr_dir_x2 = os.path.join(output_dir, 'lr_x2')
    lr_dir_x4 = os.path.join(output_dir,'lr_x4')

    # 创建一个列表，包含0到len之间的所有偶数
    even_numbers = [i for i in range(len(l_imgs_list)) if i % 2 == 0]
    # 随机选择20个不同的偶数
    selected_numbers = random.sample(even_numbers, choice_n)

    for i in selected_numbers:

        img_hr_0_x4, img_lr_0_x4 = get_lr_image_by_bicubic_x5(l_imgs_list[i], scale = 4)
        img_hr_1_x4, img_lr_1_x4 = get_lr_image_by_bicubic_x5(r_imgs_list[i], scale = 4)

        # 这里直接用x4的crop之后的hr图像 避免出现问题
        hr0_size = np.array(img_hr_0_x4.size)
        hr1_size = np.array(img_hr_1_x4.size)
        img_lr_0_x2 = Image.fromarray(np.uint8(img_hr_0_x4)).resize((hr0_size[0]//2, hr0_size[1]//2), Image.BICUBIC)
        img_lr_1_x2 = Image.fromarray(np.uint8(img_hr_1_x4)).resize((hr1_size[0]//2, hr1_size[1]//2), Image.BICUBIC)

        img_name = os.path.basename(l_imgs_list[i])
        img_name = os.path.splitext(img_name)[0] # 000000_10
        img_name = img_name.split('_')[0] # 000000

        hr_img_dir = os.path.join(hr_dir, f'{img_name}')
        lr_img_dir_x4 = os.path.join(lr_dir_x4, f'{img_name}')
        lr_img_dir_x2 = os.path.join(lr_dir_x2, f'{img_name}')
        os.makedirs(hr_img_dir, exist_ok=True)
        os.makedirs(lr_img_dir_x4, exist_ok=True)
        os.makedirs(lr_img_dir_x2, exist_ok=True)

        Image.fromarray(np.uint8(img_hr_0_x4)).save(os.path.join(hr_img_dir, 'hr0.png'))
        Image.fromarray(np.uint8(img_hr_1_x4)).save(os.path.join(hr_img_dir, 'hr1.png'))
        
        Image.fromarray(np.uint8(img_lr_0_x4)).save(os.path.join(lr_img_dir_x4, 'lr0.png'))
        Image.fromarray(np.uint8(img_lr_1_x4)).save(os.path.join(lr_img_dir_x4, 'lr1.png'))

        Image.fromarray(np.uint8(img_lr_0_x2)).save(os.path.join(lr_img_dir_x2, 'lr0.png'))
        Image.fromarray(np.uint8(img_lr_1_x2)).save(os.path.join(lr_img_dir_x2, 'lr1.png'))
        print(f'{i/2}--KITTI2015--{data_type} samples have been generated...')


def process_KITTI2012_old(data_dir, data_type, scale, output_dir):
    if data_type == 'Validation':
        print("The KITTI2012 dataset is not used for validation, only for testing.")
        data_type = 'Test'

    data_dir = os.path.join(data_dir, 'testing') # KITTI2012/testing

    img_extensions = ["png",]

    l_data_dir = os.path.join(data_dir, 'colored_0') # 左图
    r_data_dir = os.path.join(data_dir, 'colored_1') # 右图
    l_imgs_list = [] 
    for fname in os.listdir(l_data_dir):
        if any(fname.endswith(ext) for ext in img_extensions):
            l_imgs_list.append(os.path.join(l_data_dir, fname))
    l_imgs_list = sorted(l_imgs_list)

    r_imgs_list = []
    for fname in os.listdir(r_data_dir):
        if any(fname.endswith(ext) for ext in img_extensions):
            r_imgs_list.append(os.path.join(r_data_dir, fname))
    r_imgs_list = sorted(r_imgs_list)


    hr_dir = os.path.join(output_dir, 'hr')
    lr_dir = os.path.join(output_dir, f'lr_x{scale}')

    # 创建一个列表，包含0到389之间的所有偶数
    even_numbers = [i for i in range(len(l_imgs_list)) if i % 2 == 0]
    # 随机选择20个不同的偶数
    selected_numbers = random.sample(even_numbers, 20)

    for i in selected_numbers:
        img_hr_0, img_lr_0 = get_lr_image_by_bicubic(l_imgs_list[i], scale = scale)
        img_hr_1, img_lr_1 = get_lr_image_by_bicubic(r_imgs_list[i], scale = scale)

        img_name = os.path.basename(l_imgs_list[i])
        img_name = os.path.splitext(img_name)[0] # 000000_10
        img_name = img_name.split('_')[0] # 000000

        hr_img_dir = os.path.join(hr_dir, f'{img_name}')
        lr_img_dir = os.path.join(lr_dir, f'{img_name}')
        os.makedirs(hr_img_dir, exist_ok=True)
        os.makedirs(lr_img_dir, exist_ok=True)

        Image.fromarray(np.uint8(img_hr_0)).save(os.path.join(hr_img_dir, 'hr0.png'))
        Image.fromarray(np.uint8(img_hr_1)).save(os.path.join(hr_img_dir, 'hr1.png'))
        Image.fromarray(np.uint8(img_lr_0)).save(os.path.join(lr_img_dir, 'lr0.png'))
        Image.fromarray(np.uint8(img_lr_1)).save(os.path.join(lr_img_dir, 'lr1.png'))
        print(f'{i/2}--KITTI2012--{data_type} samples have been generated...')


def process_KITTI2015_old(data_dir, data_type, scale, output_dir):
    if data_type == 'Validation':
        print("The KITTI2015 dataset is not used for validation, only for testing.")
        data_type = 'Test'
    
    data_dir = os.path.join(data_dir, 'testing') # KITTI2015/testing

    img_extensions = ["png",]

    l_data_dir = os.path.join(data_dir, 'image_2') # 左图
    r_data_dir = os.path.join(data_dir, 'image_3') # 右图
    l_imgs_list = [] 
    for fname in os.listdir(l_data_dir):
        if any(fname.endswith(ext) for ext in img_extensions):
            l_imgs_list.append(os.path.join(l_data_dir, fname))
    l_imgs_list = sorted(l_imgs_list)

    r_imgs_list = []
    for fname in os.listdir(r_data_dir):
        if any(fname.endswith(ext) for ext in img_extensions):
            r_imgs_list.append(os.path.join(r_data_dir, fname))
    r_imgs_list = sorted(r_imgs_list)


    hr_dir = os.path.join(output_dir, 'hr')
    lr_dir = os.path.join(output_dir, f'lr_x{scale}')

    # 创建一个列表，包含0到len之间的所有偶数
    even_numbers = [i for i in range(len(l_imgs_list)) if i % 2 == 0]
    # 随机选择20个不同的偶数
    selected_numbers = random.sample(even_numbers, 20)

    for i in selected_numbers:
        img_hr_0, img_lr_0 = get_lr_image_by_bicubic(l_imgs_list[i], scale = scale)
        img_hr_1, img_lr_1 = get_lr_image_by_bicubic(r_imgs_list[i], scale = scale)

        img_name = os.path.basename(l_imgs_list[i])
        img_name = os.path.splitext(img_name)[0] # 000000_10
        img_name = img_name.split('_')[0] # 000000

        hr_img_dir = os.path.join(hr_dir, f'{img_name}')
        lr_img_dir = os.path.join(lr_dir, f'{img_name}')
        os.makedirs(hr_img_dir, exist_ok=True)
        os.makedirs(lr_img_dir, exist_ok=True)

        Image.fromarray(np.uint8(img_hr_0)).save(os.path.join(hr_img_dir, 'hr0.png'))
        Image.fromarray(np.uint8(img_hr_1)).save(os.path.join(hr_img_dir, 'hr1.png'))
        Image.fromarray(np.uint8(img_lr_0)).save(os.path.join(lr_img_dir, 'lr0.png'))
        Image.fromarray(np.uint8(img_lr_1)).save(os.path.join(lr_img_dir, 'lr1.png'))
        print(f'{i/2}--KITTI2015--{data_type} samples have been generated...')


def process_Middlebury(data_dir, data_type, scale, output_dir):
    if data_type == 'Validation':
        print("The Middlebury2014 dataset is not used for validation, only for testing.")
        data_type = 'Test'
    
    all_items = os.listdir(data_dir)
    folders = [item for item in all_items if os.path.isdir(os.path.join(data_dir, item))]

    hr_dir = os.path.join(output_dir, 'hr')
    lr_dir_x2 = os.path.join(output_dir, 'lr_x2')
    lr_dir_x4 = os.path.join(output_dir,'lr_x4')

    for folder in folders:
        l_data_dir = os.path.join(data_dir, folder, 'im0.png')
        r_data_dir = os.path.join(data_dir, folder, 'im1.png')
        img_hr_0_x4, img_lr_0_x4 = get_lr_image_by_bicubic_x5(l_data_dir, scale = 4)
        img_hr_1_x4, img_lr_1_x4 = get_lr_image_by_bicubic_x5(r_data_dir, scale = 4)


        # 这里直接用x4的crop之后的hr图像 避免出现问题
        hr0_size = np.array(img_hr_0_x4.size)
        hr1_size = np.array(img_hr_1_x4.size)
        img_lr_0_x2 = Image.fromarray(np.uint8(img_hr_0_x4)).resize((hr0_size[0]//2, hr0_size[1]//2), Image.BICUBIC)
        img_lr_1_x2 = Image.fromarray(np.uint8(img_hr_1_x4)).resize((hr1_size[0]//2, hr1_size[1]//2), Image.BICUBIC)


        img_name = folder

        hr_img_dir = os.path.join(hr_dir, f'{img_name}')
        lr_img_dir_x4 = os.path.join(lr_dir_x4, f'{img_name}')
        lr_img_dir_x2 = os.path.join(lr_dir_x2, f'{img_name}')

        os.makedirs(hr_img_dir, exist_ok=True)
        os.makedirs(lr_img_dir_x4, exist_ok=True)
        os.makedirs(lr_img_dir_x2, exist_ok=True)

        Image.fromarray(np.uint8(img_hr_0_x4)).save(os.path.join(hr_img_dir, 'hr0.png'))
        Image.fromarray(np.uint8(img_hr_1_x4)).save(os.path.join(hr_img_dir, 'hr1.png'))

        Image.fromarray(np.uint8(img_lr_0_x4)).save(os.path.join(lr_img_dir_x4, 'lr0.png'))
        Image.fromarray(np.uint8(img_lr_1_x4)).save(os.path.join(lr_img_dir_x4, 'lr1.png'))

        Image.fromarray(np.uint8(img_lr_0_x2)).save(os.path.join(lr_img_dir_x2, 'lr0.png'))
        Image.fromarray(np.uint8(img_lr_1_x2)).save(os.path.join(lr_img_dir_x2, 'lr1.png'))

        print(f'{folder}--Middlebury2014--{data_type} samples have been generated...')
 

def process_Middlebury2021(data_dir, data_type, scale, output_dir):
    if data_type == 'Validation':
        print("The Middlebury2021 dataset is not used for validation, only for testing.")
        data_type = 'Test'
    data_dir = os.path.join(data_dir, 'data') # Middlebury2021/data

    all_items = os.listdir(data_dir)
    folders = [item for item in all_items if os.path.isdir(os.path.join(data_dir, item))]

    hr_dir = os.path.join(output_dir, 'hr')
    lr_dir_x2 = os.path.join(output_dir, 'lr_x2')
    lr_dir_x4 = os.path.join(output_dir,'lr_x4')

    for folder in folders:
        l_data_dir = os.path.join(data_dir, folder, 'im0.png')
        r_data_dir = os.path.join(data_dir, folder, 'im1.png')
        img_hr_0_x4, img_lr_0_x4 = get_lr_image_by_bicubic_x5(l_data_dir, scale = 4)
        img_hr_1_x4, img_lr_1_x4 = get_lr_image_by_bicubic_x5(r_data_dir, scale = 4)

        # 这里直接用x4的crop之后的hr图像 避免出现问题
        hr0_size = np.array(img_hr_0_x4.size)
        hr1_size = np.array(img_hr_1_x4.size)
        img_lr_0_x2 = Image.fromarray(np.uint8(img_hr_0_x4)).resize((hr0_size[0]//2, hr0_size[1]//2), Image.BICUBIC)
        img_lr_1_x2 = Image.fromarray(np.uint8(img_hr_1_x4)).resize((hr1_size[0]//2, hr1_size[1]//2), Image.BICUBIC)

        img_name = folder.split('-')[0]

        hr_img_dir = os.path.join(hr_dir, f'{img_name}')
        lr_img_dir_x4 = os.path.join(lr_dir_x4, f'{img_name}')
        lr_img_dir_x2 = os.path.join(lr_dir_x2, f'{img_name}')

        os.makedirs(hr_img_dir, exist_ok=True)
        os.makedirs(lr_img_dir_x4, exist_ok=True)
        os.makedirs(lr_img_dir_x2, exist_ok=True)

        Image.fromarray(np.uint8(img_hr_0_x4)).save(os.path.join(hr_img_dir, 'hr0.png'))
        Image.fromarray(np.uint8(img_hr_1_x4)).save(os.path.join(hr_img_dir, 'hr1.png'))

        Image.fromarray(np.uint8(img_lr_0_x4)).save(os.path.join(lr_img_dir_x4, 'lr0.png'))
        Image.fromarray(np.uint8(img_lr_1_x4)).save(os.path.join(lr_img_dir_x4, 'lr1.png'))

        Image.fromarray(np.uint8(img_lr_0_x2)).save(os.path.join(lr_img_dir_x2, 'lr0.png'))
        Image.fromarray(np.uint8(img_lr_1_x2)).save(os.path.join(lr_img_dir_x2, 'lr1.png'))
        print(f'{folder}--Middlebury2021--{data_type} samples have been generated...')


def temp_process():
    name_list=["Flickr1024","KITTI2012","KITTI2015","Middlebury_test","Middlebury2021"]
    
    from os import path as osp
    
    for name in name_list:
        data_dir = "."
        data_dir = osp.join(data_dir,name)
        # print(data_dir)
        hr_dir = osp.join(data_dir,"hr")
        lr_dir = osp.join(data_dir,"lr_x2")


        all_items = os.listdir(hr_dir)
        folders_1 = [os.path.join(hr_dir, item) for item in all_items]
        folders_2 = [os.path.join(lr_dir, item) for item in all_items]

        folders_1 = sorted(folders_1)
        folders_2 = sorted(folders_2)

        for i in range(len(folders_1)):
            print(folders_1[i])
            hr0_img = Image.open(os.path.join(folders_1[i], "hr0.png"))
            hr1_img = Image.open(os.path.join(folders_1[i], "hr1.png"))

            hr0_size = np.array(hr0_img.size)
            hr1_size = np.array(hr1_img.size)
            
            lr0_img = hr0_img.resize((hr0_size[0]//2, hr0_size[1]//2), Image.BICUBIC)
            lr1_img = hr1_img.resize((hr1_size[0]//2, hr1_size[1]//2), Image.BICUBIC)

            lr0_img.save(os.path.join(folders_2[i], "lr0.png"))
            lr1_img.save(os.path.join(folders_2[i], "lr1.png"))


def main():
    args = parser_setting()
    # 保存文件夹路径
    val_output_dir = os.path.join(args.data_dir,'val_data')
    test_output_dir = os.path.join(args.data_dir,'test_data')

    if not os.path.exists(val_output_dir):
        os.makedirs(val_output_dir, exist_ok=True)
    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir, exist_ok=True)
    # 原始数据的路径
    data_dir = os.path.join(args.data_dir, args.dataset_name) # datasets/Flickr1024 or datasets/KITTI2012 ...

    # 测试数据需要进行数据集区分 验证数据不需要区分
    if args.data_type == 'Validation':
        output_dir = val_output_dir
    else:
        output_dir = os.path.join(test_output_dir, args.dataset_name)

    cur_module = sys.modules[__name__]
    process_func = getattr(cur_module, f'process_{args.dataset_name}', None)

    if process_func:
        process_func(data_dir = data_dir, data_type = args.data_type, output_dir = output_dir)
    else:
        print(f"Dataset {args.dataset_name} is not supported.")

if __name__=="__main__":
    main()