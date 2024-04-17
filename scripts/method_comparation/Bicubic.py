import os
from PIL import Image
import numpy as np
from tqdm import tqdm


method_name = "bicubic"


dataset_dir = "../../datasets/test_data"
output_dir_x4 = f"../../results/{method_name}/visualization"
output_dir_x2 = f"../../results/{method_name}_x2/visualization"

dataset_names=  ["Flickr1024", "KITTI2012", "KITTI2015", "Middlebury_test", "Middlebury2021"]

def main():

    if not os.path.exists(output_dir_x4):
        os.makedirs(output_dir_x4, exist_ok=True)
    if not os.path.exists(output_dir_x2):
        os.makedirs(output_dir_x2, exist_ok=True)

    for dataset_name in dataset_names:
        data_dir_x4 = os.path.join(dataset_dir, dataset_name, "lr_x4")
        data_dir_x2 = os.path.join(dataset_dir, dataset_name, "lr_x2")

        all_items = os.listdir(data_dir_x4) # 下面是 lr0.png 和 lr1.png

        for item in tqdm(all_items):
            L_img_path_x4 = os.path.join(data_dir_x4, item, "lr0.png")
            R_img_path_x4 = os.path.join(data_dir_x4, item, "lr1.png")
            L_img_path_x2 = os.path.join(data_dir_x2, item, "lr0.png")
            R_img_path_x2 = os.path.join(data_dir_x2, item, "lr1.png")

            L_img_x4 = Image.open(L_img_path_x4)
            R_img_x4 = Image.open(R_img_path_x4)
            L_img_x2 = Image.open(L_img_path_x2)
            R_img_x2 = Image.open(R_img_path_x2)

            L_img_x4 = L_img_x4.resize((L_img_x4.size[0] * 4, L_img_x4.size[1] * 4), Image.BICUBIC)
            R_img_x4 = R_img_x4.resize((R_img_x4.size[0] * 4, R_img_x4.size[1] * 4), Image.BICUBIC)
            L_img_x2 = L_img_x2.resize((L_img_x2.size[0] * 2, L_img_x2.size[1] * 2), Image.BICUBIC)
            R_img_x2 = R_img_x2.resize((R_img_x2.size[0] * 2, R_img_x2.size[1] * 2), Image.BICUBIC)

            L_img_x4.save(os.path.join(output_dir_x4, dataset_name, item + "_L.png"))
            R_img_x4.save(os.path.join(output_dir_x4, dataset_name, item + "_R.png"))
            L_img_x2.save(os.path.join(output_dir_x2, dataset_name, item + "_L.png"))
            R_img_x2.save(os.path.join(output_dir_x2, dataset_name, item + "_R.png"))

if __name__=="__main__":
    main()
    