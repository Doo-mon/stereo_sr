import yaml
import os
import argparse

def parse_setting():
    parser = argparse.ArgumentParser(description='Write yaml')

    parser.add_argument("--name", type=str, help="name of the model")
    parser.add_argument("--train_num_gpu", default=1, type=int, help="number of gpu")
    parser.add_argument("--total_iter", default=200000, type=int, help="total iteration")
    parser.add_argument("--batch_size_per_gpu", default=8, type=int, help="batch size per gpu")


    parser.add_argument("--model_type", default="ImageRestorationModel", type=str, help="type of model")
    parser.add_argument("--network_g_type", default="newNAFSSR", type=str, help="type of network_g")

    return parser.parse_args()


if __name__=="__main__":

    name = "hab_scam_t_x2"
    train_num_gpu = 1
    model_type = "ImageRestorationModel"
    network_g_type = "newNAFSSR"
    total_iter = 200000
    batch_size_per_gpu = 8
    test_data_num = 5

    args = parse_setting()
    if args.name is not None:
        name = args.name
    if args.train_num_gpu is not None:
        train_num_gpu = args.train_num_gpu
    if args.total_iter is not None:
        total_iter = args.total_iter
    if args.batch_size_per_gpu is not None:
        batch_size_per_gpu = args.batch_size_per_gpu
    if args.model_type is not None:
        model_type = args.model_type
    if args.network_g_type is not None:
        network_g_type = args.network_g_type
    



    str_list = name.split("_") # 目前只考虑 base_mdia_t 和 base_mdia_t_x2 这两种形式
    if len(str_list) > 3 and str_list[3] == "x2":
        scale = 2
        file_num = 7
        train_data = "~/stereo_sr/datasets/train_data/patches_x2_0/"
        val_data_gt = "~/stereo_sr/datasets/val_data/hr/"
        val_data_lq = "~/stereo_sr/datasets/val_data/lr_x2/"
        gt_size_h = 60
        gt_size_w = 180

    else:
        scale = 4
        file_num = 2
        train_data = "~/stereo_sr/datasets/train_data/patches_x4_0/"
        val_data_gt = "~/stereo_sr/datasets/val_data/hr/"
        val_data_lq = "~/stereo_sr/datasets/val_data/lr_x4/"
        gt_size_h = 120
        gt_size_w = 360
    
    if str_list[0] == "base":
        e_block = "NAFBlock"
    else:
        e_block = str_list[0].upper()

    f_block = str_list[1].upper()

    if str_list[2] == "t":
        width = 48
        num_blks = 16
        drop_path_rate = 0.
        drop_out_rate = 0.
    elif str_list[2] == "s":
        width = 64
        num_blks = 32
        drop_path_rate = 0.1
        drop_out_rate = 0.  
    elif str_list[2] == "b":
        width = 96
        num_blks = 64
        drop_path_rate = 0.2
        drop_out_rate = 0.
    elif str_list[2] == "l":
        width = 128
        num_blks = 128
        drop_path_rate = 0.3
        drop_out_rate = 0.



    new_train_yaml = {}
    with open("./options/train_template.yml", "r") as f:
        train_template = yaml.safe_load(f)

        new_train_yaml = train_template

        new_train_yaml["name"] = name
        new_train_yaml["num_gpu"] = train_num_gpu
        new_train_yaml["scale"]= scale
        new_train_yaml["model_type"] = model_type

        # ===== 原网络结构设置 （所有设置的网络都要以network_开头）===== 
        network_g = new_train_yaml["network_g"]

        network_g["type"] = network_g_type
        network_g["Extraction_Block"] = e_block
        network_g["Fusion_Block"] = f_block
        network_g["up_scale"] = scale
        network_g["width"] = width
        network_g["num_blks"] = num_blks
        network_g["drop_path_rate"] = drop_path_rate
        network_g["drop_out_rate"] = drop_out_rate

        # ===== 训练设置 ===== 

        train = new_train_yaml["train"]

        train["scheduler"]["T_max"] = total_iter
        train["total_iter"] = total_iter

        # ===== 数据集设置 ===== 
        datasets = new_train_yaml["datasets"]

        datasets["train"]["dataroot_gt"] = train_data
        datasets["train"]["dataroot_lq"] = train_data
        datasets["train"]["file_num"] = file_num
        datasets["train"]["gt_size_h"] = gt_size_h
        datasets["train"]["gt_size_w"] = gt_size_w
        datasets["train"]["batch_size_per_gpu"] = batch_size_per_gpu


        datasets["val"]["dataroot_gt"] = val_data_gt
        datasets["val"]["dataroot_lq"] = val_data_lq

    train_out_file_name = "train_" + name + ".yml"
    train_save_dir = os.path.join("./options", "x" + str(scale), train_out_file_name)

    with open(train_save_dir, "w") as wf:
        yaml.dump(new_train_yaml, wf, sort_keys = False)


    new_test_yaml = {}
    with open("./options/test_template.yml", "r") as f:
        test_template = yaml.safe_load(f)
        new_test_yaml = test_template

        new_test_yaml["name"] = name
        new_test_yaml["num_gpu"] = 1
        new_test_yaml["scale"]= scale
        new_test_yaml["model_type"] = model_type

        network_g = new_test_yaml["network_g"]
        network_g["type"] = network_g_type
        network_g["Extraction_Block"] = e_block
        network_g["Fusion_Block"] = f_block
        network_g["up_scale"] = scale
        network_g["width"] = width
        network_g["num_blks"] = num_blks

        new_test_yaml["path"]["pretrain_network_g"] = "experiments/" + name + "/models/net_g_latest.pth"
        for i in range(test_data_num):
            test_n = "test" + str(i)
            dataroot_lq = new_test_yaml["datasets"][test_n]["dataroot_lq"][:-1] + str(scale)
            new_test_yaml["datasets"][test_n]["dataroot_lq"] = dataroot_lq

    test_out_file_name = "test_" + name + ".yml"
    test_save_dir = os.path.join("./options", "x" + str(scale), test_out_file_name)

    with open(test_save_dir, "w") as wf:
        yaml.dump(new_test_yaml, wf, sort_keys = False)


    


