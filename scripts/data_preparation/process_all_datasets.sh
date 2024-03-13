
# 切换到该脚本所在目录再执行 
cd "$(dirname "$0")"

echo " ===========================  Processing Flickr1024 dataset... ===========================  "

# 训练集处理
# python train_data_Flickr1024_process.py --dataset_name Flickr1024 --scale 2 --start_file_num 0
# python train_data_Flickr1024_process.py --dataset_name Flickr1024 --scale 4 --start_file_num 0


# # 验证集处理
# python val_test_data_process.py --dataset_name Flickr1024 --data_type Validation --scale 2
# python val_test_data_process.py --dataset_name Flickr1024 --data_type Validation --scale 4


# # 测试集处理
# python val_test_data_process.py --dataset_name Flickr1024 --data_type Test --scale 2
# python val_test_data_process.py --dataset_name Flickr1024 --data_type Test --scale 4


echo " ===========================  Processing Middlebury dataset... ===========================  "


# 60 个训练集处理
# python train_data_Middlebury_process.py --dataset_name Middlebury --scale 2 --start_file_num 5
# python train_data_Middlebury_process.py --dataset_name Middlebury --scale 4 --start_file_num 0

# # 5个测试集处理
# python val_test_data_process.py --dataset_name Middlebury_test --data_type Test --scale 2
# python val_test_data_process.py --dataset_name Middlebury_test --data_type Test --scale 4


echo " ===========================  Processing other datasets... ===========================  "

# # KITTI2012 测试集处理
python val_test_data_process.py --dataset_name KITTI2012 --data_type Test


# # KITTI2015 测试集处理
python val_test_data_process.py --dataset_name KITTI2015 --data_type Test


# # Middlebury2021 测试集处理
python val_test_data_process.py --dataset_name Middlebury2021 --data_type Test --scale 2
python val_test_data_process.py --dataset_name Middlebury2021 --data_type Test --scale 4