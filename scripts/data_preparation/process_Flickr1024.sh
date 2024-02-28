
# 切换到该脚本所在目录再执行 
cd "$(dirname "$0")"

# 训练集处理
python train_data_process.py --dataset_name Flickr1024 --scale 2
python train_data_process.py --dataset_name Flickr1024 --scale 4


# 验证集处理
python val_test_data_process.py --dataset_name Flickr1024 --data_type Validation --scale 2
python val_test_data_process.py --dataset_name Flickr1024 --data_type Validation --scale 4


# 测试集处理
python val_test_data_process.py --dataset_name Flickr1024 --data_type Test --scale 2
python val_test_data_process.py --dataset_name Flickr1024 --data_type Test --scale 4
