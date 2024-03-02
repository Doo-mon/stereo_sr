
# 切换到该脚本所在目录再执行 
cd "$(dirname "$0")"

# KITTI2012 测试集处理
python val_test_data_process.py --dataset_name KITTI2012 --data_type Test --scale 2
python val_test_data_process.py --dataset_name KITTI2012 --data_type Test --scale 4

# KITTI2015 测试集处理
python val_test_data_process.py --dataset_name KITTI2015 --data_type Test --scale 2
python val_test_data_process.py --dataset_name KITTI2015 --data_type Test --scale 4

# Middlebury2014 测试集处理
python val_test_data_process.py --dataset_name Middlebury2014 --data_type Test --scale 2
python val_test_data_process.py --dataset_name Middlebury2014 --data_type Test --scale 4

# Middlebury2021 测试集处理
python val_test_data_process.py --dataset_name Middlebury2021 --data_type Test --scale 2
python val_test_data_process.py --dataset_name Middlebury2021 --data_type Test --scale 4