# 毕设

暂时还没有完善好



```shell
# 
pip install -r requirements.txt
python setup.py develop --no_cuda_ext 
# develop 开发者模式可以之间反映修改
# --no_cuda_ext 这是传递给setup.py脚本的自定义选项，用于指示安装过程中跳过编译CUDA扩展。
```




**然后就是准备训练数据**

```shell
mkdir ~/stereo_sr/datasets
# 放置好数据集的zip文件后
cd ~
bash ~/stereo_sr/scripts/data_preparation/process_Flickr1024.sh
```


**一键训练**

具体见 narrow_setup.py 文件