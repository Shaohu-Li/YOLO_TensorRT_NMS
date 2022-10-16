# Onnx -> TensorRT

​		在这个项目中，我们实现了从 YOLO 的 onnx 模型到 tensorrt 模型的转变，并在其中加入了 NMS tenosrrt 的插件编写，使得网络的速度提升了很多。

## 1、安装一些必要的安装包

```python
pip install --upgrade setuptools pip --user
```
```python
pip install nvidia-pyindex
```
```python
pip install --upgrade nvidia-tensorrt
```
```python
pip install pycuda
```

## 2、具体操作流程
1、下载 yolov7, 并导出 onnx。

1.1、下载源码

```python
git clone https://github.com/WongKinYiu/yolov7.git
```

1.2、下载权重

```python
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt
```

1.3、安装所需要的包

```python
pip install -r yolov7/requirements.txt
```

​	上面这一步，一般情况下，不要直接安装，有可能会破坏你原有的环境，点开 requirements.txt ，看自己缺少什么文件包，再依次安装就好了。当然要是你喜欢直接一点的，那一下子全部安装也是可以的。

1.4、导出 onnx 模型

```python
python yolov7/export.py --weights yolov7-tiny.pt --grid  --simplify
```

`--grid  --simplify` 可以将 $20 *20 *3 *85、40 *40 *3 *85、80 *80 *3 *85$ 的不同阶段的输出合并在一起，省事，不然再转变 TRT 模型的时候，输出的为字节的话，自己还需要进行转变。

2、运行 export_TRT.py, 进行 TRT 模型的导出和图片的测试
具体的步骤参考：examples.ipynb 文件

## demo

在 images 文件下，有一个时评 demo，检测代码的时候，可以尝试运行这个视频