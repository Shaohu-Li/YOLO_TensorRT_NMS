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

权重再 yolov7 的[官方仓库](https://github.com/WongKinYiu/yolov7)里面有链接，大家可以选择自己的想要转化的权重， 都是OK 的。

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

2、运行 export_TRT.py

目前里面是包含了写了代码，加载了 TensorRT 官方的 NMS  的 plugin，所以支持 end2end 的操作。

2.1、不包含 NMS plugin

```python
python export_TRT.py -o weights/yolov7x.onnx
```

 2.2、包含 NMS plugin

```python
python export_TRT.py -o weights/yolov7x.onnx -s --end2end
```

也可直接运行：examples.ipynb 文件

<font color=red>说明 : </font> 

1、export_TRT.py 可以在运行终端输入很多参数，但大多数的参数都包含了默认值，想要修改的话，直接在终端输入对应的参数的即可。

2、默认运行 导出 TensorRT 模型，并测试网络性能，到处的 TensorRT 模型和输出的 ONNX 模型位置相同，名字相同，仅仅是后缀名不相同。

3、代码中包含 fps 测试模块，可以对于使用 NMS plugin 和不使用 NMS plugin 的 fps 上的差别；

## demo

在 images 文件下，有一个时评 demo，检测代码的时候，可以尝试运行这个视频



## 参考：

参考大神仓库：[https://github.com/Linaom1214/TensorRT-For-YOLO-Series](https://github.com/Linaom1214/TensorRT-For-YOLO-Series)