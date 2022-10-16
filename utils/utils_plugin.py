import numpy as np
import tensorrt as trt

def nms_plugin(network, trt_logger, max_det, conf_thres, iou_thres):

    # 1、使用可选名称空间将所有现有 TensorRT 插件初始化并注册到 IPluginRegistry。
    # 主要我们要创建自己的 plugin， 所以我们要初始化插件库，方便我们访问
    trt.init_libnvinfer_plugins(trt_logger, namespace="")
    
    # 在当前的 runtime 的 环境下，返回 plugin 的 register
    registry = trt.get_plugin_registry()
    assert(registry)
    # 根据输入的 plugin 的 type version plugin_namsspace, 返回一个 plugin creator
    # EfficientNMS_TRT 是在 tensorrt 中已经写的 layer 我们现在将其中放入我们的输入
    creator = registry.get_plugin_creator("EfficientNMS_TRT", "1")
    assert(creator)

    # 首先你要明白你需要进行的步骤，按照正常的神经网络的搭建方式将
    # NMS 构建为一层，添加进去自己的网络层当中，
    # 首先，输入为 1 * 25200 * 85
    # 
    # 需要先将之前标签为 输出 的输出标签拿掉，后续我们再自己加
    previous_output = network.get_output(0)
    network.unmark_output(previous_output)

    # 拆分自己得到的输出，将其满足 EfficientNMS_TRT 的输入的要去
    strides = trt.Dims([1,1,1])
    starts = trt.Dims([0,0,0])
    bs, num_boxes, temp = previous_output.shape
    shapes = trt.Dims([bs, num_boxes, 4])
    # 从 [0, 0, 0] 开始，按照 [1, num_boxes , 4] 点乘 [1, 1, 1] 的步幅大小进行切片操作 -》  [1, num_boxes, 4] 
    boxes = network.add_slice(previous_output, starts, shapes, strides)
    num_classes = temp -5 
    starts[2] = 4
    shapes[2] = 1
    # [0, 0, 4] [1, 8400, 1] [1, 1, 1]
    obj_score = network.add_slice(previous_output, starts, shapes, strides)
    starts[2] = 5
    shapes[2] = num_classes
    # [0, 0, 5] [1, 8400, 80] [1, 1, 1]
    scores = network.add_slice(previous_output, starts, shapes, strides)
    # scores = obj_score * class_scores => [bs, num_boxes, nc]
    # 再重新计算分数
    updated_scores = network.add_elementwise(obj_score.get_output(0), scores.get_output(0), trt.ElementWiseOperation.PROD)

    # 我们再看看 EfficientNMS_TRT 版本为 1 的输入的要求
    '''
    "plugin_version": "1",
    "background_class": -1,  # no background class
    "max_output_boxes": detections_per_img,
    "score_threshold": score_thresh,
    "iou_threshold": nms_thresh,
    "score_activation": False,
    "box_coding": 1,
    '''

    # 将一些必要的参数 trt 化，并后面传入 EfficientNMS_TRT 中
    fc = []
    fc.append(trt.PluginField("background_class", np.array([-1], dtype=np.int32), trt.PluginFieldType.INT32))
    fc.append(trt.PluginField("max_output_boxes", np.array([max_det], dtype=np.int32), trt.PluginFieldType.INT32))
    fc.append(trt.PluginField("score_threshold", np.array([conf_thres], dtype=np.float32), trt.PluginFieldType.FLOAT32))
    fc.append(trt.PluginField("iou_threshold", np.array([iou_thres], dtype=np.float32), trt.PluginFieldType.FLOAT32))
    fc.append(trt.PluginField("box_coding", np.array([1], dtype=np.int32), trt.PluginFieldType.INT32))

    fc = trt.PluginFieldCollection(fc) 
    nms_layer = creator.create_plugin("nms_layer", fc)

    layer = network.add_plugin_v2([boxes.get_output(0), updated_scores.get_output(0)], nms_layer)
    layer.get_output(0).name = "num"
    layer.get_output(1).name = "boxes"
    layer.get_output(2).name = "scores"
    layer.get_output(3).name = "classes"
    for i in range(4):
        network.mark_output(layer.get_output(i))
    return network

if __name__ == "__main__":
    print("这里是你自己的插件库")