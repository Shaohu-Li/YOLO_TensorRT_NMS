#---------------------------------#
# 在这个里面定一些常用的函数
#---------------------------------#
import os
import sys
import cv2
import torch
import time
import torchvision
import numpy as np

#------------------------------------------------------------------------------------------------#
# 前处理
#------------------------------------------------------------------------------------------------#

def prepareImage(org_img, netinput_size):
    """对输入的图片进行预处理, 包括 正则化, 不改变宽高比的resize, 还有改变通道顺序

    Args:
        org_img         : 原始的读取的图片
        netinput_size   : 网络需要的图片的大小

    Returns:
        返回处理好的图片，并返回改变率
    """
    if len(org_img.shape) == 3:
        padded_img = np.ones((netinput_size[0], netinput_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(netinput_size) * 114.0

    img = np.array(org_img)
    ratio = min(netinput_size[0] / img.shape[0], netinput_size[1] / img.shape[1])

    resized_img = cv2.resize(img, (int(img.shape[1] * ratio), int(img.shape[0] * ratio)), interpolation=cv2.INTER_LINEAR,).astype(np.float32)

    padded_img[: int(img.shape[0] * ratio), : int(img.shape[1] * ratio)] = resized_img

    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0

    padded_img = padded_img.transpose((2, 0 ,1))
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)

    return padded_img, ratio

#------------------------------------------------------------------------------------------------#
# 后处理需要的工具
#------------------------------------------------------------------------------------------------#
def xywh2xyxy(x):
    """转换输入的坐标形式
    [x, y, w, h] -> [x_t, y_l, x_b, y_r]

    Args:
        x: 需要转变的坐标
    Returns:
        转换好的坐标
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def Effient_nms(boxes, scores, nms_thr):
    """对输入的检测框进行非极大值抑制

    Args:
        boxes   : 输入的所有的检测框
        scores  : 所有检测框对应的分数
        nms_thr : 检测框之间的交集的面积的阈值
    Returns:
        返回输入检测框保留的序号
    """
    # 去除所有 框 的四个点的坐标
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # 计算 每个框 的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 对得分进行排序，得出分数的序号
    order = scores.argsort()[::-1]
    # 最后的保留的序号
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        # 下面是同时计算当前检测框和剩余检测框的 交集的面积
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # 保留交集的面积小于阈值的检测框，重复上述
        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep

def NMS(prediction, ratio, num_classes = 80, conf_thres=0.5, iou_thres=0.45):
    """对检测头输出的多个检测框, 进行非极大值抑制

    Args:
        prediction  : 检测头输出的全部的检测框，具体的维度信息为 (batch * 25200 * 85)
        ratio       : 恢复到原图的尺度需要的尺度改变大小
        conf_thres  : 检测框的置信度. Defaults to 0.25.
        iou_thres   : 框和框之间的 iou 阈值. Defaults to 0.45.
    Returns:
        输出每张图片上进行过非极大值抑制的结果，最终的维度为：(n, 6); 6 -> [xyxy, conf, cls]
    """

    # 0、最终的输出结果
    boxes_after_nms = []

    # 1、首先将 prediction 的维度转换一下 (batch * 25200 * 85)-> (25200, 85)
    prediction          = np.reshape(prediction, (1, -1, int(5 + num_classes)))[0]

    # 2、得到每个检测框的的得分数，-> box_scores = obj_conf * cls_conf
    scores               = prediction[:, 4:5] * prediction[:, 5:]

    # 3、转换 (center x, center y, width, height) to (x1, y1, x2, y2), 并转换为适应图片的大小
    boxes                = xywh2xyxy(prediction[:, :4]) / ratio

    # 4、按照不同的 类别 进行 nms
    for class_i in range(num_classes):
        cls_scores = scores[:, class_i]
        cls_score_mask = cls_scores > conf_thres
        if cls_score_mask.sum() == 0:
            continue
        else:
            cls_scores = cls_scores[cls_score_mask]
            cls_boxes = boxes[:,:4][cls_score_mask]

            keep = Effient_nms(cls_boxes, cls_scores, iou_thres)

            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * class_i
                dets = np.concatenate([cls_boxes[keep], cls_scores[keep, None], cls_inds], 1)
                boxes_after_nms.append(dets)

    if len(boxes_after_nms) == 0:
        return None

    return np.concatenate(boxes_after_nms, 0)

def result_visual(img, boxes, scores, cls_ids, classes_and_colors, begin_time=0 ,is_fps=True):
    """对输入的图片进行可视化

    Args:
        img                 : 输入的图片
        boxes               : 要显示的检测框
        scores              : 检测框对应的分数
        cls_ids             : 检测框对应的雷被
        classes_and_colors  : 要显示的颜色

    Returns:
        绘制好的检测框的图片
    """
    for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(cls_ids[i])
            score = scores[i]

            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])

            color = (classes_and_colors["color"][cls_id])
            text = '{}:{:.1f}%'.format(classes_and_colors["label"][cls_id], score * 100)
            font = cv2.FONT_HERSHEY_SIMPLEX

            txt_size = cv2.getTextSize(text, font, 0.6, 2)[0]
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
            cv2.rectangle(img, (x0, y0 + 1), (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])), color, 1)
            cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.6, color, thickness=2)
    if is_fps and begin_time != 0:
        end_time = time.time()
        fps = 1. / (end_time - begin_time)
        img = cv2.putText(img, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return img

# coco 数据集对应的标签和自己设置的颜色
COCO ={
    "label":[ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
         'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
         'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 
         'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 
         'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush' ],

    "color":[(244, 67, 54), (233, 30, 99), (156, 39, 176), (103, 58, 183), (100, 30, 60), (63, 81, 181), (33, 150, 243), (3, 169, 244), (0, 188, 212),(20, 55, 200),
            (0, 150, 136), (76, 175, 80), (139, 195, 74), (205, 220, 57), (70, 25, 100), (255, 235, 59), (255, 193, 7), (255, 152, 0), (255, 87, 34), (90, 155, 50),
            (121, 85, 72), (158, 158, 158), (96, 125, 139), (15, 67, 34), (98, 55, 20), (21, 82, 172), (58, 128, 255), (196, 125, 39), (75, 27, 134), (90, 125, 120),
            (121, 82, 7), (158, 58, 8), (96, 25, 9), (115, 7, 234), (8, 155, 220), (221, 25, 72), (188, 58, 158), (56, 175, 19), (215, 67, 64), (198, 75, 20),
            (62, 185, 22), (108, 70, 58), (160, 225, 39), (95, 60, 144), (78, 155, 120), (101, 25, 142), (48, 198, 28), (96, 225, 200), (150, 167, 134), (18, 185, 90),
            (21, 145, 172), (98, 68, 78), (196, 105, 19), (215, 67, 84), (130, 115, 170), (255, 0, 255), (255, 255, 0), (196, 185, 10), (95, 167, 234),(18, 25, 190),
            (0, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (155, 0, 0), (0, 155, 0), (0, 0, 155), (46, 22, 130), (255, 0, 155), (155, 0, 255), 
            (255, 155, 0),(155, 255, 0), (0, 155, 255), (0, 255, 155), (18, 5, 40), (120, 120, 255), (255, 58, 30), (60, 45, 60), (75, 27, 244), (128, 25, 70)]
    }
