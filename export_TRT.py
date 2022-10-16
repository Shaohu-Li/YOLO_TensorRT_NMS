
import argparse
from ast import arg, parse
from genericpath import isfile
import os
import sys
import cv2
import torch
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from utils.utils import *
from utils.utils_trt import *


class TRT():
    def __init__(self, args) -> None:
        self.onnx_file_path     = args.onnx
        self.engine_file_path   = args.engine
        self.precision_flop     = args.precision
        self.end2end            = args.end2end

        self.inputs             = []
        self.outputs            = []
        self.bindings           = []

        self.img_size           = (640, 640)
        self.conf_threshold     = args.conf_thres
        self.nms_threshold      = args.iou_thres

        self.COCO               = COCO

        self.Init_model()

    def Init_model(self):
        """加载 TRT 模型, 并加载一些多次推理过程共用的参数。
            情况 1、TRT 模型不存在，会先从输入的 onnx 模型创建一个 TRT 模型，并保存，再进行推导；
            情况 2、TRT 模型存在，直接进行推导
        """
        # 1、加载 logger 等级
        self.logger = trt.Logger(trt.Logger.WARNING)

        trt.init_libnvinfer_plugins(self.logger, namespace="")

        if not self.engine_file_path:
            self.engine_file_path = self.onnx_file_path.split('.')[0] + '.trt'

        # 2、加载 TRT 模型
        if os.path.isfile(self.engine_file_path):
            self.engine = readTrtFile(self.engine_file_path, self.logger)
            assert self.engine, "从 TRT 文件中读取的 engine 为 None ! "
        else:
            self.engine = onnxToTRTModel(onnx_file_path=self.onnx_file_path, engine_file_path=self.engine_file_path, logger=self.logger, 
                                        precision_flop=self.precision_flop, end2end=self.end2end, conf_threshold=self.conf_threshold, nms_threshold=self.nms_threshold)
            assert self.engine, "从 onnx 文件中转换的 engine 为 None ! "
        
        # 3、创建上下管理器，后面进行推导使用
        self.context = self.engine.create_execution_context()
        assert self.context, "创建的上下文管理器 context 为空，请检查相应的操作"

        # 4、创建数据传输流，在 cpu <--> gpu 之间传输数据的时候使用。
        self.stream = cuda.Stream()

        # 5、在 cpu 和 gpu 上申请内存
        for binding in self.engine:
            # 对应的输入输出内容的 个数，！！！注意是个数，不是内存的大小，
            size = trt.volume(self.engine.get_binding_shape(binding))
            # 内存的类型，如 int， bool。单个数据所占据的内存大小
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # 个数 * 单个内存的大小 = 内存的真实大小，先申请 cpu 上的内存
            host_mem = cuda.pagelocked_empty(size, dtype)
            # 分配 gpu 上的内存
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            # print("size: {}, dtype: {}, device_mem: {}".format(size, dtype, device_mem))
            # 区分输入的和输出 申请的内存
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
    
    def infer_single_img(self, img):
        """对单张图片进行推理，返回处理好的图片

        Args:
            img: 输入的图片
        Returns:
            返回 trt 推理的结果
        """

        # 1、对输入的数据进行处理
        self.inputs[0]['host'] = np.ravel(img) # 目前数据是放在 cpu 上
        # 2、将输入的数据同步到 gpu 上面 , 从 host -> device
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)

        # 3、执行推理（Execute / Executev2）
        # execute_async_v2  ： 对批处理异步执行推理。此方法仅适用于从没有隐式批处理维度的网络构建的执行上下文。
        # execute_v2：      ： 在批次上同步执行推理。此方法仅适用于从没有隐式批处理维度的网络构建的执行上下文。
        # 同步和异步的差异    ： 在同一个上下文管理器中，程序的执行是否严格按照从上到下的过程。
        #                     如，连续输入多张图片，同步 会等处理完结果再去获得下一张，异步会开启多线程，提前处理数据 
        self.context.execute_async_v2(
                                bindings=self.bindings, # 要进行推理的数据，放进去的时候，只有输入，出来输入、输出都有了
                                stream_handle=self.stream.handle # 将在其上执行推理内核的 CUDA 流的句柄。
                )
        # 4、Buffer 拷贝操作	Device to Host
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)

        # 5、将 stream 中的数据进行梳理
        self.stream.synchronize()

        # 6、整理输出
        data = [out['host'] for out in self.outputs]
        return data

    def inference(self, img_path, mode="video", record=False):
        """根据包不同的模式，对输入的路径进行推理

        Args:
            img_path: 输入的图片路径
            mode    : 要进行处理的模式. 默认为, "video". choice = ["video", "img"].
            record  : 是否要保存检测之后的图片的或者视频。默认是不开启的。
        """
        img_path = os.path.realpath(img_path)

        if mode == "video":
            cap = cv2.VideoCapture(img_path)
            ret, frame = cap.read()
            if not ret:
                print("视频读取出错，请检查错误. 当前输入路径为: {}. ".format(img_path))
                sys.exit(-1)
                
            if record:
                outpath         = img_path.split('.')[0] + "_result.avi"
                video_fps       = 30.0
                fourcc          = cv2.VideoWriter_fourcc(*'XVID')
                size            = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                write_handle    = cv2.VideoWriter(outpath, fourcc, video_fps, size, True ) 
            else:
                write_handle = None
            
            while ret:
                begin_time = time.time()
                ret, frame = cap.read()
                img, ratio = prepareImage(frame, self.img_size)
                engine_infer_output = self.infer_single_img(img)
                final_img = self.post_process(engine_infer_output, frame, ratio, begin_time)
                cv2.imshow("TRT inference result", final_img)

                if write_handle:
                    write_handle.write(final_img)

                if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == 27 : # 27 对应 Esc
                    break
            cap.release()
            cv2.destroyAllWindows()
        else:
            if not os.path.isfile(img_path):
                print("输入单张图片的路径出错，请检查相应的路径：{}".format(img_path))
                sys.exit(-1)
            begin_time = time.time()
            frame = cv2.imread(img_path)
            img, ratio = prepareImage(frame, self.img_size)
            engine_infer_output = self.infer_single_img(img)
            final_img = self.post_process(engine_infer_output, frame, ratio, begin_time)
            cv2.imshow("TRT inference result", final_img)
            if record:
                out_path = img_path.replace('.', '_result.')
                cv2.imwrite(out_path, final_img)
            if cv2.waitKey(-1) == ord('q') or cv2.waitKey(-1) == 27 :
                cv2.destroyAllWindows()

    # 如果在非 end2end 的情况下的时候, 我们需要对当输出的结果进行 NMS
    def post_process(self, engine_infer_output, origin_img, ratio, begin_time=0):
        """对网络输出的结果进行后处理

        Args:
            engine_infer_output : 网络输出的结果，-> ( 25200, 85)
            origin_img          : 送入网络之前的原始图片
            ratio               : 原始图片的大小 / 送入网络的图片大小

        Returns:
            最终绘制完层检测框的图片
        """
        # 再没有进行非极大值抑制的情况下，原始网络输出为 25200*85 = （ 20 * 20 + 40 *40 + 80 * 80） * 85 * 3 (三个输出头)
        if self.end2end :
            num, final_boxes, final_scores, final_cls_inds = engine_infer_output
            final_boxes = np.reshape(final_boxes/ratio, (-1, 4))
            dets = np.concatenate(
                                [final_boxes[:num[0]], 
                                np.array(final_scores)[:num[0]].reshape(-1, 1), 
                                np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], 
                                axis=-1
                                )
        else:
            dets = NMS(engine_infer_output, ratio)

        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:,:4], dets[:, 4], dets[:, 5]
            origin_img = result_visual(origin_img, final_boxes, final_scores, final_cls_inds, self.COCO, begin_time, is_fps=True)
        return origin_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx", 
                        help="输入要转化的 onnx 模型路径. ")
    parser.add_argument("-e", "--engine", default=None,
                        help="输出模型的路径，没有参数的情况下，保存为 onnx 模型相同的路径下. ")
    parser.add_argument("-p", "--precision", default="fp16", choices=["fp32", "fp16", "int8"], 
                        help="构建 TRT 模型选择的精度, 可选项为： 'fp32', 'fp16' or 'int8', default: 'fp16'")
    parser.add_argument("--end2end", default=False, action="store_true",
                        help="导出的模型是否包含 nms plugin, default: False")
    parser.add_argument("--conf_thres", default=0.5, type=float,
                        help="模型预测的置信度大小, default: 0.5")
    parser.add_argument("--iou_thres", default=0.45, type=float,
                        help="NMS 的时候， iou 的阈值, default: 0.45")
    parser.add_argument("-i", "--img_path", default="images/video1.mp4",
                        help="输入图片或者视频的路径. ")
    parser.add_argument("-m", "--mode", default="video",
                        help="是否为视频检测模式, 输入为视频的时候请开启此选项. ")
    parser.add_argument("-s", "--save", default=False,action="store_true",
                        help="是否要将检测的结果保存下来. ")

    
    args = parser.parse_args()
    print(args)
    if not args.onnx:
        parser.print_help()
        print("These arguments are required at least: --onnx")
        sys.exit(1)
    trt_model = TRT(args)
    if args.img_path:
        trt_model.inference(args.img_path, mode=args.mode, record=args.save)