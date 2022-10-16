import os
import tensorrt as trt
import pycuda.driver as cuda
from utils.utils_plugin import nms_plugin

def readTrtFile(engine_file_path, logger):
    """从已经存在的文件中读取 TRT 模型

    Args:
        engine_file_path: 已经存在的 TRT 模型的路径

    Returns:
        加载完成的 engine
    """

    engine_file_path = os.path.realpath(engine_file_path)
    print("Loading TRT fil from : {}".format(engine_file_path))

    runtime = trt.Runtime(logger)

    with open(engine_file_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    assert engine, "反序列化之后的 engien 为空，确保转换过程的正确性 . "
    print("From {} load engine sucess . ".format(engine_file_path))
    return engine

def onnxToTRTModel(onnx_file_path, engine_file_path, logger, 
                    precision_flop='fp16', end2end=False, max_det=100, conf_threshold=0.5, nms_threshold=0.45):
    """构建期 -> 转换网络模型为 TRT 模型

    Args:
        onnx_file_path  : 要转换的 onnx 模型的路径
        engine_file_path: 转换之后的 TRT engine 的路径
        precision_flop  : 转换过程中所使用的精度

    Returns:
        转化成功: engine
        转换失败: None
    """
    #---------------------------------#
    # 准备全局信息
    #---------------------------------#
    # 构建一个 构建器
    builder = trt.Builder(logger)
    builder.max_batch_size = 1

    #---------------------------------#
    # 第一步，读取 onnx
    #---------------------------------#
    # 1-1、设置网络读取的 flag
    # EXPLICIT_BATCH 相教于 IMPLICIT_BATCH 模式，会显示的将 batch 的维度包含在张量维度当中，
    # 有了 batch大小的，我们就可以进行一些必须包含 batch 大小的操作了，如 Layer Normalization。  
    #不然在推理阶段，应当指定推理的 batch 的大小。目前主流的使用的 EXPLICIT_BATCH 模式
    network_flags 	= (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    # 1-3、构建一个空的网络计算图
    network 		= builder.create_network(network_flags)
    # 1-4、将空的网络计算图和相应的 logger 设置装载进一个 解析器里面
    parser 			= trt.OnnxParser(network, logger)
    # 1-5、打开 onnx 压缩文件，进行模型的解析工作。
    # 解析器 工作完成之后，网络计算图的内容为我们所解析的网络的内容。
    onnx_file_path 	= os.path.realpath(onnx_file_path) # 将路径转换为绝对路径防止出错
    if not os.path.isfile(onnx_file_path):
        print("ONNX file not exist. Please check the onnx file path is right ? ")
        return None
    else:
        with open(onnx_file_path, 'rb') as model:
            parser_flag = parser.parse(model.read())
            if not parser_flag:
                print("ERROR: Failed to parse the onnx file {} . ".format(onnx_file_path))
                # 出错了，将相关错误的地方打印出来，进行可视化处理`-`
                for error in range(parser.num_errors):
                    print(parser.num_errors)
                    print(parser.get_error(error))
                return None
        print("Completed parsing ONNX file . ")
    # 6、将转换之后的模型的输入输出的对应的大小进行打印，从而进行验证
    inputs 	= [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]

    print("Network Description")
    for input in inputs:
        # 获取当前转化之前的 输入的 batch_size
        batch_size = input.shape[0]
        print("Input '{}' with shape {} and dtype {} . ".format(input.name, input.shape, input.dtype))
    for output in outputs:
        print("Output '{}' with shape {} and dtype {} . ".format(output.name, output.shape, output.dtype))
    # 确保 输入的 batch_size 不为零
    assert batch_size > 0, "输入的 batch_size < 0, 请确定输入的参数是否满足要求. "

    if end2end:
        network = nms_plugin(network, logger, max_det, conf_threshold, nms_threshold)

    #---------------------------------#
    # 第二步，转换为 TRT 模型
    #---------------------------------#
    # 2-1、设置 构建器 的 相关配置器
    # 应当丢弃老版本的 builder. 进行设置的操作
    config = builder.create_builder_config()
    # 2-2、设置 可以为 TensorRT 提供策略的策略源。如CUBLAS、CUDNN 等
    # 也就是在矩阵计算和内存拷贝的过程中选择不同的策略
    config.set_tactic_sources(1 << int(trt.TacticSource.CUBLAS))
    # 2-3、给出模型中任一层能使用的内存上限，这里是 2^30,为 2GB
    # 每一层需要多少内存系统分配多少，并不是每次都分 2 GB
    config.max_workspace_size = 2 << 30
    # 2-4、设置 模型的转化精度
    config.set_flag(trt.BuilderFlag.STRICT_TYPES)
    if precision_flop == "fp16":
        if not builder.platform_has_fast_fp16:
            print("FP16 is not supported natively on this platform/device . ")
        else:
            config.set_flag(trt.BuilderFlag.FP16)

    # 2-5，从构建器 构建引擎
    engine = builder.build_engine(network, config)

    #---------------------------------#
    # 第三步，生成 SerializedNetwork
    #---------------------------------#
    # 3-1、删除已经已经存在的版本
    engine_file_path 	= os.path.realpath(engine_file_path) # 将路径转换为绝对路径防止出错
    if os.path.isfile(engine_file_path):
        try:
            os.remove(engine_file_path)
        except Exception:
            print("Cannot removing existing file: {} ".format(engine_file_path))

    print("Creating Tensorrt Engine: {}".format(engine_file_path))

    # 3-2、打开要写入的 TRT engine，利用引擎写入
    with open(engine_file_path, "wb") as f:
        f.write(engine.serialize())
    print("ONNX -> TRT Success。 Serialized Engine Saved at: {} . ".format(engine_file_path))

    return engine