# Ref : https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html#tf-trt-api-20

# My tensorflow version : tf-nightly-gpu-2.0-preview

"""
The name of the Python class has the suffix V2, i.e. TrtGraphConverterV2.

Note: The original Python function create_inference_graph that was used in TensorFlow 1.13 and earlier is removed in to TensorFlow 2.0.

The constructor of TrtGraphConverterV2 supports the following optional arguments. The most important of these arguments that are used to configure TF-TRT or TensorRT to get better performance are stored in the namedtupleTrtConversionParams as explained in the following sections of this chapter.
input_saved_model_dir

Default value is None. This is the directory to load the SavedModel which contains the input graph to transforms.

input_saved_model_tags
    Default value is None. This is a list of tags to load the SavedModel.

input_saved_model_signature_key
    Default value is None. This is the key of the signature to optimize the graph for.

conversion_params
    Default value is DEFAULT_TRT_CONVERSION_PARAMS. An instance of namedtupleTrtConversionParams consisting the following items:

    rewriter_config_template
        A template RewriterConfig proto used to create a TRT-enabled RewriterConfig. If None, it will use a default one.

    max_workspace_size_bytes
        Default value is 1GB. The maximum GPU temporary memory which the TensorRT engine can use at execution time. This corresponds to the workspaceSize parameter of nvinfer1::IBuilder::setMaxWorkspaceSize().

    precision_mode
        Default value is TrtPrecisionMode.FP32. This is one of TrtPrecisionMode.supported_precision_modes(), in other words, FP32, FP16 or INT8 (lowercase is also supported).

    minimum_segment_size
        Default value is 3. This is the minimum number of nodes required for a subgraph to be replaced by TRTEngineOp.

    maximum_cached_engines
        Default value is 1. This is the maximum number of cached TensorRT engines in dynamic TensorRT ops. If the number of cached engines is already at max but none of them can serve the input, the TRTEngineOp will fall back to run the TensorFlow function based on which the TRTEngineOp is created.

use_calibration
    Default value is True. This argument is ignored if precision_mode is not INT8. If set to True, a calibration graph will be created to calibrate the missing ranges. The calibration graph must be converted to an inference graph by running calibration with calibrate(). If set to False, quantization nodes will be expected for every tensor in the graph (excluding those which will be fused). If a range is missing, an error will occur.
    Note: Accuracy may be negatively affected if there is a mismatch between which tensors TensorRT quantizes and which tensors were trained with fake quantization.
    The main methods you can use in the TrtGraphConverter class are the following:
    TrtGraphConverter.convert(calibration_input_fn)
    This method runs the conversion and returns the converted TensorFlow function (note that this method returns the converted GraphDef in TensorFlow 1.x). The conversion and optimization that are performed depends on the arguments passed to the constructor as explained above.

    This method only segments the graph in order to separate the TensorRT subgraphs, i.e. optimizing each TensorRT subgraph happens later during runtime (in TensorFlow 1.x this behaviour depends on is_dynamic_mode but this argument is not supported in TensorFlow 2.0 anymore; i.e. only is_dynamic_op=True is supported).

    This method has only one optional argument which should be used in case INT8 calibration is desired. The argument calibration_input_fn is a generator function that yields input data as a list or tuple, which will be used to execute the converted signature for INT8 calibration. All the returned input data should have the same shape. Note that in TensorFlow 1.x, the INT8 calibration was performed using the separate method calibrate() which is removed from TensorFlow 2.0.

TrtGraphConverter.build(input_fn)
    This method optimizes the converted function (returned by convert()) by building TensorRT engines. This is useful in case the user wants to perform the optimizations before runtime. The optimization is done by running inference on the converted function using the input data received from the argument input_fn. This argument is a generator function that yields input data as a list or tuple.

TrtGraphConverter.save
    This method saves the converted function as a SavedModel. Note that the saved TensorFlow model is still not optimized yet with TensorRT (engines are not built) in case build() is not called.
"""

### TF-TRT in TensorFlow 1.x (with default configuration)

# SavedModel format:
from tensorflow.python.compiler.tensorrt import trt_convert as trt
converter = trt.TrtGraphConverter(input_saved_model_dir=input_saved_model_dir)
converter.convert()
converter.save(output_saved_model_dir)

# Frozen graph:
from tensorflow.python.compiler.tensorrt import trt_convert as trt
converter = trt.TrtGraphConverter(
	input_graph_def=frozen_graph,
	nodes_blacklist=['logits', 'classes'])
frozen_graph = converter.convert()

### TF-TRT in TensorFlow 2.0 (with default configuration)

# SavedModel format:
from tensorflow.python.compiler.tensorrt import trt_convert as trt
converter = trt.TrtGraphConverterV2(input_saved_model_dir=input_saved_model_dir)
converter.convert()
converter.save(output_saved_model_dir)

# Full Example:

import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
conversion_params = conversion_params._replace(
    max_workspace_size_bytes=(1<<32),
    precision_mode="FP16",
    maximum_cached_engines=100)

converter = trt.TrtGraphConverterV2(
    input_saved_model_dir=input_saved_model_dir,
    conversion_params=conversion_params)

converter.convert()
def batched_input_fn(batch_size=32,image_size=(299,299),channels=3):
# Input for a single inference call, for a network that has two input tensors:
  inp1 = np.random.normal(size=(batch_size, image_size[0],image_size[1], channels)).astype(np.float32)
  inp2 = np.random.normal(size=(batch_size, image_size[0],image_size[1], channels)).astype(np.float32)
  yield (inp1, inp2)

converter.build(input_fn=batched_input_fn)
converter.save(output_saved_model_dir)

saved_model_loaded = tf.saved_model.load(output_saved_model_dir, tags=[tag_constants.SERVING])
graph_func = saved_model_loaded.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
frozen_func = convert_to_constants.convert_variables_to_constants_v2(graph_func)
output = frozen_func(input_data)[0].numpy()

# Using Automatic Mixed Precision
opt = tf.train.MomentumOptimizer(learning_rate, momentum)
opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
update_op = opt.minimize(loss)