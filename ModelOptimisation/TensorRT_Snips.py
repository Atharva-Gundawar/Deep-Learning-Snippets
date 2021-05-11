"""
TF-TRT in TensorFlow 1.x (with default configuration)
"""
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

"""
TF-TRT in TensorFlow 2.0 (with default configuration)
"""

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