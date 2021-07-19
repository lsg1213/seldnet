import os
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import  convert_variables_to_constants_v2_as_graph

from params import get_param
from train import get_dataset
import models
from model_complexity import conv_temporal_complexity
from complexity import conv2d_complexity


def get_flops(model):
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(
        [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
        return flops.total_float_ops


if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    config, model_config = get_param()
    
    testset = get_dataset(config, 'test')
    x, y = [(x, y) for x, y in testset.take(1)][0]
    input_shape = x.shape
    model_config['n_classes'] = 12
    model = getattr(models, config.model)(input_shape, model_config)
    print(get_flops(model))
    # model.summary()
    