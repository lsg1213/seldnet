import copy
import tensorflow as tf
from tensorflow.keras.layers import *
import tensorflow_addons as tfa
from layers import *
from utils import safe_tuple


stages_1d = ['bidirectional_GRU_stage',
             'transformer_encoder_stage',
             'simple_dense_stage',
             'conformer_encoder_stage']

stages_2d = ['mother_stage',
             'DPRNN_stage']


"""
Modules

This is only for implementing modules.
Use only custom layers or predefined 
"""

"""            STAGES            """
def simple_conv_block(model_config: dict):
    # mandatory parameters
    filters = model_config['filters']
    pool_size = model_config['pool_size']

    dropout_rate = model_config.get('dropout_rate', 0.)
    kernel_regularizer = tf.keras.regularizers.l1_l2(
        **model_config.get('kernel_regularizer', {'l1': 0., 'l2': 0.}))

    if len(filters) == 0:
        filters = filters * len(pool_size)
    elif len(filters) != len(pool_size):
        raise ValueError("len of filters and pool_size do not match")
    
    def conv_block(inputs):
        x = inputs
        for i in range(len(filters)):
            x = conv2d_bn(filters[i], kernel_size=3, 
                          kernel_regularizer=kernel_regularizer)(x)
            x = MaxPooling2D(pool_size=pool_size[i])(x)
            x = Dropout(dropout_rate)(x)
        return x
    return conv_block


def DPRNN_stage(model_config: dict):
    depth = model_config['DPRNN_depth']

    def stage(x):
        for _ in range(depth):
            x = DPRNN_block(model_config)(x)
        return x
    return stage


def single_RNN_block(inp, units, dropout=0., bidirectional=False, rnn='GRU'):
    RNN = getattr(tf.keras.layers, rnn)(units, dropout=dropout, return_sequences=True)
    if bidirectional:
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Bidirectional(RNN))(inp)
    else:
        x = tf.keras.layers.TimeDistributed(RNN)(inp)
    x = tf.keras.layers.Dense(inp.shape[-1])(x)
    return x


def DPRNN_block(model_config: dict):
    units = model_config['DPRNN_units']
    bidirectional = model_config['DPRNN_bidirectional']
    rnn = model_config['DPRNN_rnn']
    dropout = model_config.get('DPRNN_dropout', 0.)

    def block(x):
        output = x

        row_input = tf.keras.layers.Permute([2, 1, 3])(output) # B, dim2, dim1, N
        row_output = single_RNN_block(row_input, units, dropout=dropout, bidirectional=True)
        row_output = tf.keras.layers.Permute([2, 1, 3])(row_output)
        row_output = tfa.layers.GroupNormalization(1, epsilon=1e-8)(row_output)
        output += row_output

        col_output = single_RNN_block(output, units, dropout=dropout, bidirectional=True)
        col_output = tfa.layers.GroupNormalization(1, epsilon=1e-8)(col_output)

        output += col_output
        return output # (B, dim1, dim2, output_size)
    return block


def mother_stage(model_config: dict):
    depth = model_config['mother_depth']
    strides = model_config['strides']
    model_config = copy.deepcopy(model_config)

    def stage(x):
        for i in range(depth):
            x = mother_block(model_config)(x)
            model_config['strides'] = (1, 1)
        return x
    return stage


def bidirectional_GRU_stage(model_config: dict):
    '''
    essential configs
        depth: int
        units: int

    non-essential configs
        dropout_rate: (default=0.)
    '''
    depth = model_config['GRU_depth']
    units = model_config['GRU_units']
    model_config = copy.deepcopy(model_config)
    model_config['GRU_units'] = [units] * depth

    return bidirectional_GRU_block(model_config)


def simple_dense_stage(model_config: dict):
    '''
    essential configs
        depth: int
        units: int

    non-essential configs
        activation: (default=None)
        dropout_rate: (default=0.)
        kernel_regularizer: (default={'l1': 0., 'l2': 0.})
    '''
    depth = model_config['dense_depth']
    units = model_config['dense_units']
    model_config = copy.deepcopy(model_config)
    model_config['dense_units'] = [units] * depth
    model_config['dense_activation'] = model_config.get('activation', None)
    
    return simple_dense_block(model_config)


def transformer_encoder_stage(model_config: dict):
    '''
    essential configs
        depth: int
        n_head: int
        key_dim: int
        ff_multiplier: int or float
        kernel_size: int

    non-essential configs
        activation: (default=relu)
        dropout_rate: (default=0.1)
    '''
    depth = model_config['transformer_depth']

    def stage(x):
        x = force_1d_inputs()(x)
        for i in range(depth):
            x = transformer_encoder_block(model_config)(x)
        return x
    return stage


def conformer_encoder_stage(model_config: dict):
    '''
    essential configs
        depth: int

    non-essential configs
        key_dim: (default=36)
        n_head: (default=4)
        kernel_size: (default=32)
        activation: (default=swish)
        dropout_rate: (default=0.1)
        multiplier: (default=4)
        ffn_factor: (default=0.5)
        pos_encoding: (default=basic)
        kernel_regularizer: (default={'l1': 0., 'l2': 0.})
    '''
    depth = model_config['conformer_depth']

    def stage(x):
        inputs = force_1d_inputs()(x)
        for i in range(depth):
            x = conformer_encoder_block(model_config)(x)
        return x
    return stage


"""            BLOCKS WITH 2D OUTPUTS            """
def mother_block(model_config: dict):
    filters0 = model_config['filters0'] # 0 if skipped
    filters1 = model_config['filters1'] # 0 if skipped
    filters2 = model_config['filters2'] # 0 if skipped
    kernel_size0 = model_config['kernel_size0'] # 0 if skipped
    kernel_size1 = model_config['kernel_size1'] # 0 if skipped
    kernel_size2 = model_config['kernel_size2'] # 0 if skipped
    connect0 = model_config['connect0'] # len of 1 (0: input)
    connect1 = model_config['connect1'] # len of 2 (0: input, 1: out0)
    connect2 = model_config['connect2'] # len of 3 (0: input, 1: out0, 2: out1)

    strides = safe_tuple(model_config.get('strides', (1, 1)))
    activation = model_config.get('activation', 'relu')
    try:
        if (filters0 == 0) != (kernel_size0 == 0):
            raise ValueError('0) skipped layer must have 0 filters, 0 kernel size')
        if (filters1 == 0) != (kernel_size1 == 0):
            raise ValueError('1) skipped layer must have 0 filters, 0 kernel size')
        if (filters2 == 0) != (kernel_size2 == 0):
            raise ValueError('2) skipped layer must have 0 filters, 0 kernel size')

        if filters0 == 0 and max(connect1[1], connect2[1]):
            raise ValueError('cannot link skipped layer (first layer)')
        if filters1 == 0 and connect2[2] > 0:
            raise ValueError('cannot link skipped layer (second layer)')

        if (filters0 != 0) + sum(connect0) == 0:
            raise ValueError('cannot pass zero inputs to the second layer')
        if (filters1 != 0) + sum(connect1) == 0:
            raise ValueError('cannot pass zero inputs to the third layer')
        if (filters2 != 0) + sum(connect2) == 0:
            raise ValueError('cannot pass zero inputs to the final output')

        if filters1 == 0 and tuple(strides) != (1, 1):
            raise ValueError('if strides are set, the second layer must be active')
    except:
        print(filters0, filters1, filters2)
        print(kernel_size0, kernel_size1, kernel_size2)
        print(connect0, connect1, connect2)
        import pdb; pdb.set_trace()

    def block(inputs):
        outputs = [inputs]

        # first layer
        if filters0 > 0:
            out = Conv2D(filters0, kernel_size0, padding='same')(outputs[-1])
            out = BatchNormalization()(out)
            if connect0[0] == 1:
                skip = outputs[-1]
                if skip.shape[-3:] != out.shape[-3:]:
                    skip = Conv2D(filters0, 1)(skip)
                    skip = BatchNormalization()(skip)
                out += skip
            out = Activation(activation)(out)
        else:
            out = outputs[-1]
        outputs.append(out)

        # second layer (apply strides)
        if filters1 > 0:
            out = Conv2D(filters1, kernel_size1, padding='same',
                         strides=strides)(outputs[-1])
            out = BatchNormalization()(out)
            for i in range(len(connect1)):
                if connect1[i] == 1:
                    skip = outputs[i]
                    if skip.shape[-3:] != out.shape[-3:]:
                        skip = Conv2D(filters1, 1, strides=strides)(skip)
                        skip = BatchNormalization()(skip)
                    out += skip
            out = Activation(activation)(out)
        else:
            out = []
            for i in range(len(connect1)):
                if connect1[i] == 1:
                    out.append(outputs[i])
            out = tf.concat(out, axis=-1)
        outputs.append(out)

        # third layer
        if filters2 > 0:
            out = Conv2D(filters2, kernel_size2, padding='same')(outputs[-1])
            out = BatchNormalization()(out)
            for i in range(len(connect2)):
                if connect2[i] == 1:
                    skip = outputs[i]
                    if skip.shape[-3:] != out.shape[-3:]:
                        skip = Conv2D(
                            filters2, 1, 
                            strides=(1, 1) if i == 2 else strides)(skip)
                        skip = BatchNormalization()(skip)
                    out += skip
            out = Activation(activation)(out)
        else:
            out = []
            for i in range(len(connect2)):
                if connect2[i] == 1:
                    skip = outputs[i]
                    if connect2[-1] == 1 and tuple(strides) != (1, 1) and i < 2:
                        # connect with strided outputs
                        skip = Conv2D(skip.shape[-1], 1, strides=strides)(skip)
                    out.append(skip)
            out = tf.concat(out, axis=-1)

        return out
    return block


"""            BLOCKS WITH 1D OUTPUTS            """
def bidirectional_GRU_block(model_config: dict):
    # mandatory parameters
    units_per_layer = model_config['GRU_units']

    dropout_rate = model_config.get('dropout_rate', 0.)

    def GRU_block(inputs):
        x = force_1d_inputs()(inputs)

        for units in units_per_layer:
            x = Bidirectional(
                GRU(units, activation='tanh', 
                    dropout=dropout_rate, recurrent_dropout=dropout_rate, 
                    return_sequences=True),
                merge_mode='mul')(x)
        return x

    return GRU_block


def simple_dense_block(model_config: dict):
    # assumes 1D inputs
    # mandatory parameters
    units_per_layer = model_config['dense_units']

    kernel_size = model_config.get('kernel_size', 1)
    activation = model_config.get('dense_activation', None)
    dropout_rate = model_config.get('dropout_rate', 0)
    kernel_regularizer = tf.keras.regularizers.l1_l2(
        **model_config.get('kernel_regularizer', {'l1': 0., 'l2': 0.}))

    def dense_block(inputs):
        x = force_1d_inputs()(inputs)

        for units in units_per_layer:
            if len(x.shape) == 2:
                x = Dense(units, activation=activation,
                          kernel_regularizer=kernel_regularizer)(x)
            else:
                x = Conv1D(units, kernel_size, padding='same',
                           activation=activation,
                           kernel_regularizer=kernel_regularizer)(x)
            if dropout_rate > 0:
                x = Dropout(dropout_rate)(x)
        return x

    return dense_block


def transformer_encoder_block(model_config: dict):
    n_head = model_config['transformer_n_head']
    key_dim = model_config['transformer_key_dim']
    ff_multiplier = model_config['ff_multiplier'] # default to 4 
    kernel_size = model_config['transformer_kernel_size'] # default to 1

    activation = model_config.get('activation', 'relu')
    dropout_rate = model_config.get('dropout_rate', 0.1)
    
    def block(inputs):
        x = force_1d_inputs()(inputs)
        d_model = x.shape[-1]

        attn = MultiHeadAttention(
            n_head, key_dim, dropout=dropout_rate)(x, x)
        attn = Dropout(dropout_rate)(attn)
        x = LayerNormalization()(x + attn)

        # FFN
        ffn = Conv1D(int(ff_multiplier*d_model), kernel_size, padding='same',
                     activation=activation)(x)
        ffn = Dropout(dropout_rate)(ffn)
        ffn = Conv1D(d_model, kernel_size, padding='same')(ffn)
        ffn = Dropout(dropout_rate)(ffn)
        x = LayerNormalization()(x + ffn)

        return x

    return block


def conformer_encoder_block(model_config: dict):
    # mandatory parameters
    key_dim = model_config.get('conformer_key_dim', 36)
    n_head = model_config.get('conformer_n_head', 4)
    kernel_size = model_config.get('conformer_kernel_size', 32) # 32 
    activation = model_config.get('activation', 'swish')
    dropout_rate = model_config.get('dropout_rate', 0.1)
    multiplier = model_config.get('multiplier', 4)
    ffn_factor = model_config.get('ffn_factor', 0.5)
    pos_encoding = model_config.get('pos_encoding', 'basic')
    kernel_regularizer = tf.keras.regularizers.l1_l2(
        **model_config.get('kernel_regularizer', {'l1': 0., 'l2': 0.}))

    if pos_encoding == 'basic':
        pos_encoding = basic_pos_encoding
    elif pos_encoding == 'rff': # random fourier feature
        pos_encoding = rff_pos_encoding
    else:
        pos_encoding = None
    
    def conformer_block(inputs):
        inputs = force_1d_inputs()(inputs)
        x = inputs
        batch, time, emb = x.shape

        # FFN Modules
        ffn = LayerNormalization()(x)
        ffn = Dense(multiplier*emb, activation=activation)(ffn)
        ffn = Dropout(dropout_rate)(ffn)
        ffn = Dense(emb)(ffn)
        ffn = Dropout(dropout_rate)(ffn)
        x = x + ffn_factor*ffn

        # Positional Encoding
        if pos_encoding:
            x += pos_encoding(x.shape)(x)
            
        # Multi Head Self Attention module
        attn = LayerNormalization()(x)
        attn = MultiHeadAttention(n_head,
                                  key_dim,
                                  dropout=dropout_rate)(attn, attn)
        attn = Dropout(dropout_rate)(attn)
        x = attn + x

        # Conv Module
        conv = LayerNormalization()(x)
        conv = Conv1D(filters=2*emb, 
                      kernel_size=1,
                      kernel_regularizer=kernel_regularizer)(conv)
        
        # GLU Part
        conv_1, conv_2 = tf.split(conv, 2, axis=-1)
        conv_2 = tf.keras.activations.sigmoid(conv_2)
        conv = conv_1 * conv_2

        #Depth Wise
        conv = Conv1D(filters=emb,
                      kernel_size=kernel_size,
                      strides=1,
                      padding='same',
                      groups=emb, 
                      kernel_regularizer=kernel_regularizer)(conv)

        conv = BatchNormalization()(conv)
        conv = tf.keras.activations.swish(conv) 
        conv = Conv1D(filters=emb, 
                      kernel_size=1,
                      padding="same",
                      kernel_regularizer=kernel_regularizer)(conv)
        conv = Dropout(dropout_rate)(conv)
        conv = conv + x

        # FFN
        ffn = LayerNormalization()(conv)
        ffn = Dense(multiplier*emb, activation=activation)(ffn)
        ffn = Dropout(dropout_rate)(ffn)
        ffn = Dense(emb)(ffn)
        ffn = Dropout(dropout_rate)(ffn)
        
        x = LayerNormalization()(x + ffn_factor*ffn)

        return x

    return conformer_block


"""                 OTHER BLOCKS                 """
def identity_block(model_config: dict):
    def identity(inputs):
        return inputs
    return identity

