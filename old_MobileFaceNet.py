import tensorflow

regularizer = tensorflow.contrib.layers.l1_l2_regularizer(1e-7, 1e-7)

def prelu(x, name):
    alpha = tensorflow.get_variable(
        name='alpha%s' % name,
        shape=x.shape.as_list()[-1],
        initializer=tensorflow.initializers.constant(0.25),
        regularizer=tensorflow.contrib.layers.l1_l2_regularizer(1e-7, 1e-7))
    y = tensorflow.nn.relu(x) - alpha * tensorflow.nn.relu(-x)
    return y

def MobileNetV2Block(x, stride, t, output_channel, layer, training):
    with tensorflow.variable_scope('MobileNetV2Block%s' % layer):
        weight1 = tensorflow.get_variable(
            name='weight1',
            shape=[1, 1, x.shape.as_list()[-1], int(t * x.shape.as_list()[-1])],
            initializer=tensorflow.initializers.glorot_uniform(),
            regularizer=tensorflow.contrib.layers.l1_l2_regularizer(1e-7, 1e-7))
        bias1 = tensorflow.get_variable(
            name='bias1',
            shape=int(t * x.shape.as_list()[-1]),
            initializer=tensorflow.initializers.zeros(),
            regularizer=tensorflow.contrib.layers.l1_l2_regularizer(1e-7, 1e-7))
        y = tensorflow.nn.conv2d(
            input=x,
            filter=weight1,
            strides=[1, 1, 1, 1],
            padding='SAME') + bias1
        y = tensorflow.layers.batch_normalization(
            inputs=y,
            beta_regularizer=None,
            gamma_regularizer=None,
            training=training,
            fused=None)
        y = prelu(y, 1)

        weight2 = tensorflow.get_variable(
            name='weight2',
            shape=[3, 3, y.shape.as_list()[-1], 1],
            initializer=tensorflow.initializers.glorot_uniform(),
            regularizer=tensorflow.contrib.layers.l1_l2_regularizer(1e-7, 1e-7))
        bias2 = tensorflow.get_variable(
            name='bias2',
            shape=y.shape.as_list()[-1],
            initializer=tensorflow.initializers.zeros(),
            regularizer=tensorflow.contrib.layers.l1_l2_regularizer(1e-7, 1e-7))
        y = tensorflow.nn.depthwise_conv2d(
            input=y,
            filter=weight2,
            strides=[1, stride, stride, 1],
            padding='SAME') + bias2
        y = tensorflow.layers.batch_normalization(
            inputs=y,
            beta_regularizer=None,
            gamma_regularizer=None,
            training=training,
            fused=None)
        y = prelu(y, 2)

        weight3 = tensorflow.get_variable(
            name='weight3',
            shape=[1, 1, y.shape.as_list()[-1], output_channel],
            initializer=tensorflow.initializers.glorot_uniform(),
            regularizer=tensorflow.contrib.layers.l1_l2_regularizer(1e-7, 1e-7))
        bias3 = tensorflow.get_variable(
            name='bias3',
            shape=output_channel,
            initializer=tensorflow.initializers.zeros(),
            regularizer=tensorflow.contrib.layers.l1_l2_regularizer(1e-7, 1e-7))
        y = tensorflow.nn.conv2d(
            input=y,
            filter=weight3,
            strides=[1, 1, 1, 1],
            padding='SAME') + bias3
        y = tensorflow.layers.batch_normalization(
            inputs=y,
            beta_regularizer=None,
            gamma_regularizer=None,
            training=training,
            fused=None)

        if x.shape.as_list() == y.shape.as_list():
            y = x + y
    return y

def MobileFaceNet(x, num_classes, training):
    with tensorflow.variable_scope('MobileFaceNet', reuse=tensorflow.AUTO_REUSE):
        with tensorflow.variable_scope('layer1'):
            weight = tensorflow.get_variable(
                name='weight',
                shape=[3, 3, x.shape.as_list()[-1], 64],
                initializer=tensorflow.initializers.glorot_uniform(),
                regularizer=tensorflow.contrib.layers.l1_l2_regularizer(1e-7, 1e-7))
            bias = tensorflow.get_variable(
                name='bias',
                shape=64,
                initializer=tensorflow.initializers.zeros(),
                regularizer=tensorflow.contrib.layers.l1_l2_regularizer(1e-7, 1e-7))
            y = tensorflow.nn.conv2d(
                input=x,
                filter=weight,
                strides=[1, 2, 2, 1],
                padding='SAME') + bias
            y = tensorflow.layers.batch_normalization(
                inputs=y,
                beta_regularizer=None,
                gamma_regularizer=None,
                training=training,
                fused=None)
            y = prelu(y, 1)

        with tensorflow.variable_scope('layer2'):
            weight = tensorflow.get_variable(
                name='weight',
                shape=[3, 3, y.shape.as_list()[-1], 1],
                initializer=tensorflow.initializers.glorot_uniform(),
                regularizer=tensorflow.contrib.layers.l1_l2_regularizer(1e-7, 1e-7))
            bias = tensorflow.get_variable(
                name='bias',
                shape=y.shape.as_list()[-1],
                initializer=tensorflow.initializers.zeros(),
                regularizer=tensorflow.contrib.layers.l1_l2_regularizer(1e-7, 1e-7))
            y = tensorflow.nn.depthwise_conv2d(
                input=y,
                filter=weight,
                strides=[1, 1, 1, 1],
                padding='SAME') + bias
            y = tensorflow.layers.batch_normalization(
                inputs=y,
                beta_regularizer=None,
                gamma_regularizer=None,
                training=training,
                fused=None)
            y = prelu(y, 2)

        y = MobileNetV2Block(y, 2, 2, 64, 3, training)
        y = MobileNetV2Block(y, 1, 2, 64, 4, training)
        y = MobileNetV2Block(y, 1, 2, 64, 5, training)
        y = MobileNetV2Block(y, 1, 2, 64, 6, training)
        y = MobileNetV2Block(y, 1, 2, 64, 7, training)

        y = MobileNetV2Block(y, 2, 4, 128, 8, training)
        y = MobileNetV2Block(y, 1, 2, 128, 9, training)
        y = MobileNetV2Block(y, 1, 2, 128, 10, training)
        y = MobileNetV2Block(y, 1, 2, 128, 11, training)
        y = MobileNetV2Block(y, 1, 2, 128, 12, training)
        y = MobileNetV2Block(y, 1, 2, 128, 13, training)
        y = MobileNetV2Block(y, 1, 2, 128, 14, training)

        y = MobileNetV2Block(y, 2, 4, 128, 15, training)

        y = MobileNetV2Block(y, 1, 2, 128, 16, training)
        y = MobileNetV2Block(y, 1, 2, 128, 17, training)

        with tensorflow.variable_scope('layer18'):
            weight = tensorflow.get_variable(
                name='weight',
                shape=[1, 1, y.shape.as_list()[-1], 512],
                initializer=tensorflow.initializers.glorot_uniform(),
                regularizer=tensorflow.contrib.layers.l1_l2_regularizer(1e-7, 1e-7))
            bias = tensorflow.get_variable(
                name='bias',
                shape=512,
                initializer=tensorflow.initializers.zeros(),
                regularizer=tensorflow.contrib.layers.l1_l2_regularizer(1e-7, 1e-7))
            y = tensorflow.nn.conv2d(
                input=y,
                filter=weight,
                strides=[1, 1, 1, 1],
                padding='SAME') + bias
            y = tensorflow.layers.batch_normalization(
                inputs=y,
                beta_regularizer=None,
                gamma_regularizer=None,
                training=training,
                fused=None)
            y = prelu(y, 1)

        with tensorflow.variable_scope('layer19'):
            weight = tensorflow.get_variable(
                name='weight',
                shape=[7, 7, y.shape.as_list()[-1], 1],
                initializer=tensorflow.initializers.glorot_uniform(),
                regularizer=tensorflow.contrib.layers.l1_l2_regularizer(1e-7, 1e-7))
            bias = tensorflow.get_variable(
                name='bias',
                shape=y.shape.as_list()[-1],
                initializer=tensorflow.initializers.zeros(),
                regularizer=tensorflow.contrib.layers.l1_l2_regularizer(1e-7, 1e-7))
            y = tensorflow.nn.depthwise_conv2d(
                input=y,
                filter=weight,
                strides=[1, 1, 1, 1],
                padding='VALID') + bias
            y = tensorflow.layers.batch_normalization(
                inputs=y,
                beta_regularizer=None,
                gamma_regularizer=None,
                training=training,
                fused=None)

        with tensorflow.variable_scope('layer20'):
            weight = tensorflow.get_variable(
                name='weight',
                shape=[1, 1, y.shape.as_list()[-1], 128],
                initializer=tensorflow.initializers.glorot_uniform(),
                regularizer=tensorflow.contrib.layers.l1_l2_regularizer(1e-7, 1e-7))
            bias = tensorflow.get_variable(
                name='bias',
                shape=128,
                initializer=tensorflow.initializers.zeros(),
                regularizer=tensorflow.contrib.layers.l1_l2_regularizer(1e-7, 1e-7))
            y = tensorflow.nn.conv2d(
                input=y,
                filter=weight,
                strides=[1, 1, 1, 1],
                padding='SAME') + bias
            y = tensorflow.layers.batch_normalization(
                inputs=y,
                beta_regularizer=None,
                gamma_regularizer=None,
                training=training,
                fused=None)
            embed = tensorflow.squeeze(y, [1, 2])

        with tensorflow.variable_scope('layer21'):
            weight = tensorflow.get_variable(
                name='weight',
                shape=[embed.shape.as_list()[-1], num_classes],
                initializer=tensorflow.initializers.glorot_uniform(),
                regularizer=tensorflow.contrib.layers.l1_l2_regularizer(1e-7, 1e-7))
            predict = tensorflow.matmul(embed, weight)
    return predict

if __name__ == '__main__':
    tensorflow.enable_eager_execution(config=tensorflow.ConfigProto(allow_soft_placement=True, gpu_options=tensorflow.GPUOptions(allow_growth=True)))
    data = tensorflow.random.uniform([4, 112, 112, 3])
    predict = MobileFaceNet(data, 10, True)