import tensorflow

bias_initializer = None
conv2d_regularizer = None
depthwise_regularizer = None
alpha_regularizer = None
dense_regularizer = None

def MobileNetV2Block(x, stride, t, output_channel):
    y = tensorflow.keras.layers.Conv2D(
        filters=int(t * x.shape.as_list()[3]),
        kernel_size=1,
        strides=1,
        padding='same',
        bias_initializer=bias_initializer,
        kernel_regularizer=conv2d_regularizer,
        bias_regularizer=conv2d_regularizer)(x)
    y = tensorflow.keras.layers.BatchNormalization(
        beta_regularizer=conv2d_regularizer,
        gamma_regularizer=conv2d_regularizer)(y)
    y = tensorflow.keras.layers.PReLU(
        alpha_initializer=tensorflow.keras.initializers.Constant(0.25),
        alpha_regularizer=alpha_regularizer,
        shared_axes=[1, 2])(y)

    y = tensorflow.keras.layers.DepthwiseConv2D(
        kernel_size=3,
        strides=stride,
        padding='same',
        bias_initializer=bias_initializer,
        kernel_regularizer=depthwise_regularizer,
        bias_regularizer=depthwise_regularizer)(y)
    y = tensorflow.keras.layers.BatchNormalization(
        beta_regularizer=depthwise_regularizer,
        gamma_regularizer=depthwise_regularizer)(y)
    y = tensorflow.keras.layers.PReLU(
        alpha_initializer=tensorflow.keras.initializers.Constant(0.25),
        alpha_regularizer=alpha_regularizer,
        shared_axes=[1, 2])(y)

    y = tensorflow.keras.layers.Conv2D(
        filters=output_channel,
        kernel_size=1,
        strides=1,
        padding='same',
        bias_initializer=bias_initializer,
        kernel_regularizer=conv2d_regularizer,
        bias_regularizer=conv2d_regularizer)(y)
    y = tensorflow.keras.layers.BatchNormalization(
        beta_regularizer=conv2d_regularizer,
        gamma_regularizer=conv2d_regularizer)(y)

    if x.shape.as_list() == y.shape.as_list():
        y = tensorflow.keras.layers.Add()([x, y])
    return y

def MobileFaceNet(num_classes):
    inputs = tensorflow.keras.Input(shape=(112, 112, 3))

    y = tensorflow.keras.layers.Conv2D(
        filters=64,
        kernel_size=3,
        strides=2,
        padding='same',
        bias_initializer=bias_initializer,
        kernel_regularizer=conv2d_regularizer,
        bias_regularizer=conv2d_regularizer)(inputs)
    y = tensorflow.keras.layers.BatchNormalization(
        beta_regularizer=conv2d_regularizer,
        gamma_regularizer=conv2d_regularizer)(y)
    y = tensorflow.keras.layers.PReLU(
        alpha_initializer=tensorflow.keras.initializers.Constant(0.25),
        alpha_regularizer=alpha_regularizer,
        shared_axes=[1, 2])(y)

    y = tensorflow.keras.layers.DepthwiseConv2D(
        kernel_size=3,
        strides=1,
        padding='same',
        bias_initializer=bias_initializer,
        kernel_regularizer=depthwise_regularizer,
        bias_regularizer=depthwise_regularizer)(y)
    y = tensorflow.keras.layers.BatchNormalization(
        beta_regularizer=depthwise_regularizer,
        gamma_regularizer=depthwise_regularizer)(y)
    y = tensorflow.keras.layers.PReLU(
        alpha_initializer=tensorflow.keras.initializers.Constant(0.25),
        alpha_regularizer=alpha_regularizer,
        shared_axes=[1, 2])(y)

    y = MobileNetV2Block(y, 2, 2, 64)
    y = MobileNetV2Block(y, 1, 2, 64)
    y = MobileNetV2Block(y, 1, 2, 64)
    y = MobileNetV2Block(y, 1, 2, 64)
    y = MobileNetV2Block(y, 1, 2, 64)

    y = MobileNetV2Block(y, 2, 4, 128)
    y = MobileNetV2Block(y, 1, 2, 128)
    y = MobileNetV2Block(y, 1, 2, 128)
    y = MobileNetV2Block(y, 1, 2, 128)
    y = MobileNetV2Block(y, 1, 2, 128)
    y = MobileNetV2Block(y, 1, 2, 128)
    y = MobileNetV2Block(y, 1, 2, 128)

    y = MobileNetV2Block(y, 2, 4, 128)

    y = MobileNetV2Block(y, 1, 2, 128)
    y = MobileNetV2Block(y, 1, 2, 128)

    y = tensorflow.keras.layers.Conv2D(
        filters=512,
        kernel_size=1,
        strides=1,
        padding='same',
        bias_initializer=bias_initializer,
        kernel_regularizer=conv2d_regularizer,
        bias_regularizer=conv2d_regularizer)(y)
    y = tensorflow.keras.layers.BatchNormalization(
        beta_regularizer=conv2d_regularizer,
        gamma_regularizer=conv2d_regularizer)(y)
    y = tensorflow.keras.layers.PReLU(
        alpha_initializer=tensorflow.keras.initializers.Constant(0.25),
        alpha_regularizer=alpha_regularizer,
        shared_axes=[1, 2])(y)

    y = tensorflow.keras.layers.DepthwiseConv2D(
        kernel_size=7,
        strides=1,
        bias_initializer=bias_initializer,
        kernel_regularizer=depthwise_regularizer,
        bias_regularizer=depthwise_regularizer)(y)
    y = tensorflow.keras.layers.BatchNormalization(
        beta_regularizer=depthwise_regularizer,
        gamma_regularizer=depthwise_regularizer)(y)

    y = tensorflow.keras.layers.Conv2D(
        filters=128,
        kernel_size=1,
        strides=1,
        bias_initializer=bias_initializer,
        kernel_regularizer=conv2d_regularizer,
        bias_regularizer=conv2d_regularizer)(y)
    y = tensorflow.keras.layers.BatchNormalization(
        beta_regularizer=conv2d_regularizer,
        gamma_regularizer=conv2d_regularizer)(y)
    embed = tensorflow.keras.layers.Flatten()(y)

    predict = tensorflow.keras.layers.Dense(
        units=num_classes,
        use_bias=False,
        name='last_layer',
        kernel_regularizer=dense_regularizer)(embed)

    model = tensorflow.keras.Model(inputs=inputs, outputs=[embed, predict])
    return model

if __name__ == '__main__':
    tensorflow.enable_eager_execution(config=tensorflow.ConfigProto(allow_soft_placement=True, gpu_options=tensorflow.GPUOptions(allow_growth=True)))
    data = tensorflow.random.uniform([4, 112, 112, 3])
    model = MobileFaceNet(10)
    embed = model(data)