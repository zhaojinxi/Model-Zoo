import tensorflow

bias_initializer = tensorflow.keras.initializers.glorot_uniform()
conv2d_regularizer = None
depthwise_regularizer = None
alpha_regularizer = None
dense_regularizer = None

class ShuffleNetV2Block(tensorflow.keras.Model):
    def __init__(self, stride, output_channel):
        super(ShuffleNetV2Block, self).__init__()
        self.stride = stride
        self.output_channel = output_channel
        self.inner_channel = self.output_channel // 2

    def build(self, input_shape):
        if self.stride != 1:
            self.DepthwiseConv2D_11 = tensorflow.keras.layers.DepthwiseConv2D(
                kernel_size=3,
                strides=self.stride,
                padding='same',
                bias_initializer=bias_initializer,
                kernel_regularizer=depthwise_regularizer,
                bias_regularizer=depthwise_regularizer)
            self.BatchNormalization_11 = tensorflow.keras.layers.BatchNormalization(
                beta_regularizer=depthwise_regularizer,
                gamma_regularizer=depthwise_regularizer)

            self.Conv2D_12 = tensorflow.keras.layers.Conv2D(
                filters=self.inner_channel,
                kernel_size=1,
                strides=1,
                padding='same',
                bias_initializer=bias_initializer,
                kernel_regularizer=conv2d_regularizer,
                bias_regularizer=conv2d_regularizer)
            self.BatchNormalization_12 = tensorflow.keras.layers.BatchNormalization(
                beta_regularizer=conv2d_regularizer,
                gamma_regularizer=conv2d_regularizer)
            self.ReLU_12 = tensorflow.keras.layers.ReLU()

        self.Conv2D_21 = tensorflow.keras.layers.Conv2D(
            filters=self.inner_channel,
            kernel_size=1,
            strides=1,
            padding='same',
            bias_initializer=bias_initializer,
            kernel_regularizer=conv2d_regularizer,
            bias_regularizer=conv2d_regularizer)
        self.BatchNormalization_21 = tensorflow.keras.layers.BatchNormalization(
            beta_regularizer=conv2d_regularizer,
            gamma_regularizer=conv2d_regularizer)
        self.ReLU_21 = tensorflow.keras.layers.ReLU()

        self.DepthwiseConv2D_22 = tensorflow.keras.layers.DepthwiseConv2D(
            kernel_size=3,
            strides=self.stride,
            padding='same',
            bias_initializer=bias_initializer,
            kernel_regularizer=depthwise_regularizer,
            bias_regularizer=depthwise_regularizer)
        self.BatchNormalization_22 = tensorflow.keras.layers.BatchNormalization(
            beta_regularizer=depthwise_regularizer,
            gamma_regularizer=depthwise_regularizer)

        self.Conv2D_23 = tensorflow.keras.layers.Conv2D(
            filters=self.inner_channel,
            kernel_size=1,
            strides=1,
            padding='same',
            bias_initializer=bias_initializer,
            kernel_regularizer=conv2d_regularizer,
            bias_regularizer=conv2d_regularizer)
        self.BatchNormalization_23 = tensorflow.keras.layers.BatchNormalization(
            beta_regularizer=conv2d_regularizer,
            gamma_regularizer=conv2d_regularizer)
        self.ReLU_23 = tensorflow.keras.layers.ReLU()

    def call(self, x, training):
        if self.stride != 1:
            x2 = x

            x1 = self.DepthwiseConv2D_11(x)
            x1 = self.BatchNormalization_11(x1)

            x1 = self.Conv2D_12(x1)
            x1 = self.BatchNormalization_12(x1)
            x1 = self.ReLU_12(x1)
        else:
            x1 = x[:, :, :, : self.inner_channel]
            x2 = x[:, :, :, self.inner_channel :]

        x2 = self.Conv2D_21(x2)
        x2 = self.BatchNormalization_21(x2)
        x2 = self.ReLU_21(x2)

        x2 = self.DepthwiseConv2D_22(x2)
        x2 = self.BatchNormalization_22(x2)

        x2 = self.Conv2D_23(x2)
        x2 = self.BatchNormalization_23(x2)
        x2 = self.ReLU_23(x2)

        y = tensorflow.concat([x1, x2], 3)

        temp = y.shape.as_list()
        temp = tensorflow.reshape(y, [temp[0], temp[1], temp[2], 4, -1])
        temp = tensorflow.transpose(temp, [0, 1, 2, 4, 3])
        y = tensorflow.reshape(temp, y.shape.as_list())
        return y

class ShuffleNetV2(tensorflow.keras.Model):
    def __init__(self, num_classes):
        super(ShuffleNetV2, self).__init__(name='ShuffleNetV2')

        self.Conv2D_1 = tensorflow.keras.layers.Conv2D(
            filters=24,
            kernel_size=3,
            strides=2,
            padding='same',
            bias_initializer=bias_initializer,
            kernel_regularizer=conv2d_regularizer,
            bias_regularizer=conv2d_regularizer)
        self.MaxPool2D_1 = tensorflow.keras.layers.MaxPool2D(
            pool_size=3,
            strides=2,
            padding='same')

        self.ShuffleNetV2Block_2 = ShuffleNetV2Block(2, 116)
        self.ShuffleNetV2Block_3 = ShuffleNetV2Block(1, 116)
        self.ShuffleNetV2Block_4 = ShuffleNetV2Block(1, 116)
        self.ShuffleNetV2Block_5 = ShuffleNetV2Block(1, 116)

        self.ShuffleNetV2Block_6 = ShuffleNetV2Block(2, 232)
        self.ShuffleNetV2Block_7 = ShuffleNetV2Block(1, 232)
        self.ShuffleNetV2Block_8 = ShuffleNetV2Block(1, 232)
        self.ShuffleNetV2Block_9 = ShuffleNetV2Block(1, 232)
        self.ShuffleNetV2Block_10 = ShuffleNetV2Block(1, 232)
        self.ShuffleNetV2Block_11 = ShuffleNetV2Block(1, 232)
        self.ShuffleNetV2Block_12 = ShuffleNetV2Block(1, 232)
        self.ShuffleNetV2Block_13 = ShuffleNetV2Block(1, 232)

        self.ShuffleNetV2Block_14 = ShuffleNetV2Block(2, 464)
        self.ShuffleNetV2Block_15 = ShuffleNetV2Block(1, 464)
        self.ShuffleNetV2Block_16 = ShuffleNetV2Block(1, 464)
        self.ShuffleNetV2Block_17 = ShuffleNetV2Block(1, 464)

        self.Conv2D_18 = tensorflow.keras.layers.Conv2D(
            filters=1024,
            kernel_size=1,
            strides=1,
            padding='same',
            bias_initializer=bias_initializer,
            kernel_regularizer=conv2d_regularizer,
            bias_regularizer=conv2d_regularizer)

        self.AveragePooling2D_19 = tensorflow.keras.layers.AveragePooling2D(
            pool_size=7,
            strides=1)

        self.Dense_20 = tensorflow.keras.layers.Dense(
            units=num_classes,
            use_bias=False,
            kernel_regularizer=dense_regularizer)

    def call(self, x, training):
        x = self.Conv2D_1(x)
        x = self.MaxPool2D_1(x)

        x = self.ShuffleNetV2Block_2(x, training=training)
        x = self.ShuffleNetV2Block_3(x, training=training)
        x = self.ShuffleNetV2Block_4(x, training=training)
        x = self.ShuffleNetV2Block_5(x, training=training)

        x = self.ShuffleNetV2Block_6(x, training=training)
        x = self.ShuffleNetV2Block_7(x, training=training)
        x = self.ShuffleNetV2Block_8(x, training=training)
        x = self.ShuffleNetV2Block_9(x, training=training)
        x = self.ShuffleNetV2Block_10(x, training=training)
        x = self.ShuffleNetV2Block_11(x, training=training)
        x = self.ShuffleNetV2Block_12(x, training=training)
        x = self.ShuffleNetV2Block_13(x, training=training)

        x = self.ShuffleNetV2Block_14(x, training=training)
        x = self.ShuffleNetV2Block_15(x, training=training)
        x = self.ShuffleNetV2Block_16(x, training=training)
        x = self.ShuffleNetV2Block_17(x, training=training)

        x = self.Conv2D_18(x)

        x = self.AveragePooling2D_19(x)
        x = tensorflow.squeeze(x, [1, 2])

        x = self.Dense_20(x)
        return x

if __name__ == '__main__':
    tensorflow.enable_eager_execution(config=tensorflow.ConfigProto(allow_soft_placement=True, gpu_options=tensorflow.GPUOptions(allow_growth=True)))
    data = tensorflow.random.uniform([64, 224, 224, 3])
    model = ShuffleNetV2(1000)
    y = model(data, True)