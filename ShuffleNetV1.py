import tensorflow

bias_initializer = tensorflow.keras.initializers.glorot_uniform()
conv2d_regularizer = None
depthwise_regularizer = None
alpha_regularizer = None
dense_regularizer = None

class ShuffleNetV1Block(tensorflow.keras.Model):
    def __init__(self, stride, scale, group, output_channel):
        super(ShuffleNetV1Block, self).__init__()
        self.stride = stride
        self.scale = scale
        self.group = group
        self.output_channel = output_channel

    def build(self, input_shape):
        if self.stride == 1:
            self.per_outter_channel = self.output_channel // self.group
        else:
            self.per_outter_channel = (self.output_channel - input_shape.as_list()[3]) // self.group
        self.per_inner_channel = self.per_outter_channel // self.scale

        for i in range(self.group):
            Conv2D = tensorflow.keras.layers.Conv2D(
                filters=int(self.per_inner_channel),
                kernel_size=1,
                strides=1,
                padding='same',
                bias_initializer=bias_initializer,
                kernel_regularizer=conv2d_regularizer,
                bias_regularizer=conv2d_regularizer)
            exec("self.Conv2D_1%s=Conv2D" % i)
        self.BatchNormalization_1 = tensorflow.keras.layers.BatchNormalization(
            beta_regularizer=conv2d_regularizer,
            gamma_regularizer=conv2d_regularizer)
        self.ReLU_1 = tensorflow.keras.layers.ReLU()

        self.DepthwiseConv2D_2 = tensorflow.keras.layers.DepthwiseConv2D(
            kernel_size=3,
            strides=self.stride,
            padding='same',
            bias_initializer=bias_initializer,
            kernel_regularizer=depthwise_regularizer,
            bias_regularizer=depthwise_regularizer)
        self.BatchNormalization_2 = tensorflow.keras.layers.BatchNormalization(
            beta_regularizer=depthwise_regularizer,
            gamma_regularizer=depthwise_regularizer)

        for i in range(self.group):
            Conv2D = tensorflow.keras.layers.Conv2D(
                filters=int(self.per_outter_channel),
                kernel_size=1,
                strides=1,
                padding='same',
                bias_initializer=bias_initializer,
                kernel_regularizer=conv2d_regularizer,
                bias_regularizer=conv2d_regularizer)
            exec("self.Conv2D_3%s=Conv2D" % i)
        self.BatchNormalization_3 = tensorflow.keras.layers.BatchNormalization(
            beta_regularizer=conv2d_regularizer,
            gamma_regularizer=conv2d_regularizer)

        if self.stride != 1:
            self.AveragePooling2D_4 = tensorflow.keras.layers.AveragePooling2D(
                pool_size=3,
                strides=self.stride,
                padding='same')

        self.ReLU_5 = tensorflow.keras.layers.ReLU()

    def call(self, x, training):
        s = tensorflow.split(x, self.group, 3)
        tt = []
        for i, one_group in enumerate(s):
            temp = eval('self.Conv2D_1%s(one_group)' % i)
            tt.append(temp)
        y = tensorflow.concat(tt, 3)
        y = self.BatchNormalization_1(y, training=training)
        y = self.ReLU_1(y)

        temp = y.shape.as_list()
        shuffle_channel = [temp[0], temp[1], temp[2], self.group, self.per_inner_channel]
        temp = tensorflow.reshape(y, shuffle_channel)
        temp = tensorflow.transpose(temp, [0, 1, 2, 4, 3])
        y = tensorflow.reshape(temp, y.shape.as_list())

        y = self.DepthwiseConv2D_2(y)
        y = self.BatchNormalization_2(y, training=training)

        s = tensorflow.split(y, self.group, 3)
        tt = []
        for i, one_group in enumerate(s):
            temp = eval('self.Conv2D_3%s(one_group)' % i)
            tt.append(temp)
        y = tensorflow.concat(tt, 3)
        y = self.BatchNormalization_3(y, training=training)

        if self.stride != 1:
            y1 = self.AveragePooling2D_4(x)
            y = tensorflow.concat([y, y1], 3)
        else:
            y = x + y

        y = self.ReLU_5(y)
        return y

class ShuffleNetV1(tensorflow.keras.Model):
    def __init__(self, num_classes):
        super(ShuffleNetV1, self).__init__(name='ShuffleNetV1')

        self.Conv2D_1 = tensorflow.keras.layers.Conv2D(
            filters=32,
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

        self.ShuffleNetV1Block_2 = ShuffleNetV1Block(2, 4, 4, 272)
        self.ShuffleNetV1Block_3 = ShuffleNetV1Block(1, 4, 4, 272)
        self.ShuffleNetV1Block_4 = ShuffleNetV1Block(1, 4, 4, 272)
        self.ShuffleNetV1Block_5 = ShuffleNetV1Block(1, 4, 4, 272)

        self.ShuffleNetV1Block_6 = ShuffleNetV1Block(2, 4, 4, 544)
        self.ShuffleNetV1Block_7 = ShuffleNetV1Block(1, 4, 4, 544)
        self.ShuffleNetV1Block_8 = ShuffleNetV1Block(1, 4, 4, 544)
        self.ShuffleNetV1Block_9 = ShuffleNetV1Block(1, 4, 4, 544)
        self.ShuffleNetV1Block_10 = ShuffleNetV1Block(1, 4, 4, 544)
        self.ShuffleNetV1Block_11 = ShuffleNetV1Block(1, 4, 4, 544)
        self.ShuffleNetV1Block_12 = ShuffleNetV1Block(1, 4, 4, 544)
        self.ShuffleNetV1Block_13 = ShuffleNetV1Block(1, 4, 4, 544)

        self.ShuffleNetV1Block_14 = ShuffleNetV1Block(2, 4, 4, 1088)
        self.ShuffleNetV1Block_15 = ShuffleNetV1Block(1, 4, 4, 1088)
        self.ShuffleNetV1Block_16 = ShuffleNetV1Block(1, 4, 4, 1088)
        self.ShuffleNetV1Block_17 = ShuffleNetV1Block(1, 4, 4, 1088)

        self.AveragePooling2D_18 = tensorflow.keras.layers.AveragePooling2D(
            pool_size=7,
            strides=1)

        self.Dense_19 = tensorflow.keras.layers.Dense(
            units=num_classes,
            use_bias=False,
            kernel_regularizer=dense_regularizer)

    def call(self, x, training):
        x = self.Conv2D_1(x)
        x = self.MaxPool2D_1(x)

        x = self.ShuffleNetV1Block_2(x, training=training)
        x = self.ShuffleNetV1Block_3(x, training=training)
        x = self.ShuffleNetV1Block_4(x, training=training)
        x = self.ShuffleNetV1Block_5(x, training=training)

        x = self.ShuffleNetV1Block_6(x, training=training)
        x = self.ShuffleNetV1Block_7(x, training=training)
        x = self.ShuffleNetV1Block_8(x, training=training)
        x = self.ShuffleNetV1Block_9(x, training=training)
        x = self.ShuffleNetV1Block_10(x, training=training)
        x = self.ShuffleNetV1Block_11(x, training=training)
        x = self.ShuffleNetV1Block_12(x, training=training)
        x = self.ShuffleNetV1Block_13(x, training=training)

        x = self.ShuffleNetV1Block_14(x, training=training)
        x = self.ShuffleNetV1Block_15(x, training=training)
        x = self.ShuffleNetV1Block_16(x, training=training)
        x = self.ShuffleNetV1Block_17(x, training=training)

        x = self.AveragePooling2D_18(x)
        x = tensorflow.squeeze(x, [1, 2])

        x = self.Dense_19(x)
        return x

if __name__ == '__main__':
    tensorflow.enable_eager_execution(config=tensorflow.ConfigProto(allow_soft_placement=True, gpu_options=tensorflow.GPUOptions(allow_growth=True)))
    data = tensorflow.random.uniform([64, 224, 224, 3])
    model = ShuffleNetV1(1000)
    y = model(data, True)