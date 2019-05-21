import tensorflow

class MobileNetV2Block(tensorflow.keras.layers.Layer):
    def __init__(self, stride, t, input_shape, output_shape):

        super(MobileNetV2Block, self).__init__()
        self.stride = stride
        self.t = t
        self.in_shape = input_shape
        self.out_shape = output_shape

        self.Conv2D_1 = tensorflow.keras.layers.Conv2D(
            filters=self.t * self.in_shape[2],
            kernel_size=1,
            strides=1,
            padding='same')
        self.BatchNormalization_1 = tensorflow.keras.layers.BatchNormalization()
        self.PReLU_1 = tensorflow.keras.layers.PReLU()

        self.DepthwiseConv2D_2 = tensorflow.keras.layers.DepthwiseConv2D(
            kernel_size=3,
            strides=self.stride,
            padding='same',
            activation=tensorflow.keras.layers.ReLU(max_value=6))
        self.BatchNormalization_2 = tensorflow.keras.layers.BatchNormalization()
        self.PReLU_2 = tensorflow.keras.layers.PReLU()

        self.Conv2D_3 = tensorflow.keras.layers.Conv2D(
            filters=self.out_shape[2],
            kernel_size=1,
            strides=1,
            padding='same')
        self.BatchNormalization_3 = tensorflow.keras.layers.BatchNormalization()

    def call(self, x):
        y = self.Conv2D_1(x)
        y = self.BatchNormalization_1(y)
        y = self.PReLU_1(y)

        y = self.DepthwiseConv2D_2(y)
        y = self.BatchNormalization_2(y)
        y = self.PReLU_2(y)

        y = self.Conv2D_3(y)
        y = self.BatchNormalization_3(y)

        if self.in_shape == self.out_shape:
            y = x + y
        return y

class MobileFaceNet(tensorflow.keras.Model):
    def __init__(self, num_classes=1000):
        super(MobileFaceNet, self).__init__(name='MobileFaceNet')

        self.Conv2D_1 = tensorflow.keras.layers.Conv2D(
            filters=64,
            kernel_size=3,
            strides=2,
            padding='same')
        self.BatchNormalization_1 = tensorflow.keras.layers.BatchNormalization()
        self.PReLU_1 = tensorflow.keras.layers.PReLU()

        self.DepthwiseConv2D_2 = tensorflow.keras.layers.DepthwiseConv2D(
            kernel_size=3,
            strides=1,
            padding='same')
        self.BatchNormalization_2 = tensorflow.keras.layers.BatchNormalization()
        self.PReLU_2 = tensorflow.keras.layers.PReLU()

        self.MobileNetV2Block_3 = MobileNetV2Block(2, 2, [56, 56, 64], [28, 28, 64])
        self.MobileNetV2Block_4 = MobileNetV2Block(1, 2, [28, 28, 64], [28, 28, 64])
        self.MobileNetV2Block_5 = MobileNetV2Block(1, 2, [28, 28, 64], [28, 28, 64])
        self.MobileNetV2Block_6 = MobileNetV2Block(1, 2, [28, 28, 64], [28, 28, 64])
        self.MobileNetV2Block_7 = MobileNetV2Block(1, 2, [28, 28, 64], [28, 28, 64])

        self.MobileNetV2Block_8 = MobileNetV2Block(2, 4, [28, 28, 64], [14, 14, 128])

        self.MobileNetV2Block_9 = MobileNetV2Block(1, 2, [14, 14, 128], [14, 14, 128])
        self.MobileNetV2Block_10 = MobileNetV2Block(1, 2, [14, 14, 128], [14, 14, 128])
        self.MobileNetV2Block_11 = MobileNetV2Block(1, 2, [14, 14, 128], [14, 14, 128])
        self.MobileNetV2Block_12 = MobileNetV2Block(1, 2, [14, 14, 128], [14, 14, 128])
        self.MobileNetV2Block_13 = MobileNetV2Block(1, 2, [14, 14, 128], [14, 14, 128])
        self.MobileNetV2Block_14 = MobileNetV2Block(1, 2, [14, 14, 128], [14, 14, 128])

        self.MobileNetV2Block_15 = MobileNetV2Block(2, 4, [14, 14, 128], [7, 7, 128])

        self.MobileNetV2Block_16 = MobileNetV2Block(1, 2, [7, 7, 128], [7, 7, 128])
        self.MobileNetV2Block_17 = MobileNetV2Block(1, 2, [7, 7, 128], [7, 7, 128])

        self.Conv2D_18 = tensorflow.keras.layers.Conv2D(
            filters=512,
            kernel_size=1,
            strides=1,
            padding='same')
        self.BatchNormalization_18 = tensorflow.keras.layers.BatchNormalization()
        self.PReLU_18 = tensorflow.keras.layers.PReLU()

        self.DepthwiseConv2D_19 = tensorflow.keras.layers.DepthwiseConv2D(
            kernel_size=7,
            strides=1)
        self.BatchNormalization_19 = tensorflow.keras.layers.BatchNormalization()

        self.Conv2D_20 = tensorflow.keras.layers.Conv2D(
            filters=128,
            kernel_size=1,
            strides=1,
            padding='same')
        self.BatchNormalization_20 = tensorflow.keras.layers.BatchNormalization()

        self.Conv2D_21 = tensorflow.keras.layers.Conv2D(
            filters=num_classes,
            kernel_size=1,
            strides=1,
            padding='same')

    def call(self, x):
        x = self.Conv2D_1(x)
        x = self.BatchNormalization_1(x)
        x = self.PReLU_1(x)

        x = self.DepthwiseConv2D_2(x)
        x = self.BatchNormalization_2(x)
        x = self.PReLU_2(x)

        x = self.MobileNetV2Block_3(x)
        x = self.MobileNetV2Block_4(x)
        x = self.MobileNetV2Block_5(x)
        x = self.MobileNetV2Block_6(x)
        x = self.MobileNetV2Block_7(x)

        x = self.MobileNetV2Block_8(x)

        x = self.MobileNetV2Block_9(x)
        x = self.MobileNetV2Block_10(x)
        x = self.MobileNetV2Block_11(x)
        x = self.MobileNetV2Block_12(x)
        x = self.MobileNetV2Block_13(x)
        x = self.MobileNetV2Block_14(x)

        x = self.MobileNetV2Block_15(x)

        x = self.MobileNetV2Block_16(x)
        x = self.MobileNetV2Block_17(x)

        x = self.Conv2D_18(x)
        x = self.BatchNormalization_18(x)
        x = self.PReLU_18(x)

        x = self.DepthwiseConv2D_19(x)
        x = self.BatchNormalization_19(x)

        x = self.Conv2D_20(x)
        x = self.BatchNormalization_20(x)

        x = self.Conv2D_21(x)
        return x

if __name__ == '__main__':
    tensorflow.enable_eager_execution()
    data = tensorflow.random.uniform([128, 112, 112, 3])
    model = MobileFaceNet()
    y = model(data)