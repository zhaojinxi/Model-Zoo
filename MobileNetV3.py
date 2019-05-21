import tensorflow

def swish(x):
    y = x * tensorflow.nn.sigmoid(x)
    return y

def hard_swish(x):
    y = x * (tensorflow.nn.relu6(x + 3)) / 6
    return y

class MobileNetV3Block(tensorflow.keras.layers.Layer):
    def __init__(self, input_shape, kernel_size, exp_size, output_shape, use_se, activation, stride):
        super(MobileNetV3Block, self).__init__()

        self.in_shape = input_shape
        self.stride = stride
        self.kernel_size = kernel_size
        self.exp_size = exp_size
        self.use_se = use_se
        self.out_shape = output_shape
        if activation == 're':
            self.active = tensorflow.nn.relu6
        elif activation == 'hs':
            self.active = hard_swish

        self.Conv2D_1 = tensorflow.keras.layers.Conv2D(
            filters=self.exp_size,
            kernel_size=1,
            strides=1,
            padding='same')
        self.BatchNormalization_1 = tensorflow.keras.layers.BatchNormalization()

        self.DepthwiseConv2D_2 = tensorflow.keras.layers.DepthwiseConv2D(
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding='same')
        self.BatchNormalization_2 = tensorflow.keras.layers.BatchNormalization()

        if self.use_se == True:
            self.AveragePooling2D_3 = tensorflow.keras.layers.AveragePooling2D(
                pool_size=[x // self.stride for x in self.in_shape[:2]],
                strides=1)

            self.Conv2D_4 = tensorflow.keras.layers.Conv2D(
                filters=self.exp_size // 4,
                kernel_size=1,
                strides=1,
                activation=tensorflow.nn.relu)

            self.Conv2D_5 = tensorflow.keras.layers.Conv2D(
                filters=self.exp_size,
                kernel_size=1,
                strides=1,
                activation=tensorflow.keras.activations.hard_sigmoid)

        self.Conv2D_6 = tensorflow.keras.layers.Conv2D(
            filters=self.out_shape[2],
            kernel_size=1,
            strides=1,
            padding='same')
        self.BatchNormalization_6 = tensorflow.keras.layers.BatchNormalization()

    def call(self, x):
        y = self.Conv2D_1(x)
        y = self.BatchNormalization_1(y)
        y = self.active(y)

        y = self.DepthwiseConv2D_2(y)
        y = self.BatchNormalization_2(y)
        y = self.active(y)

        if self.use_se == True:
            z = self.AveragePooling2D_3(y)

            z = self.Conv2D_4(z)

            z = self.Conv2D_5(z)

            y = tensorflow.multiply(y, z)

        y = self.Conv2D_6(y)
        y = self.BatchNormalization_6(y)

        if self.in_shape == self.out_shape:
            y = y + x
        return y

class MobileNetV3Large(tensorflow.keras.Model):
    def __init__(self, num_classes=1000):
        super(MobileNetV3Large, self).__init__(name='MobileNetV3Large')

        self.Conv2D_1 = tensorflow.keras.layers.Conv2D(
            filters=16,
            kernel_size=3,
            strides=2,
            padding='same')
        self.BatchNormalization_1 = tensorflow.keras.layers.BatchNormalization()
        self.hard_swish_1 = hard_swish

        self.MobileNetV3Block_2 = MobileNetV3Block([112, 112, 16], 3, 16, [112, 112, 16], False, 're', 1)

        self.MobileNetV3Block_3 = MobileNetV3Block([112, 112, 16], 3, 64, [56, 56, 24], False, 're', 2)

        self.MobileNetV3Block_4 = MobileNetV3Block([56, 56, 24], 3, 72, [56, 56, 24], False, 're', 1)

        self.MobileNetV3Block_5 = MobileNetV3Block([56, 56, 24], 5, 72, [28, 28, 40], True, 're', 2)

        self.MobileNetV3Block_6 = MobileNetV3Block([28, 28, 40], 5, 120, [28, 28, 40], True, 're', 1)

        self.MobileNetV3Block_7 = MobileNetV3Block([28, 28, 40], 5, 120, [28, 28, 40], True, 're', 1)

        self.MobileNetV3Block_8 = MobileNetV3Block([28, 28, 40], 3, 240, [14, 14, 80], False, 'hs', 2)

        self.MobileNetV3Block_9 = MobileNetV3Block([14, 14, 80], 3, 200, [14, 14, 80], False, 'hs', 1)

        self.MobileNetV3Block_10 = MobileNetV3Block([14, 14, 80], 3, 184, [14, 14, 80], False, 'hs', 1)

        self.MobileNetV3Block_11 = MobileNetV3Block([14, 14, 80], 3, 184, [14, 14, 80], False, 'hs', 1)

        self.MobileNetV3Block_12 = MobileNetV3Block([14, 14, 80], 3, 480, [14, 14, 112], True, 'hs', 1)

        self.MobileNetV3Block_13 = MobileNetV3Block([14, 14, 112], 3, 672, [14, 14, 112], True, 'hs', 1)

        self.MobileNetV3Block_14 = MobileNetV3Block([14, 14, 112], 5, 672, [7, 7, 160], True, 'hs', 2)

        self.MobileNetV3Block_15 = MobileNetV3Block([7, 7, 160], 5, 960, [7, 7, 160], True, 'hs', 1)

        self.MobileNetV3Block_16 = MobileNetV3Block([7, 7, 160], 5, 960, [7, 7, 160], True, 'hs', 1)

        self.Conv2D_17 = tensorflow.keras.layers.Conv2D(
            filters=960,
            kernel_size=1,
            strides=1,
            padding='same')
        self.BatchNormalization_17 = tensorflow.keras.layers.BatchNormalization()
        self.hard_swish_17 = hard_swish

        self.AveragePooling2D_18 = tensorflow.keras.layers.AveragePooling2D(
            pool_size=7,
            strides=1)

        self.Conv2D_19 = tensorflow.keras.layers.Conv2D(
            filters=1280,
            kernel_size=1,
            strides=1,
            padding='same',
            activation=hard_swish)

        self.Conv2D_20 = tensorflow.keras.layers.Conv2D(
            filters=num_classes,
            kernel_size=1,
            strides=1,
            padding='same')

    def call(self, x):
        x = self.Conv2D_1(x)
        x = self.BatchNormalization_1(x)
        x = self.hard_swish_1(x)

        x = self.MobileNetV3Block_2(x)

        x = self.MobileNetV3Block_3(x)

        x = self.MobileNetV3Block_4(x)

        x = self.MobileNetV3Block_5(x)

        x = self.MobileNetV3Block_6(x)

        x = self.MobileNetV3Block_7(x)

        x = self.MobileNetV3Block_8(x)

        x = self.MobileNetV3Block_9(x)

        x = self.MobileNetV3Block_10(x)

        x = self.MobileNetV3Block_11(x)

        x = self.MobileNetV3Block_12(x)

        x = self.MobileNetV3Block_13(x)

        x = self.MobileNetV3Block_14(x)

        x = self.MobileNetV3Block_15(x)

        x = self.MobileNetV3Block_16(x)

        x = self.Conv2D_17(x)
        x = self.BatchNormalization_17(x)
        x = self.hard_swish_17(x)

        x = self.AveragePooling2D_18(x)

        x = self.Conv2D_19(x)

        x = self.Conv2D_20(x)
        return x

class MobileNetV3Small(tensorflow.keras.Model):
    def __init__(self, num_classes=1000):
        super(MobileNetV3Small, self).__init__(name='MobileNetV3Small')

        self.Conv2D_1 = tensorflow.keras.layers.Conv2D(
            filters=16,
            kernel_size=3,
            strides=2,
            padding='same')
        self.BatchNormalization_1 = tensorflow.keras.layers.BatchNormalization()
        self.hard_swish_1 = hard_swish

        self.MobileNetV3Block_2 = MobileNetV3Block([112, 112, 16], 3, 16, [56, 56, 16], True, 're', 2)

        self.MobileNetV3Block_3 = MobileNetV3Block([56, 56, 16], 3, 72, [28, 28, 24], False, 're', 2)

        self.MobileNetV3Block_4 = MobileNetV3Block([28, 28, 24], 3, 88, [28, 28, 24], False, 're', 1)

        self.MobileNetV3Block_5 = MobileNetV3Block([28, 28, 24], 5, 96, [14, 14, 40], True, 'hs', 2)

        self.MobileNetV3Block_6 = MobileNetV3Block([14, 14, 40], 5, 240, [14, 14, 40], True, 'hs', 1)

        self.MobileNetV3Block_7 = MobileNetV3Block([14, 14, 40], 5, 240, [14, 14, 40], True, 'hs', 1)

        self.MobileNetV3Block_8 = MobileNetV3Block([14, 14, 40], 5, 120, [14, 14, 48], True, 'hs', 1)

        self.MobileNetV3Block_9 = MobileNetV3Block([14, 14, 48], 5, 144, [14, 14, 48], True, 'hs', 1)

        self.MobileNetV3Block_10 = MobileNetV3Block([14, 14, 48], 5, 288, [7, 7, 96], True, 'hs', 2)

        self.MobileNetV3Block_11 = MobileNetV3Block([7, 7, 96], 5, 576, [7, 7, 96], True, 'hs', 1)

        self.MobileNetV3Block_12 = MobileNetV3Block([7, 7, 96], 5, 576, [7, 7, 96], True, 'hs', 1)

        self.Conv2D_13 = tensorflow.keras.layers.Conv2D(
            filters=576,
            kernel_size=1,
            strides=1,
            padding='same')
        self.BatchNormalization_13 = tensorflow.keras.layers.BatchNormalization()
        self.hard_swish_13 = hard_swish

        self.AveragePooling2D_14 = tensorflow.keras.layers.AveragePooling2D(
            pool_size=7,
            strides=1)

        self.Conv2D_15 = tensorflow.keras.layers.Conv2D(
            filters=1280,
            kernel_size=1,
            strides=1,
            padding='same',
            activation=hard_swish)

        self.Conv2D_16 = tensorflow.keras.layers.Conv2D(
            filters=num_classes,
            kernel_size=1,
            strides=1,
            padding='same')

    def call(self, x):
        x = self.Conv2D_1(x)
        x = self.BatchNormalization_1(x)
        x = self.hard_swish_1(x)

        x = self.MobileNetV3Block_2(x)

        x = self.MobileNetV3Block_3(x)

        x = self.MobileNetV3Block_4(x)

        x = self.MobileNetV3Block_5(x)

        x = self.MobileNetV3Block_6(x)

        x = self.MobileNetV3Block_7(x)

        x = self.MobileNetV3Block_8(x)

        x = self.MobileNetV3Block_9(x)

        x = self.MobileNetV3Block_10(x)

        x = self.MobileNetV3Block_11(x)

        x = self.MobileNetV3Block_12(x)

        x = self.Conv2D_13(x)
        x = self.BatchNormalization_13(x)
        x = self.hard_swish_13(x)

        x = self.AveragePooling2D_14(x)

        x = self.Conv2D_15(x)

        x = self.Conv2D_16(x)
        return x

if __name__ == '__main__':
    tensorflow.enable_eager_execution()
    data = tensorflow.random.uniform([128, 224, 224, 3])
    model = MobileNetV3Small()
    y = model(data)