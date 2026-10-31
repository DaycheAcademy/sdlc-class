
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K



class ResNet(object):

    @staticmethod
    def residual_module(data, k, stride, chanDim,
                        red=False, bnMom=0.9, bnEps=2e-5, reg=0.0001):

        # Epsilon -> default value in keras = 0.001
        # Epsilon -> default value in pytorch = 0.00001

        # Momentum -> default value in keras = 0.99
        # Momentum -> default value in pytorch = 0.01
        # creating shortcut
        shortcut = data

        bn1 = BatchNormalization(axis=chanDim, momentum=bnMom, epsilon=bnEps)(data)
        act1 = Activation('relu')(bn1)
        # output = (input * kernel) + bias
        conv1 = Conv2D(int(k * 0.25), (1, 1),
                       kernel_regularizer=l2(reg), use_bias=False)(act1)

        bn2 = BatchNormalization(axis=chanDim, momentum=bnMom, epsilon=bnEps)(conv1)
        act2 = Activation('relu')(bn2)
        conv2 = Conv2D(int(k * 0.25), (3, 3), padding='same', strides=stride,
                       kernel_regularizer=l2(reg), use_bias=False)(act2)

        bn3 = BatchNormalization(axis=chanDim, momentum=bnMom, epsilon=bnEps)(conv2)
        act3 = Activation('relu')(bn3)
        conv3 = Conv2D(k, (1, 1),
                       kernel_regularizer=l2(reg), use_bias=False)(act3)

        if red:
            shortcut = Conv2D(k, (1, 1), strides=stride,
                              kernel_regularizer=l2(reg), use_bias=False)(act1)

        x = add([shortcut, conv3])
        return x

    @staticmethod
    def build( width, height, depth, classes, stages, filters,
              bnMom=0.9, bnEps=2e-5, reg=0.0001, dataset='cifar'):
        # stages = (3, 4, 6)
        # filters = (64, 128, 256, 512) -> K

        # Note: always check for : "channels first" or "channels last"

        # we assume "channels last" for data
        inputShape = height, width, depth
        chanDim = -1

        if K.image_data_format() == 'channels_first':
            inputShape = depth, height, width
            chanDim = 1

        inputs = Input(shape=inputShape)
        x = BatchNormalization(axis=chanDim, momentum=bnMom, epsilon=bnEps)(inputs)

        if dataset == 'cifar':
            x = Conv2D(filters[0], (3, 3), use_bias=False,
                       padding='same', kernel_regularizer=l2(reg))(x)


        # (stages and filters)
        # stages = (3, 6, 4)
        # filters = [64, 128, 256, 512]
        # we use the first filter value for the input layer's convolution

        # from there on, we stack stages[i] number of residual modules
        # with filters[k+1] number of filter




        # loop over the number of stage
        for i in range(0, len(stages)):
            stride = (1, 1) if i == 0 else (2, 2)
            x = ResNet.residual_module(x, filters[i+1], stride, chanDim,
                                       red=True, bnMom=bnMom, bnEps=bnEps)

            #
            for j in range(0, stages[i]-1):
                x = ResNet.residual_module(x, filters[i+1], (1, 1), chanDim,
                                           bnMom=bnMom, bnEps=bnEps)

        x = BatchNormalization(axis=chanDim, momentum=bnMom, epsilon=bnEps)(x)
        x = Activation('relu')(x)
        x = AveragePooling2D((8, 8))(x)  # final volume size of ResNet: 8 * 8 * num_filters
        # Note: output dimension of average pooling : 1 * 1 * classes

        # apply classifier:
        x = Flatten()(x)
        x = Dense(classes, kernel_regularizer=l2(reg))(x)
        x = Activation('softmax')(x)

        # creating model
        model = Model(inputs=inputs, outputs=x, name='resnet')
        return model