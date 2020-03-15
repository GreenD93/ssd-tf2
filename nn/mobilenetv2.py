import tensorflow as tf
from tensorflow.keras import Model
import tensorflow.keras.layers as layers

from tensorflow.keras.layers import Conv2D, ZeroPadding2D, AveragePooling2D, DepthwiseConv2D, Flatten, Dense, BatchNormalization, Dropout, ReLU
from tensorflow.keras.models import Sequential

def conv_bn(inp, oup, stride, block_id):
    strides = (stride, stride)

    return Sequential([
        Conv2D(oup, 
               (3, 3), 
               padding='same', 
               use_bias=False,
               strides=strides,
               name='conv_{0}'.format(block_id)),
        BatchNormalization(name='conv_{0}_bn'.format(block_id)),
        ReLU(name='conv_{0}_relu'.format(block_id))
    ])

def conv_1x1_bn(inp, oup, block_id):
    
    return Sequential([
        Conv2D(oup, 
               (1, 1),
               use_bias=False,
               strides=(1, 1)),
        BatchNormalization(name='conv_1x1_{0}_bn'.format(block_id)),
        ReLU(name='conv_1x1_{0}_relu'.format(block_id))
    ])

class InvertedResidual(Model):
    def __init__(self, inp, oup, stride, expand_ratio, block_id):
        super(InvertedResidual, self).__init__()

        # assert는 뒤의 조건이 True가 아니면 AssertError를 발생한다.
        hidden_dim = round(inp * expand_ratio)

        self.stride = (stride, stride)
        self.use_res_connect = self.stride == (1, 1) and inp == oup
        
        if expand_ratio == 1:
            self.conv = Sequential([
                
                    #dw
                    DepthwiseConv2D((3, 3),
                                   padding='same',
                                   use_bias=False,
                                   strides=self.stride,
                                   name='dw_{0}'.format(block_id)),
                    BatchNormalization(name='dw_{0}_bn'.format(block_id)),
                    ReLU(name='dw_{0}_relu'.format(block_id)),
                
                    #pw-linear
                    Conv2D(oup,
                          (1, 1),
                          use_bias=False,
                          strides=self.stride),
                    BatchNormalization(name='pw-linear_{0}_bn'.format(block_id))
                ])
            
        else:
            self.conv = Sequential([
                    #pw
                    Conv2D(hidden_dim,
                          (1, 1),
                          use_bias=False,
                          strides=(1, 1)),
                    BatchNormalization(name='pw_{0}_bn'.format(block_id)),
                    ReLU(name='pw_{0}_relu'.format(block_id)),

                    #dw
                    DepthwiseConv2D((3, 3),
                                   padding='same',
                                   use_bias=False,
                                   strides=self.stride,
                                   name='dw_{0}'.format(block_id)),
                    BatchNormalization(name='dw_{0}_bn'.format(block_id)),
                    ReLU(name='dw_{0}_relu'.format(block_id)),
                
                    #pw-linear
                    Conv2D(oup,
                          (1, 1),
                          use_bias=False,
                          strides=(1, 1)),
                    BatchNormalization(name='pw-linear_{0}_bn'.format(block_id))
                ])
    def call(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        
        else:
            return self.conv(x)
        
class MobilenetV2(Model):
    def __init__(self, num_classes=1000, width_mult=1., dropout_ratio=0.2):
        super(MobilenetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        
        # t: expand_ratio
        # s: stride
        
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        
        input_channel = int(input_channel * width_mult)
        
        self.block_id = 0
        
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2, block_id=self.block_id)]
        
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, stride=s,
                                               expand_ratio=t, block_id=self.block_id))
                else:
                    self.features.append(block(input_channel, output_channel, stride=1,
                                               expand_ratio=t, block_id=self.block_id))
                self.block_id+=1    
                input_channel = output_channel
                                         
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel, block_id=self.block_id))
        self.block_id+=1
        
        self.features = Sequential([*self.features])
        
        self.avg_pool2d = AveragePooling2D(pool_size=(10, 10))
        self.fc = Dense(num_classes)
        self.softmax = layers.Activation('softmax', name='act_softmax')
        
    def call(self, x):
        x = self.features(x)
        x = self.avg_pool2d(x)
        x = tf.reshape(x, [-1, 1280])
        x = self.softmax(self.fc(x))
        return x