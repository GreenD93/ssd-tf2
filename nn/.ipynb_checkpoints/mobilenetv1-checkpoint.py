import tensorflow as tf

from tensorflow.keras import Model
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, AveragePooling2D, DepthwiseConv2D, Flatten, Dense, BatchNormalization, Dropout, ReLU
from tensorflow.keras.models import Sequential

class MobilenetV1(Model):
    
    def __init__(self, num_classes=1024):
        super(MobilenetV1, self).__init__()
        
        def conv_bn(inp, oup, stride, block_id):
            strides = (stride, stride)
            
            return Sequential([
                Conv2D(oup, (3, 3), 
                       padding='same', 
                       use_bias=False,
                       strides=strides,
                       name='conv_{0}'.format(block_id)),
                BatchNormalization(name='conv_{0}_bn'.format(block_id)),
                ReLU(name='conv_{0}_relu'.format(block_id))
            ])
            
        def conv_dw(inp, oup, stride, block_id):
            strides = (stride, stride)
            
            return Sequential([
                # dw
                DepthwiseConv2D((3, 3), 
                                padding='same',
                                strides=strides,
                                use_bias=False,
                                name='conv_dw_{0}'.format(block_id)),
                BatchNormalization(name='conv_dw_{0}_bn'.format(block_id)),
                ReLU(name='conv_dw_{0}_relu'.format(block_id)),
                
                # pw
                Conv2D(oup, (1, 1), 
                       padding='same', 
                       use_bias=False,
                       strides=(1,1),
                       name='conv_pw_{0}'.format(block_id)),
                BatchNormalization(name='conv_pw_{0}_bn'.format(block_id)),
                ReLU(name='conv_pw_{0}_relu'.format(block_id))
            ])
        
        self.features = Sequential([
            # inp, oup, stride, block_id
            conv_bn(3, 32, 2, 1),
            conv_dw(32, 64, 1, 2),
            conv_dw(64, 128, 2, 3),
            conv_dw(128, 128, 1, 4),
            conv_dw(128, 256, 2, 5),
            conv_dw(256, 256, 1, 6),
            conv_dw(256, 512, 2, 7),
            conv_dw(512, 512, 1, 8),
            conv_dw(512, 512, 1, 9),
            conv_dw(512, 512, 1, 10),
            conv_dw(512, 512, 1, 11),
            conv_dw(512, 512, 1, 12),
            conv_dw(512, 1024, 2, 13),
            conv_dw(1024, 1024, 1, 14)
        ])
        
        self.avg_pool2d = AveragePooling2D(pool_size=(7, 7))
        self.fc = Dense(num_classes)
        self.softmax = layers.Activation('softmax', name='act_softmax')
        
    def call(self, x):
        x = self.features(x)
        print(x.shape)
        x = self.avg_pool2d(x)
        x = tf.reshape(x, [-1, 1024])
        x = self.softmax(self.fc(x))
        return x