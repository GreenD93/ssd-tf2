import tensorflow as tf

from tensorflow.keras import Model
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, AveragePooling2D, DepthwiseConv2D, Flatten, Dense, BatchNormalization, Dropout, ReLU
from tensorflow.keras.models import Sequential

from ssd.ssd import SSD
from nn.mobilenetv1 import MobilenetV1

from settings import *

def create_ssd_mobilenetv1_lite():
    
    num_classes = NUM_CLASSES
    arch = 'ssd300-mobilenetv1_lite'

    conv4_block, conv7_block = base_net()
    extra_layers = create_mobilenetv1_lite_extra_layers()
    conf_head_layers = create_mobilenetv1_lite_conf_head_layers(num_classes)
    loc_head_layers = create_mobilenetv1_lite_loc_head_layers()

    return SSD(num_classes, arch, conv4_block, conv7_block, extra_layers, conf_head_layers, loc_head_layers)

def SeperableConv2d(out_dim, kernel_size, stride):
    sperable_conv = Sequential([
        
        # Depthwise 
        layers.DepthwiseConv2D(kernel_size=kernel_size, strides=(stride, stride), padding='same', 
                               activation='relu'),
        
        # Pointwise
        layers.Conv2D(out_dim, kernel_size=1)
        ])
    
    return sperable_conv

def base_net():
    
    base_model = MobilenetV1().features
    
    # 4th block
    # 19, 19, 512
    mobile_v1_conv4 = []
    for layer in base_model.layers[:12]:
        mobile_v1_conv4.append(layer)
    
    x = layers.Input(shape=[None, None, 3])
    out = x
    
    for layer in mobile_v1_conv4:
        out = layer(out)
        
    #zero-padding이 성능에 악화? 
    mobile_v1_conv4 = tf.keras.Model(x, out)


    # 7th block
    # 10, 10, 1024
    mobile_v1_conv7 = base_model.layers[12:]
    
    x = layers.Input(shape=[None, None, 512])
    
    out = x
    for layer in mobile_v1_conv7:
        out = layer(out)

    mobile_v1_conv7 = tf.keras.Model(x, out)

    return mobile_v1_conv4, mobile_v1_conv7

def create_mobilenetv1_lite_extra_layers():
    """ 
        Create extra layers
        8th to 11th 
    """
    extra_layers = [
        
        # 8th block output shape: B, 512, 10, 10
        Sequential([
            layers.Conv2D(256, 1, activation='relu'),
            SeperableConv2d(512, 3, 2)
        ]),
        # 9th block output shape: B, 256, 5, 5
        Sequential([
            layers.Conv2D(128, 1, activation='relu'),
            SeperableConv2d(256, 3, 2)
        ]),
        # 10th block output shape: B, 256, 3, 3
        Sequential([
            layers.Conv2D(128, 1, activation='relu'),
            SeperableConv2d(256, 3, 2)
        ]),
        # 11th block output shape: B, 256, 1, 1
        Sequential([
            layers.Conv2D(128, 1, activation='relu'),
            SeperableConv2d(256, 3, 2)
        ])
    ]

    return extra_layers



def create_mobilenetv1_lite_conf_head_layers(num_classes):
    """ 
        Create layers for classification
    """
    
    conf_head_layers = [
        SeperableConv2d(6 * num_classes, 3, 1), # for 4th block
        SeperableConv2d(6 * num_classes, 3, 1), # for 7th block
        SeperableConv2d(6 * num_classes, 3, 1), # for 8th block
        SeperableConv2d(6 * num_classes, 3, 1), # for 9th block
        SeperableConv2d(6 * num_classes, 3, 1), # for 10th block
        layers.Conv2D(6 * num_classes, kernel_size=3, padding='same') # for 11th block
    ]

    return conf_head_layers


def create_mobilenetv1_lite_loc_head_layers():
    """ 
        Create layers for regression
    """
    
    loc_head_layers = [
        SeperableConv2d(6 * 4, 3, 1), # for 4th block
        SeperableConv2d(6 * 4, 3, 1), # for 7th block
        SeperableConv2d(6 * 4, 3, 1), # for 8th block
        SeperableConv2d(6 * 4, 3, 1), # for 9th block
        SeperableConv2d(6 * 4, 3, 1), # for 10th block
        layers.Conv2D(6 * 4, kernel_size=3, padding='same') # for 11th block
    ]

    return loc_head_layers