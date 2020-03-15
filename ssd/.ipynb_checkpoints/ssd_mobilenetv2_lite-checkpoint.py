import tensorflow as tf

from tensorflow.keras import Model
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, AveragePooling2D, DepthwiseConv2D, Flatten, Dense, BatchNormalization, Dropout, ReLU
from tensorflow.keras.models import Sequential

from ssd.ssd import SSD
from nn.mobilenetv2 import MobilenetV2, InvertedResidual

from settings import *


def create_ssd_mobilenetv2_lite():
    
    num_classes = NUM_CLASSES
    arch = 'ssd300-mobilenetv2_lite'

    conv4_block, conv7_block = base_net()
    extra_layers = create_mobilenetv2_lite_extra_layers()
    conf_head_layers = create_mobilenetv2_lite_conf_head_layers(num_classes)
    loc_head_layers = create_mobilenetv2_lite_loc_head_layers()

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
    base_model = MobilenetV2(num_classes=1000).features
                  
    # 4th block
    # 300, 300, 3 -> 19, 19, 96 -> sub : 19, 19, 576
    mobile_v2_conv4 = []
    
    for layer in base_model.layers[:14]:
        mobile_v2_conv4.append(layer)
        
    # sub_layer : Covn2D, Batch, ReLU
    # 19, 19, 96 -> 19, 19, 576
    for sub_layer in base_model.layers[14].layers[0].layers[:3]:
        mobile_v2_conv4.append(sub_layer)
        
    x = layers.Input(shape=[None, None, 3])
    out = x
    
    for layer in mobile_v2_conv4:
        out = layer(out)
    
    mobile_v2_conv4 = tf.keras.Model(x, out)
    
    # 7th block
    # 19, 19, 576 -> 10, 10, 1280
    mobile_v2_conv7 = []

    for sub_layer in base_model.layers[14].layers[0].layers[3:]:
        mobile_v2_conv7.append(sub_layer)

    for layer in base_model.layers[16:]:
        mobile_v2_conv7.append(layer)

    x = layers.Input(shape=[None, None, 576])
    out = x

    for layer in mobile_v2_conv7:
        out = layer(out)

    mobile_v2_conv7 = tf.keras.Model(x, out)
    
    return mobile_v2_conv4, mobile_v2_conv7

def create_mobilenetv2_lite_extra_layers():
    """ 
        Create extra layers
        8th to 11th 
    """
    extra_layers = [
        
        # 8th block output shape: B, 512, 10, 10
        InvertedResidual(1280, 512, stride=2, expand_ratio=0.2, block_id='extra_1'),
        
        # 9th block output shape: B, 256, 5, 5
        InvertedResidual(512, 256, stride=2, expand_ratio=0.25, block_id='extra_2'),

        # 10th block output shape: B, 256, 3, 3
        InvertedResidual(512, 256, stride=2, expand_ratio=0.5, block_id='extra_3'),

        # 11th block output shape: B, 256, 1, 1
        InvertedResidual(512, 256, stride=2, expand_ratio=0.25, block_id='extra_4'),
        
    ]

    return extra_layers


def create_mobilenetv2_lite_conf_head_layers(num_classes):
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


def create_mobilenetv2_lite_loc_head_layers():
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
