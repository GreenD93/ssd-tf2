import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, AveragePooling2D, DepthwiseConv2D, Flatten, Dense, BatchNormalization, Dropout, ReLU
from tensorflow.keras.models import Sequential

from ssd.ssd import SSD
from nn.mobilenetv2 import InvertedResidual
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

from settings import *

def create_pre_ssd_mobilenetv2_lite(weights='imagenet'):
    
    num_classes = NUM_CLASSES
    arch = 'pre_ssd300-mobilenetv2'

    conv4_block, conv7_block = base_net(weights)
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

def base_net(weights):
    
    if weights == 'imagenet':
        base_model = MobileNetV2(include_top=False, weights=None, input_shape=(300, 300, 3))
    else:
        base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(300, 300, 3))
        
    # Up to expanded_conv_project_BN (Batch
    first_block = base_model.layers[:10]
    # Conv_1 (Conv2D) to end
    end_block = base_model.layers[150:]
                                    
    # 16 conv_block
    base = base_model.layers[10:150]
    
    block_list = []
    block = []

    start = 1

    for layer in base:
        if 'add' not in layer.name:
            block_num = int(layer.name.split('_')[1])

            if start != block_num:
                start = block_num
                block_list.append(block)

                # block 초기화
                block = []
                block.append(layer)

            else:
                block.append(layer)

    # add block_16
    block_list.append(block)

    # add first_block and end_block
    block_list.insert(0,first_block)
    block_list.append(end_block)
    
    # 4th block
    # (None, 300, 300, 3) -> (None, 19, 19, 96)
    # Up to block_12_project_BN (BatchNorma
    conv4_layer = block_list[:13]

    # 7th block
    # (None, 19, 19, 96) -> (None, 10, 10, 1280)
    # Up to end
    conv7_layer = block_list[13:]

    mobile_v2_conv4 = create_conv4_layer(conv4_layer)
    mobile_v2_conv7 = create_conv7_layer(conv7_layer)

    return mobile_v2_conv4, mobile_v2_conv7

def create_conv4_layer(conv4_layer):
                             
    base_layer = conv4_layer
    
    inputs = Input(shape=(300, 300, 3))
    x = inputs

    first = Sequential(base_layer[0])
    conv1_block = Sequential(base_layer[1])
    conv2_block = Sequential(base_layer[2])
    conv3_block = Sequential(base_layer[3])
    conv4_block = Sequential(base_layer[4])
    conv5_block = Sequential(base_layer[5])
    conv6_block = Sequential(base_layer[6])
    conv7_block = Sequential(base_layer[7])
    conv8_block = Sequential(base_layer[8])
    conv9_block = Sequential(base_layer[9])
    conv10_block = Sequential(base_layer[10])
    conv11_block = Sequential(base_layer[11])
    conv12_block = Sequential(base_layer[12])

    bech_norm_list = []
    for i in range(0, 14):
        bech_norm_list.append(BatchNormalization(epsilon=1e-3, momentum=0.999))

    x = first(x)
    x = conv1_block(x)
    x = bech_norm_list[0](conv2_block(x)) + bech_norm_list[1](x)
    x = conv3_block(x)
    x = bech_norm_list[2](conv4_block(x)) + bech_norm_list[3](x)
    x = bech_norm_list[4](conv5_block(x)) + bech_norm_list[5](x)
    x = conv6_block(x)
    x = bech_norm_list[6](conv7_block(x)) + bech_norm_list[7](x)
    x = bech_norm_list[8](conv8_block(x)) + bech_norm_list[9](x)
    x = bech_norm_list[10](conv9_block(x)) + bech_norm_list[11](x)
    x = conv10_block(x)
    x = bech_norm_list[12](conv11_block(x)) + bech_norm_list[13](x)
    x = conv12_block(x)
    
    model = Model(inputs, x)

    return model

def create_conv7_layer(conv7_layer):

    base_layer = conv7_layer

    inputs = Input(shape=(19, 19, 96))
    x = inputs

    conv13_block = Sequential(base_layer[0])
    conv14_block = Sequential(base_layer[1])
    conv15_block = Sequential(base_layer[2])
    conv16_block = Sequential(base_layer[3])
    end_block = Sequential(base_layer[4])

    bech_norm_list = []
    for i in range(0, 4):
        bech_norm_list.append(BatchNormalization(epsilon=1e-3, momentum=0.999))

    x = conv13_block(x)
    x = bech_norm_list[0](conv14_block(x)) + bech_norm_list[1](x)
    x = bech_norm_list[2](conv15_block(x)) + bech_norm_list[3](x)
    x = conv16_block(x)
    x = end_block(x)

    model = Model(inputs, x)

    return model

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