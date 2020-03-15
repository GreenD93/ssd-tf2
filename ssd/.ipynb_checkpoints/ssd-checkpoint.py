from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16
import tensorflow.keras.layers as layers
import tensorflow as tf
import numpy as np
import os

class SSD(Model):
    """ Class for SSD model
    Attributes:
        num_classes: number of classes
    """

    def __init__(self, num_classes, arch, conv4_block, conv7_block, extra_layers, conf_head_layers, loc_head_layers):
        super(SSD, self).__init__()
        self.num_classes = num_classes
        
        self.conv4_block, self.conv7_block = conv4_block, conv7_block
        self.batch_norm = layers.BatchNormalization(
            beta_initializer='glorot_uniform',
            gamma_initializer='glorot_uniform'
        )
        
        self.extra_layers = extra_layers
        self.conf_head_layers = conf_head_layers
        self.loc_head_layers = loc_head_layers
        
        if arch == 'ssd300':
            self.network = 'ssd300'
            self.extra_layers.pop(-1)
            self.conf_head_layers.pop(-2)
            self.loc_head_layers.pop(-2)
        else:
            self.network = ''
            

    def compute_heads(self, x, idx):
        """ Compute outputs of classification and regression heads
        Args:
            x: the input feature map
            idx: index of the head layer
        Returns:
            conf: output of the idx-th classification head
            loc: output of the idx-th regression head
        """

        conf = self.conf_head_layers[idx](x)
        conf = tf.reshape(conf, [conf.shape[0], -1, self.num_classes])

        loc = self.loc_head_layers[idx](x)
        loc = tf.reshape(loc, [loc.shape[0], -1, 4])

        return conf, loc

    def init_vgg16(self):
        """ 
                Initialize the VGG16 layers from pretrained weights
                and the rest from scratch using xavier initializer
        """
        base_model
        for i in range(len(self.conv4_block.layers)):
            self.conv4_block.get_layer(index=i).set_weights(
                origin_vgg.get_layer(index=i).get_weights())

        fc1_weights, fc1_biases = origin_vgg.get_layer(index=-3).get_weights()
        fc2_weights, fc2_biases = origin_vgg.get_layer(index=-2).get_weights()

        conv6_weights = np.random.choice(
            np.reshape(fc1_weights, (-1,)), (3, 3, 512, 1024))
        conv6_biases = np.random.choice(
            fc1_biases, (1024,))

        conv7_weights = np.random.choice(
            np.reshape(fc2_weights, (-1,)), (1, 1, 1024, 1024))
        conv7_biases = np.random.choice(
            fc2_biases, (1024,))

        self.conv7_block.get_layer(index=2).set_weights(
            [conv6_weights, conv6_biases])
        self.conv7_block.get_layer(index=3).set_weights(
            [conv7_weights, conv7_biases])
        
    def call(self, x):
        """ 
        The forward pass
        Args:
            x: the input image
        Returns:
            confs: list of outputs of all classification heads
            locs: list of outputs of all regression heads
        """
        confs = []
        locs = []
        head_idx = 0

            
        # 4th block
        if self.network == 'ssd300':
            for i in range(len(self.conv4_block.layers)):
                x = self.conv4_block.get_layer(index=i)(x)
                if i == len(self.conv4_block.layers) - 5:
                    conf, loc = self.compute_heads(self.batch_norm(x), head_idx)
                    confs.append(conf)
                    locs.append(loc)
                    head_idx += 1
                    
        else:
            x = self.conv4_block(x)        
            conf, loc = self.compute_heads(self.batch_norm(x), head_idx)
            
            confs.append(conf)
            locs.append(loc)
            head_idx += 1

            
        # 7th block
        x = self.conv7_block(x)

        conf, loc = self.compute_heads(x, head_idx)

        confs.append(conf)
        locs.append(loc)
        head_idx += 1
        
        # extra_layers
        for layer in self.extra_layers:
            x = layer(x)
            conf, loc = self.compute_heads(x, head_idx)
            
            confs.append(conf)
            locs.append(loc)
            head_idx += 1

        confs = tf.concat(confs, axis=1)
        locs = tf.concat(locs, axis=1)

        return confs, locs
    
def init_ssd(ssd, pretrained_type, checkpoint_path=None):
    """ 
    Create SSD model and load pretrained weights
    Args:
        num_classes: number of classes
        pretrained_type: type of pretrained weights, can be either 'VGG16' or 'ssd'
        weight_path: path to pretrained weights
    Returns:
        net: the SSD model
    """
    net = ssd
    
    dummy = tf.random.normal((1, 300, 300, 3))
    net(dummy)
    
    if pretrained_type == 'base':
        net.init_vgg16()
            
    elif pretrained_type == 'specified':
        
        if not os.path.isfile(checkpoint_path):
            raise ValueError(
                'Not a valid checkpoint file: {}'.format(checkpoint_path))

        try:
            # load weights
            print('>>> load_weights')
            net.load_weights(checkpoint_path)
            
        except Exception as e:
            raise ValueError(
                'Please check the following\n1./ Is the path correct: {}?\n2./ Is the model architecture correct: {}?'.format(
                    checkpoint_path, arch))
            
    return net