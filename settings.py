DATA_DIR = 'dataset'
CHECKPOINT_DIR = 'mobilenet_checkpoints'
CHECKPOINT_PATH = 'mobilenet_checkpoints/ssd_epoch_100.h5'
OUTPUT_IMAGES_DIR = 'outputs/image'
OUTPUT_DETECTS_DIR = 'outputs/detects'


DATA_YEAR = 2012

# for mAP score 
USE_07_METRIC = False

# model selection
ARCH = 'ssd300-mobilenetv1'


# base : vgg16 layer 
# new : new train
# specified : load weight

PRETRAINED_TYPE = 'specified'

INFO = {
    'pre_ssd300-mobilenetv2': {
          'ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
          'scales': [0.1, 0.2, 0.375, 0.55, 0.725, 0.9, 1.075],
          'fm_sizes': [19, 10, 5, 3, 2, 1],
          'image_size': 300        
    },
    
    'test_pre_ssd300-mobilenetv1': {
          'ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
          'scales': [0.1, 0.2, 0.375, 0.55, 0.725, 0.9, 1.075],
          'fm_sizes': [18, 8, 4, 2, 1],
          'image_size': 300        
    },
    
    'pre_ssd300-mobilenetv1': {
          'ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
          'scales': [0.1, 0.2, 0.375, 0.55, 0.725, 0.9, 1.075],
          'fm_sizes': [18, 8, 4, 2, 1, 1],
          'image_size': 300        
    },
    
    'ssd300-mobilenetv1': {
          'ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
          'scales': [0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1.1],
          'fm_sizes': [19, 10, 5, 3, 2, 1],
          'image_size': 300        
    },
    
    'ssd300': {
          'ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
          'scales': [0.1, 0.2, 0.375, 0.55, 0.725, 0.9, 1.075],
          'fm_sizes': [38, 19, 10, 5, 3, 1],
          'image_size': 300
    }
,
    
    'ssd512':{
          'ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2], [2]],
          'scales': [0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05],
          'fm_sizes': [64, 32, 16, 8, 6, 4, 1],
          'image_size': 512
    }    
}

if INFO[ARCH]['image_size'] == 300:
    SIZE = 300
else:
    SIZE = 512

NUM_CLASSES = 21 # class + foreground/background

IDX_TO_NAME = [
                'aeroplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant',
                'sheep', 'sofa', 'train', 'tvmonitor'    
            ]


NUM_EPOCHS = 1000
BATCH_SIZE = 28
NUM_BATCHES = -1

INITIAL_LR = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4 

NEG_RATIO = 3
CLS_SCORE = 0.6
NMS_THRESHOLD = 0.45
LIMIT = 200
