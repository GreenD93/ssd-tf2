[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1c2M7iayzqh6lzHJCihUrTR78AF0jVOsb)

## TF 2.1과 jupyter notebook에서 작동될 수 있게 기존 코드에서 수정하였습니다.

### Data 준비

1. VOC2012 dataset을 다운로드합니다.

> wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

2. 해당 데이터 압축을 풀어줍니다.

- (1) dataset 폴더를 만듭니다.
> mkdir dataset
- (2) dataset폴더 안에 VOC2012 폴더를 만듭니다.
> mkdir dataset/VOC2012
- (3) dataset > VOC2012 폴더에 다운로드 받은 데이터의 압축을 풀어줍니다. dataset/VOC2012/....) Annotations/ImageSets/JPEGImages/SegmentationClass/SegmentationObject 폴더를 준비합니다.
> tar -xvf VOCtrainval_11-May-2012.tar

### 학습 및 테스트 하기

>settings.py파일을 확인하여 학습에 필요한 parameter를 수정한 뒤 각각 train/inference에 맞는 작업을 진행합니다.

파라미터 설정:
-  `DATA_DIR` dataset 폴더 
-  `CHECKPOINT_DIR` 학습 후 저장될 model의 checkpoint가 저장될 폴더 
-  `CHECKPOINT_PATH` 학습 후 저장될 model의 checkpoint의 경로
-  `OUTPUT_IMAGES_DIR` 예측 후 저장될 이미지 폴더
-  `OUTPUT_DETECTS_DIR` 예측 후 저장될 이미지의 메타 정보(박스 위치, class)가 저장될 폴더
-  `DATA_YEAR` VOCData년도 (2007 or 2012)
-  `ARCH` SSD network 아키텍쳐 (ssd300 or ssd512)
-  `PRETRAINED_TYPE` (`base` : 기본 vgg16 model feature extractor, `specified` : 학습된 모델 weights)

- train:
  - SSD_train.ipynb

- inference:
  - SSD_inference.ipynb


# SSD (Single Shot MultiBox Detector) - Tensorflow 2.0

## Preparation
- Download PASCAL VOC dataset (2007 or 2012) and extract at `./data`
- Install necessary dependencies:
```
pip install -r requirements.txt
```

## Training
Arguments for the training script:

```
>> python train.py --help
usage: train.py [-h] [--data-dir DATA_DIR] [--data-year DATA_YEAR]
                [--arch ARCH] [--batch-size BATCH_SIZE]
                [--num-batches NUM_BATCHES] [--neg-ratio NEG_RATIO]
                [--initial-lr INITIAL_LR] [--momentum MOMENTUM]
                [--weight-decay WEIGHT_DECAY] [--num-epochs NUM_EPOCHS]
                [--checkpoint-dir CHECKPOINT_DIR]
                [--pretrained-type PRETRAINED_TYPE] [--gpu-id GPU_ID]
```
Arguments explanation:
-  `--data-dir` dataset directory (must specify to VOCdevkit folder)
-  `--data-year` the year of the dataset (2007 or 2012)
-  `--arch` SSD network architecture (ssd300 or ssd512)
-  `--batch-size` training batch size
-  `--num-batches` number of batches to train (`-1`: train all)
-  `--neg-ratio` ratio used in hard negative mining when computing loss
-  `--initial-lr` initial learning rate
-  `--momentum` momentum value for SGD
-  `--weight-decay` weight decay value for SGD
-  `--num-epochs` number of epochs to train
-  `--checkpoint-dir` checkpoint directory
-  `--pretrained-type` pretrained weight type (`base`: using pretrained VGG backbone, other options: see testing section)
-  `--gpu-id` GPU ID

- how to train SSD300 using PASCAL VOC2007 for 100 epochs:

```
python train.py --data-dir ./data/VOCdevkit --data-year 2007 --num-epochs 100
```

- how to train SSD512 using PASCAL VOC2012 for 120 epochs on GPU 1 with batch size 8 and save weights to `./checkpoints_512`:

```
python train.py --data-dir ./data/VOCdevkit --data-year 2012 --arch ssd512 --num-epochs 120 --batch-size 8 --checkpoint_dir ./checkpoints_512 --gpu-id 1
```

## Testing
Arguments for the testing script:
```
>> python test.py --help
usage: test.py [-h] [--data-dir DATA_DIR] [--data-year DATA_YEAR]
               [--arch ARCH] [--num-examples NUM_EXAMPLES]
               [--pretrained-type PRETRAINED_TYPE]
               [--checkpoint-dir CHECKPOINT_DIR]
               [--checkpoint-path CHECKPOINT_PATH] [--gpu-id GPU_ID]
```
Arguments explanation:
-  `--data-dir` dataset directory (must specify to VOCdevkit folder)
-  `--data-year` the year of the dataset (2007 or 2012)
-  `--arch` SSD network architecture (ssd300 or ssd512)
-  `--num-examples` number of examples to test (`-1`: test all)
-  `--checkpoint-dir` checkpoint directory
-  `--checkpoint-path` path to a specific checkpoint
-  `--pretrained-type` pretrained weight type (`latest`: automatically look for newest checkpoint in `checkpoint_dir`, `specified`: use the checkpoint specified in `checkpoint_path`)
-  `--gpu-id` GPU ID

- how to test the first training pattern above using the latest checkpoint:

```
python test.py --data-dir ./data/VOCdevkit --data-year 2007 --checkpoint_dir ./checkpoints
```

- how to test the second training pattern above using the 100th epoch's checkpoint, using only 40 examples:

```
python test.py --data-dir ./data/VOCdevkit --data-year 2012 --arch ssd512 --checkpoint_path ./checkpoints_512/ssd_epoch_100.h5 --num-examples 40
```

## Reference
- Single Shot Multibox Detector paper: [paper](https://arxiv.org/abs/1512.02325)
- Caffe original implementation: [code](https://github.com/weiliu89/caffe/tree/ssd)
- Pytorch implementation: [code] (https://github.com/ChunML/ssd-pytorch)
