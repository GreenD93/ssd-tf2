[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1c2M7iayzqh6lzHJCihUrTR78AF0jVOsb)

# SSD (Single Shot MultiBox Detector) - Tensorflow 2.0

## TF 2.1과 jupyter notebook에서 아래의 모델을 사용할 수 있습니다.
- VGG16
- MobilenetV1
- MobilenetV1-lite
- MobilenetV2-lite

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

- test:
  - SSD_test.ipynb

- evaluate:
  - SSD_evaluate(mAP).ipynb

## Reference
- Single Shot Multibox Detector paper: [paper](https://arxiv.org/abs/1512.02325)

- Base code : (https://github.com/ChunML/ssd-tf2)
- Ref code : (https://github.com/qfgaohao/pytorch-ssd)
