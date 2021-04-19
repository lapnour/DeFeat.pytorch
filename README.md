# DeFeat.pytorch Code Base

Implementation of our CVPR2021 paper [Distilling Object Detectors via Decoupled Features](https://arxiv.org/pdf/2103.14475.pdf)

### Abstract

Knowledge distillation is a widely used paradigm for inheriting information from a complicated teacher network to a compact student network and maintaining the strong performance. Different from image classification, object detectors are much more sophisticated with multiple loss functions in which features that semantic information rely on are tangled. In this paper, we point out that the information of features derived from regions excluding objects are also essential for distilling the student detector, which is usually ignored in existing approaches. In addition, we elucidate that features from different regions should be assigned with different importance during distillation. To this end, we present a novel distillation algorithm via decoupled features (DeFeat) for learning a better student detector. Specifically, two levels of decoupled features will be processed for embedding useful information into the student, i.e., decoupled features from neck and decoupled proposals from classification head. Extensive experiments on various detectors with different backbones show that the proposed DeFeat is able to surpass the state-of-the-art distillation methods for object detection. For example, DeFeat improves ResNet50 based Faster R-CNN from 37.4% to 40.9% mAP, and improves ResNet50 based RetinaNet from 36.5% to 39.7% mAP on COCO benchmark. 

### Environments
- Python 3.7
- MMDetection 2.x
- This repo uses: `mmdet-v2.0` `mmcv-0.5.6` `cuda 10.1`

### VOC Results
See [here](https://github.com/ggjy/DeFeat.pytorch/blob/main/configs/faster_rcnn/README.md).

## Acknowledgement
Our code is based on the open source project [MMDetection](https://github.com/open-mmlab/mmdetection).
