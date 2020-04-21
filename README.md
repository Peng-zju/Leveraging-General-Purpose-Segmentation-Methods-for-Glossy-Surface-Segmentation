# Leveraging-General-Purpose-Segmentation-Methods-for-Glossy-Surface-Segmentation

## Introduction
In this project, we measured how well can existing general-purpose semantic segmentation methods perform on the specific task of detecting mirrors, and make possible improvement to existing models based on the insight we learned. We are aiming to segment glossy surface from complex environment. Representative glossy objects like mirror will be detected in this project via deep learning approaches and five models will be compared based on results.

## Dataset
[MirrorNet](https://mhaiyang.github.io/ICCV2019_MirrorNet/index.html) is a large-scale mirror dataset that contains mirror images with the corresponding manually annotated masks. It consists of a training set and a testing set, which has 3063 and 955 mirror images taking at a indoor setting, respectively. The ground truth annotation for each mirror image is represented as an additional binary image.

## Models
- [MirrorNet](https://github.com/Mhaiyang/ICCV2019_MirrorNet)
- [MaskRCNN](https://github.com/facebookresearch/detectron2/blob/master/GETTING_STARTED.md)
- [R3Net](https://github.com/zijundeng/R3Net)
- [PSPNet](https://github.com/divamgupta/image-segmentation-keras)
- [PFAN](https://github.com/sairajk/PyTorch-Pyramid-Feature-Attention-Network-for-Saliency-Detection)

## Evaluation
`evaluation.py` takes groud truth and predicted saliency maps from different models as input to compute scores and draw evaluation figures. 
- stage 1: compute IoU
- stage 2: compute F_measure
- stage 3: compute BER
- stage 4: compute MAE
- stage 5: compute Acc
- stage 6: compute all evaluation metrics
- stage 7: get low IoU samples
- stage 8: draw mask ratio distribution figure
- stage 9: draw mask ratio and IoU relation figure
