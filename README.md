# Leveraging-General-Purpose-Segmentation-Methods-for-Glossy-Surface-Segmentation

## Introduction
In this project, we would like to measure how well can existing general-purpose semantic segmentationmethods perform on the specific task of detecting windows and mirrors, and make possible improve-ment to existing models based on the insight we learned. We are aiming to segment glossy surfacefrom complex environment. Several representative glossy object like window, mirror will be detectedin this project via deep learning approaches and the proposed will be compared with existing methodsbased on results.

## Dataset
[MirrorNet](https://mhaiyang.github.io/ICCV2019_MirrorNet/index.html) is a large-scale mirror dataset that contains mirror images with the correspondingmanually annotated masks. It is consisted of a training set and a testing set, which has 3063 and 955mirror images taking at a indoor setting, respectively. The ground truth annotation for each mirrorimage is represented as an additional binary image.

## Models
- MirrorNet
- MaskRCNN
- R3Net
- PSPNet
- PFAN
## Evaluation

`evaluation.py` takes groud truth and predicted saliency maps from different models to compute scores visual results. 
- stage 1: compute IoU
- stage 2: compute F_measure
- stage 3: compute BER
- stage 4: compute MAE
- stage 5: compute Acc
- stage 6: compute all evaluation metrics
- stage 7: get low IoU samples
- stage 8: draw mask ratio distribution figure
- stage 9: mask ratio and IOU relation figure
