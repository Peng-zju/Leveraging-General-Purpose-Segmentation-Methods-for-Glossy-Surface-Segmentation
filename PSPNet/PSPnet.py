import cv2
import numpy as np
import glob
from keras_segmentation.pretrained import pspnet_50_ADE_20K , pspnet_101_cityscapes, pspnet_101_voc12


#transform gray image to binary image
filenames = [img for img in glob.glob("/usr/code/dataset/mask/*.png")]

filenames.sort()

images = []
for img in filenames:
    n= cv2.imread(img)
    binary = n.astype("float") / 255.0
    img = img.replace("mask","mask_binary")
    cv2.imwrite(img,binary)


from keras_segmentation.models.pspnet import pspnet
from keras_segmentation.pretrained import model_from_checkpoint_path
from keras_segmentation.train import find_latest_checkpoint

#train model
model = pspnet(n_classes = 2)
model.train(train_images="dataset/image",train_annotations="dataset/mask",checkpoints_path = "/usr/code/tmp/checkpoints",epochs=5)

#load model
model_config = {'model_class':'pspnet','n_classes':2,"input_height":384,"input_width":576}
latest_weight = find_latest_checkpoint("/usr/code/tmp/checkpoints")
model = model_from_checkpoint_path(model_config,latest_weight)


#order to produce prediction images
'''
 python -m keras_segmentation predict \
 --checkpoints_path="/usr/code/tmp_2/checkpoints" \
 --input_path="/usr/code/dataset_test/image/" \
 --output_path="/usr/code/dataset_test/predict/"
 '''
