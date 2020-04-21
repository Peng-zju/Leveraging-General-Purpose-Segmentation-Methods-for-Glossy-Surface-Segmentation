from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
# register_coco_instances("my_dataset", {}, "json_annotation.json", "path/to/image/dir")
import random
from detectron2.utils.visualizer import Visualizer
import cv2
from PIL import Image
from skimage import io,data
from detectron2.utils.visualizer import ColorMode
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import os
import numpy as np
from skimage import io
import argparse




def get_IoU(mask_path, scorce_threshold, scores, pred_masks):
    
    #------------- get filtered masks -------------
    _ , h , w = pred_masks.shape
    filtered_index = np.where(scores > scorce_threshold)
    if not (scores > scorce_threshold).all():
        print("nothing over threshold")
        return 0
    GT_mask = io.imread(mask_path)
    pred_mask = np.full((h,w),False, dtype=bool)
    
    for index in filtered_index:
        pred_mask = np.logical_or(pred_mask, pred_masks[index,:,:])
    # mask = mask.astype(int)
    intersct = np.logical_and(pred_mask,GT_mask)
    union =  np.logical_or(pred_mask, GT_mask)
    IoU = np.sum(intersct) / np.sum(union)
    # if np.isnan(IoU):
    #     print(mask_path, np.sum(intersct), np.sum(union))
    #     return 999

    return IoU

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Get Setting :D')
    parser.add_argument(
        '--t', default=0.5, type=float, help="threshold")
    args = parser.parse_args()
    scorce_threshold = args.t
    folder_path = "/local-scratch/jiaqit/exp/detectron2_mirror/datasets/mirror/val2017"
    mask_folder = "/local-scratch/jiaqit/exp/data/MSD/test/mask"

    cfg = get_cfg()
    cfg.merge_from_file(
        "./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 3 classes (data, fig, hazelnut)
    cfg.OUTPUT_DIR = "output/mirror_test_1080000"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    register_coco_instances("mirror_val", {}, "/local-scratch/jiaqit/exp/detectron2_mirror/datasets/mirror/annotations/instances_val2017.json", "/local-scratch/jiaqit/exp/detectron2_mirror/datasets/mirror/val2017")
    mirror_metadata = MetadataCatalog.get("mirror_val")
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0194999.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1   # set the testing threshold for this model
    cfg.DATASETS.TEST = ("mirror_val")
    predictor = DefaultPredictor(cfg)

    img_num = len(os.listdir(folder_path))
    sum_IoU = 0
    for img_name in os.listdir(folder_path):

        # img_path = "/local-scratch/jiaqit/exp/detectron2_mirror/datasets/mirror/val2017/5153_512x640.jpg"
        img_path = os.path.join(folder_path, img_name)
        im = cv2.imread(img_path)
        outputs = predictor(im)
        mask_exists = outputs["instances"].to("cpu").pred_masks.shape[0]
        if mask_exists:
            mask_path = os.path.join(mask_folder, img_name.replace(".jpg",".png"))
            # if (get_IoU(mask_path,scorce_threshold, outputs["instances"].to("cpu").scores.numpy() ,outputs["instances"].to("cpu").pred_masks.numpy()) > 990):
            #     v = Visualizer(im[:, :, ::-1],
            #                 metadata=mirror_metadata, 
            #                 scale=0.2, 
            #                 instance_mode=ColorMode.IMAGE_BW)
            #     v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            #     im=Image.fromarray(v.get_image()[:, :, ::-1])
            #     im.show()
            # else:
            IoU = get_IoU(mask_path,scorce_threshold, outputs["instances"].to("cpu").scores.numpy() ,outputs["instances"].to("cpu").pred_masks.numpy())
            sum_IoU += IoU
            print(img_path, IoU)
        else:
            print(img_path , " don't have mask")
    print("threshold : ", scorce_threshold ," mean IoU : ", sum_IoU/img_num)
        # v = Visualizer(im[:, :, ::-1],
        #             metadata=mirror_metadata, 
        #             scale=0.2, 
        #             instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
        # )
        # v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # im=Image.fromarray(v.get_image()[:, :, ::-1])
        # im.show()




    