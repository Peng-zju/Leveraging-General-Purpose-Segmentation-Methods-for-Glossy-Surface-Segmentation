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
from matplotlib import pyplot as plt
import os
import argparse
import numpy as np
import sys
import torch


def save_result(file_path, info):
    with open(file_path,"a") as file:
        file.write(info)

def get_MAE_test(args, predictor):
    img_num = len(os.listdir(args.val_data_path))
    sum_MAE = 0
    # img_name = "3704_512x640.jpg"
    # img_path = "/local-scratch/jiaqit/exp/detectron2_mirror/datasets/mirror/val2017/3704_512x640.jpg"
    for img_name in os.listdir(args.val_data_path):
        img_path = os.path.join(args.val_data_path, img_name)
    
        im = cv2.imread(img_path)
        outputs = predictor(im)
        mask_exists = outputs["instances"].to("cpu").pred_masks.shape[0]
        if mask_exists:
            mask_path = os.path.join(args.mask_folder, img_name.replace(".jpg",".png"))
            pred_mask, GT_mask = get_pred_and_GT_mask(mask_path,args.scorce_threshold, outputs["instances"].to("cpu").scores.numpy() ,outputs["instances"].to("cpu").pred_masks.numpy())
            FP = pred_mask.astype(int) - GT_mask.astype(int)
            FP[FP <0] = 0 
            FP = FP.astype(bool)
            FN = (np.logical_not(pred_mask)).astype(int) - (np.logical_not(GT_mask)).astype(int)
            FN[FN <0] = 0 
            FN = FN.astype(bool)
            MAE = (np.sum(FN) + np.sum(FP)) / pred_mask.size
            sum_MAE += MAE
            # print(img_path, MAE, precision, recall,  np.sum(TP), np.sum(FP), np.sum(FN)) # debug
    save_result(args.result_save_path, "threshold {} mean MAE {}".format(args.scorce_threshold, sum_MAE/img_num))
    print("threshold : ", args.scorce_threshold ," mean MAE : ", sum_MAE/img_num)


def get_MAE(args, predictor):
    img_num = len(os.listdir(args.val_data_path))
    sum_MAE = 0
    # img_name = "3704_512x640.jpg"
    # img_path = "/local-scratch/jiaqit/exp/detectron2_mirror/datasets/mirror/val2017/3704_512x640.jpg"
    for img_name in os.listdir(args.val_data_path):
        img_path = os.path.join(args.val_data_path, img_name)
    
        im = cv2.imread(img_path)
        outputs = predictor(im)
        mask_exists = outputs["instances"].to("cpu").pred_masks.shape[0]
        if mask_exists:
            mask_path = os.path.join(args.mask_folder, img_name.replace(".jpg",".png"))
            pred_mask, GT_mask = get_pred_and_GT_mask(mask_path,args.scorce_threshold, outputs["instances"].to("cpu").scores.numpy() ,outputs["instances"].to("cpu").pred_masks.numpy())
            FP = pred_mask.astype(int) - GT_mask.astype(int)
            FP[FP <0] = 0 
            FP = FP.astype(bool)
            FN = (np.logical_not(pred_mask)).astype(int) - (np.logical_not(GT_mask)).astype(int)
            FN[FN <0] = 0 
            FN = FN.astype(bool)
            MAE = (np.sum(FN) + np.sum(FP)) / pred_mask.size
            sum_MAE += MAE
            # print(img_path, MAE, precision, recall,  np.sum(TP), np.sum(FP), np.sum(FN)) # debug
    save_result(args.result_save_path, "threshold {} mean MAE {}".format(args.scorce_threshold, sum_MAE/img_num))
    print("threshold : ", args.scorce_threshold ," mean MAE : ", sum_MAE/img_num)

def get_F_score(args, predictor):
    img_num = len(os.listdir(args.val_data_path))
    sum_F_score = 0
    # img_name = "3704_512x640.jpg"
    # img_path = "/local-scratch/jiaqit/exp/detectron2_mirror/datasets/mirror/val2017/3704_512x640.jpg"
    for img_name in os.listdir(args.val_data_path):
        img_path = os.path.join(args.val_data_path, img_name)
    
        im = cv2.imread(img_path)
        outputs = predictor(im)
        mask_exists = outputs["instances"].to("cpu").pred_masks.shape[0]
        if mask_exists:
            mask_path = os.path.join(args.mask_folder, img_name.replace(".jpg",".png"))
            pred_mask, GT_mask = get_pred_and_GT_mask(mask_path,args.scorce_threshold, outputs["instances"].to("cpu").scores.numpy() ,outputs["instances"].to("cpu").pred_masks.numpy())
            TP = np.logical_and(pred_mask,GT_mask)
            FP = pred_mask.astype(int) - GT_mask.astype(int)
            FP[FP <0] = 0 
            FP = FP.astype(bool)
            FN = (np.logical_not(pred_mask)).astype(int) - (np.logical_not(GT_mask)).astype(int)
            FN[FN <0] = 0 
            FN = FN.astype(bool)
            if np.sum(TP):
                precision = np.sum(TP) / (np.sum(TP) + np.sum(FP))
                recall = np.sum(TP) / (np.sum(TP) + np.sum(FN))
                F_score = ((1 + 0.3)*precision*recall)/ (0.3*precision+recall)
            else:
                F_score = 0
            sum_F_score += F_score
            # print(img_path, F_score, precision, recall,  np.sum(TP), np.sum(FP), np.sum(FN)) # debug
    save_result(args.result_save_path, "threshold {} mean F_score {}".format(args.scorce_threshold, sum_F_score/img_num))
    print("threshold : ", args.scorce_threshold ," mean F_score : ", sum_F_score/img_num)

def get_BER_score(args, predictor):
    pass

def get_Acc_score(args, predictor):
    img_num = len(os.listdir(args.val_data_path))
    sum_Acc = 0
    # img_name = "3704_512x640.jpg"
    # img_path = "/local-scratch/jiaqit/exp/detectron2_mirror/datasets/mirror/val2017/3704_512x640.jpg"
    for img_name in os.listdir(args.val_data_path):
        img_path = os.path.join(args.val_data_path, img_name)
    
        im = cv2.imread(img_path)
        outputs = predictor(im)
        mask_exists = outputs["instances"].to("cpu").pred_masks.shape[0]
        if mask_exists:
            mask_path = os.path.join(args.mask_folder, img_name.replace(".jpg",".png"))
            pred_mask, GT_mask = get_pred_and_GT_mask(mask_path,args.scorce_threshold, outputs["instances"].to("cpu").scores.numpy() ,outputs["instances"].to("cpu").pred_masks.numpy())
            TP_pixel = np.logical_and(pred_mask,GT_mask)
            TN_pixel = np.logical_and(np.logical_not(pred_mask),np.logical_not(GT_mask))
            Acc = (np.sum(TP_pixel) + np.sum(TN_pixel))/ TN_pixel.size
            sum_Acc += Acc
    save_result(args.result_save_path, "threshold {} mean Acc {}".format(args.scorce_threshold, sum_Acc/img_num))
    print("threshold : ", args.scorce_threshold ," mean Acc : ", sum_Acc/img_num)
    

def get_IoU_score(args, predictor):
    img_num = len(os.listdir(args.val_data_path))
    sum_IoU = 0
    for img_name in os.listdir(args.val_data_path):
        img_path = os.path.join(args.val_data_path, img_name)
        # print(img_path)
        im = cv2.imread(img_path)
        outputs = predictor(im)
        mask_exists = outputs["instances"].to("cpu").pred_masks.shape[0]
        if mask_exists:
            mask_path = os.path.join(args.mask_folder, img_name.replace(".jpg",".png"))
            pred_mask, GT_mask = get_pred_and_GT_mask(mask_path,args.scorce_threshold, outputs["instances"].to("cpu").scores.numpy() ,outputs["instances"].to("cpu").pred_masks.numpy())
            intersct = np.logical_and(pred_mask,GT_mask)
            union =  np.logical_or(pred_mask, GT_mask)
            IoU = np.sum(intersct) / np.sum(union)
            sum_IoU += IoU
        #     print(img_path, IoU)
        # else:
        #     print(img_path, "don't have mask")
    print("threshold : ", args.scorce_threshold ," mean IoU : ", sum_IoU/img_num)
    print("threshold : ", args.scorce_threshold )
    save_result(args.result_save_path, "threshold {} mean IoU {}".format(args.scorce_threshold, sum_IoU/img_num))
    

# return boolean predict mask and GT mask
def get_pred_and_GT_mask(mask_path, scorce_threshold, scores, pred_masks):
    
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
    # intersct = np.logical_and(pred_mask,GT_mask)
    # union =  np.logical_or(pred_mask, GT_mask)
    # IoU = np.sum(intersct) / np.sum(union)
    sum_pred_mask = np.full((h,w),False, dtype=bool)
    for one_pred_mask in pred_mask:
        sum_pred_mask = np.logical_or(one_pred_mask, sum_pred_mask)
    return sum_pred_mask, GT_mask.astype(bool)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Get Setting :D')
    # parser.add_argument(
    #     '--scene_num', default="0000", type=int, help="scene number") 
    parser.add_argument(
        '--train_json_path', default="./datasets/mirror/annotations/instances_train2017.json", type=str) 
    parser.add_argument(
        '--train_data_path', default="./datasets/mirror/train2017", type=str) 
    parser.add_argument(
        '--val_json_path', default="./datasets/mirror/annotations/instances_val2017.json", type=str) 
    parser.add_argument(
        '--val_data_path', default="./datasets/mirror/val2017", type=str) 
    parser.add_argument(
        '--dataset_name', default='mirror_dataset', 
        type=str, help="define a dataset name by yourself")
    parser.add_argument(
        '--config_yaml', default='./configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml', type=str) # ./configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml
    parser.add_argument(
        '--model_weight', default='./output/to_test/model_0134999.pth', type=str)
    #parser.add_argument(
    #    '--checkpoint_save_dir', default=None, type=str, help="check point directory path")
    parser.add_argument(
        '--mask_folder', default='/local-scratch/jiaqit/exp/ICCV2019_MirrorNet/MSD/test/mask', type=str, help="mask folder path")
    parser.add_argument(
        '--scorce_threshold', default=0.1, type=float, help="score threshold for mask")
    parser.add_argument(
        '--mode', default='3', type=str, help="(1) train (2) test [show image] (3) get score")
    parser.add_argument(
        '--score_type', default='1', type=str, help="(1) IoU (2) Acc (3) F_measure (4) MAE --- mean absolute error (5) balabce error rate")
    parser.add_argument(
        '--result_save_path', default='./test.log', type=str)
    args = parser.parse_args()
    '''
    img_path = "/local-scratch/jiaqit/exp/detectron2_mirror/datasets/mirror/train2017"
    dataset_name = "mirror_dataset"
    json_path = "/local-scratch/jiaqit/exp/detectron2_mirror/datasets/mirror/annotations/instances_train2017.json"
    '''
    
    if args.mode == "1":

        register_coco_instances(args.dataset_name, {}, args.train_json_path, args.train_data_path)
        mirror_metadata = MetadataCatalog.get(args.dataset_name)
        dataset_dicts = DatasetCatalog.get(args.dataset_name)

        cfg = get_cfg()
        cfg.merge_from_file(args.config_yaml)
        cfg.DATASETS.TRAIN = (args.dataset_name,)
        cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
        cfg.DATALOADER.NUM_WORKERS = 1
        cfg.MODEL.WEIGHTS = args.model_weight  # initialize from model zoo
        cfg.SOLVER.IMS_PER_BATCH = 4
        cfg.SOLVER.BASE_LR = 0.005
        cfg.SOLVER.STEPS = (280000, 333333)
        cfg.SOLVER.MAX_ITER = (
            360000
        )  # 300 iterations seems good enough, but you can certainly train longer
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
            128
        )  # faster, and good enough for this toy dataset
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 3 classes (data, fig, hazelnut)
        # cfg.OUTPUT_DIR = args.checkpoint_save_dir
        # os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

        # ---- show train
        cfg.MODEL.WEIGHTS = args.model_weight
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.scorce_threshold   # set the testing threshold for this model
        cfg.DATASETS.TEST = (args.dataset_name, )
        predictor = DefaultPredictor(cfg)


        for d in dataset_dicts[-1:]:    
            im = cv2.imread(d["file_name"])
            print(d["file_name"])
            outputs = predictor(im)
            v = Visualizer(im[:, :, ::-1],
                        metadata=mirror_metadata, 
                        scale=0.5, 
                        instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
            )
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            im=Image.fromarray(v.get_image()[:, :, ::-1])
            im.show()

    # ---- show validation -----
    elif args.mode == "2" :
        register_coco_instances(args.dataset_name, {}, args.val_json_path, args.val_data_path)

        mirror_metadata = MetadataCatalog.get(args.dataset_name)
        dataset_dicts = DatasetCatalog.get(args.dataset_name)
        cfg = get_cfg()
        cfg.merge_from_file(args.config_yaml)
        cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
        cfg.MODEL.WEIGHTS = args.model_weight  # initialize from model zoo
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 3 classes (data, fig, hazelnut)
        # cfg.OUTPUT_DIR = args.checkpoint_save_dir
        #cfg.MODEL.WEIGHTS = args.model_weight
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.scorce_threshold  # set the testing threshold for this model
        cfg.DATASETS.TEST = (args.dataset_name, )
        predictor = DefaultPredictor(cfg)


        # for d in dataset_dicts[5000:5005]:    
        # im = cv2.imread(d["file_name"])
        while 1:
            img_path = input("img path : ")
            # img_path = "/local-scratch/jiaqit/exp/detectron2_mirror/datasets/mirror/val2017/4978_512x640.jpg"
            img_path = os.path.join(args.val_data_path, img_path)
            if img_path == "q" :
                exit(1)
            im = cv2.imread(img_path)
            outputs = predictor(im)
            v = Visualizer(im[:, :, ::-1],
                        metadata=mirror_metadata, 
                        scale=0.5, 
                        instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
             )
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            im=Image.fromarray(v.get_image()[:, :, ::-1])
            im.show()

    # ---- calculate score to evaluate the model ----
    elif args.mode == "3":
        register_coco_instances(args.dataset_name, {}, args.val_json_path, args.val_data_path)
        mirror_metadata = MetadataCatalog.get(args.dataset_name)
        dataset_dicts = DatasetCatalog.get(args.dataset_name)
        for threshold in range(1,10,1):
            print(threshold/10)
            args.scorce_threshold = threshold/10
            cfg = get_cfg()
            cfg.merge_from_file(args.config_yaml)
            cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
            cfg.MODEL.WEIGHTS = args.model_weight  # initialize from model zoo
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 3 classes (data, fig, hazelnut)
            #cfg.OUTPUT_DIR = args.checkpoint_save_dir
            # cfg.MODEL.WEIGHTS = args.model_weight
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST =  args.scorce_threshold  # set the testing threshold for this model
            cfg.DATASETS.TEST = (args.dataset_name, )
            predictor = DefaultPredictor(cfg)
            # ------ get score --------
            # --score_type (1) IoU (2) Acc (3) F_measure (4) MAE --- mean absolute error (5) balabce error rate
            if args.score_type == "1":
                print("running get_IoU_score")
                get_IoU_score(args, predictor)
            elif args.score_type == "2":
                print("running get_Acc_score")
                get_Acc_score(args, predictor)
            elif args.score_type == "3":
                print("running get_F_score")
                get_F_score(args, predictor)
            # sys.stdout.flush()


    