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
import torch
from detectron2.modeling import build_model


def show_imgs(dataset_dicts):
    for d in dataset_dicts[-2:]:
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=mirror_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        im=Image.fromarray(vis.get_image()[:, :, ::-1])
        im.show()

if __name__ == "__main__":

    '''
    img_path = "/local-scratch/jiaqit/exp/detectron2_mirror/datasets/mirror/train2017"
    dataset_name = "mirror_dataset"
    json_path = "/local-scratch/jiaqit/exp/detectron2_mirror/datasets/mirror/annotations/instances_train2017.json"
    '''
    
    register_coco_instances("mirror", {}, "/local-scratch/jiaqit/exp/detectron2_mirror/datasets/mirror/annotations/instances_train2017.json", "/local-scratch/jiaqit/exp/detectron2_mirror/datasets/mirror/train2017")
    mirror_metadata = MetadataCatalog.get("mirror")
    dataset_dicts = DatasetCatalog.get("mirror")

    cfg = get_cfg()
    cfg.merge_from_file(
        "./configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
    )
    cfg.DATASETS.TRAIN = ("mirror",)
    cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.WEIGHTS = "./weight/X-101-32x8d.pkl"  # initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.005
    cfg.SOLVER.STEPS = (280000, 333333)
    cfg.SOLVER.MAX_ITER = (
        360000
    )  # 300 iterations seems good enough, but you can certainly train longer
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        128
    )  # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 3 classes (data, fig, hazelnut)
    cfg.OUTPUT_DIR = "output/mirror_test_1080000"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
   