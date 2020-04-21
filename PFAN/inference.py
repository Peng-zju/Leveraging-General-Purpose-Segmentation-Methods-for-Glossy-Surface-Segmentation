from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import argparse
import cv2
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms


from model import SODModel
from dataloader import InfDataloader, SODLoader


def parse_arguments():
    parser = argparse.ArgumentParser(description='Parameters to train your model.')
    parser.add_argument('--imgs_folder', default='./data/MSD/test/image', help='Path to folder containing images', type=str)
    parser.add_argument('--model_path', default='./models/alph-0.7_wbce_w0-1.0_w1-1.15/best_epoch-46_mae-0.0976_loss-0.2049.pth', help='Path to model', type=str)
    parser.add_argument('--use_gpu', default=True, help='Whether to use GPU or not', type=bool)
    parser.add_argument('--img_size', default=256, help='Image size to be used', type=int)
    parser.add_argument('--bs', default=1, help='Batch Size for testing', type=int)
    parser.add_argument('--raw', default=False, help='Save raw prediction', type=bool)

    return parser.parse_args()


def run_inference(args):
    # Determine device
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device(device='cuda')
    else:
        device = torch.device(device='cpu')

    # Load model
    model = SODModel()
    chkpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(chkpt['model'])
    model.to(device)
    model.eval()

    inf_data = InfDataloader(img_folder=args.imgs_folder, target_size=args.img_size)
    # Since the images would be displayed to the user, the batch_size is set to 1
    # Code at later point is also written assuming batch_size = 1, so do not change
    inf_dataloader = DataLoader(inf_data, batch_size=1, shuffle=True, num_workers=2)

    print("Press 'q' to quit.")
    with torch.no_grad():
        for batch_idx, (img_np, img_tor, img_name, _) in enumerate(inf_dataloader, start=1):
            img_tor = img_tor.to(device)
            pred_masks, _ = model(img_tor)

            # Assuming batch_size = 1
            img_np = np.squeeze(img_np.numpy(), axis=0)
            img_np = img_np.astype(np.uint8)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            pred_masks_raw = np.squeeze(pred_masks.cpu().numpy(), axis=(0, 1))
            pred_masks_round = np.squeeze(pred_masks.round().cpu().numpy(), axis=(0, 1))

            print('Image :', batch_idx)
            cv2.imshow('Input Image', img_np)
            cv2.imshow('Generated Saliency Mask', pred_masks_raw)
            cv2.imshow('Rounded-off Saliency Mask', pred_masks_round)

            print(img_name)

            key = cv2.waitKey(0)
            if key == ord('q'):
                break


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    # outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    SMOOTH = 1e-6

    intersection = (outputs.int() & labels.int()).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs.int() | labels.int()).float().sum((1, 2))         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded  # Or thresholded.mean() if you are interested in average across the batch


def calculate_mae(args):
    # Determine device
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device(device='cuda')
    else:
        device = torch.device(device='cpu')

    # Load model
    model = SODModel()
    chkpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(chkpt['model'])
    model.to(device)
    model.eval()

    test_data = SODLoader(mode='test', augment_data=False, target_size=args.img_size)
    test_dataloader = DataLoader(test_data, batch_size=args.bs, shuffle=False, num_workers=2)

    # List to save mean absolute error of each image
    mae_list = []
    iou_list = []
    with torch.no_grad():
        for batch_idx, (inp_imgs, gt_masks) in enumerate(tqdm.tqdm(test_dataloader), start=1):
            inp_imgs = inp_imgs.to(device)
            gt_masks = gt_masks.to(device)
            pred_masks, _ = model(inp_imgs)

            mae = torch.mean(torch.abs(pred_masks - gt_masks), dim=(1, 2, 3)).cpu().numpy()
            mae_list.extend(mae)
            iou = iou_pytorch(pred_masks.round(), gt_masks).cpu().numpy()
            iou_list.extend(iou)

    print('MAE for the test set is :', np.mean(mae_list))
    print('IoU for the test set is :', np.mean(iou_list))


def save_pred(args):
    # Determine device
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device(device='cuda')
    else:
        device = torch.device(device='cpu')

    # Load model
    model = SODModel()
    chkpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(chkpt['model'])
    model.to(device)
    model.eval()

    inf_data = InfDataloader(img_folder=args.imgs_folder, target_size=args.img_size)
    # Since the images would be displayed to the user, the batch_size is set to 1
    # Code at later point is also written assuming batch_size = 1, so do not change
    inf_dataloader = DataLoader(inf_data, batch_size=1, shuffle=True, num_workers=2)

    #directory to save the predictions
    pred_dir = './data/MSD/test/pred'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)

    with torch.no_grad():
        for batch_idx, (img_np, img_tor, img_name, hw) in enumerate(tqdm.tqdm(inf_dataloader), start=1):
            img_tor = img_tor.to(device)
            pred_masks, _ = model(img_tor)

            # Assuming batch_size = 1
            img_np = np.squeeze(img_np.numpy(), axis=0)
            img_np = img_np.astype(np.uint8)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            if args.raw is True:
            	pred_masks = np.squeeze(pred_masks.cpu().numpy(), axis=(0, 1))
            else:
            	pred_masks = np.squeeze(pred_masks.round().cpu().numpy(), axis=(0, 1))

            h,w = [int(x) for x in hw[0].split(' ')]

            s = max(h,w)
            pred_masks *= 255
            pred_masks = cv2.resize(pred_masks,(s,s),interpolation=cv2.INTER_AREA)
            
            offset_h = round((s - h)/2)
            offset_w = round((s - w)/2)
            p0, p1, p2, p3 = offset_h, s-offset_h, offset_w, s-offset_w
            pred_masks = pred_masks[p0:p1, p2:p3]

            cv2.imwrite(os.path.join(pred_dir,img_name[0] + '.png'), pred_masks)
            # print(img_name[0])
            
            # cv2.imshow('Original', img_np)
            # key = cv2.waitKey(0)
            # if key == ord('q'):
            #     break

if __name__ == '__main__':
    rt_args = parse_arguments()
    # calculate_mae(rt_args)
    # run_inference(rt_args)
    save_pred(rt_args)
