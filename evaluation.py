import os 
from skimage import io
from skimage.measure import label
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import warnings
import shutil
import cv2
import math
import argparse
import json
import shutil
import PIL
from PIL import Image

# plot fugure name_list : x axis ; num_list : y axis
def plot_figure(name_list, num_list):
    plt.bar(range(len(num_list)), num_list,color='rgb',tick_label=name_list)
    plt.show()
def get_mAP():
    pass

def save_txt_auto(args,data):
    
    with open(args.save_txt, "a") as file:
        for info in data:
            file.write(str(info))
            file.write("\n")
    print("txt saved to : ", args.save_txt)

    save_only_mean_score = args.save_txt.split(".")[0] + "_summary.txt"
    with open(save_only_mean_score, "a") as file:
        file.write(str(data[-1]))
        file.write("\n")
    print("txt saved to : ", save_only_mean_score)

def save_txt(save_path, data):
    with open(save_path, "w") as file:
        for info in data:
            file.write(str(info))
            file.write("\n")
    print("txt saved to : ", save_path)

# get non-mirror pixels (of GT mask)
def get_Nn(GT_mask):
    return len(np.where(GT_mask==0)[0])

# get mirror pixels (of GT mask)
def get_Np(GT_mask):
    return len(np.where(GT_mask!=0)[0])

def get_TP(pred_mask, GT_mask):
    TP_region =  np.logical_and(pred_mask, GT_mask)
    return len(np.where(TP_region==True)[0])


def get_FP(pred_mask, GT_mask):
    # pred_pos and GT_false
    GT_false = np.logical_not(GT_mask)
    FP_region = np.logical_and(pred_mask, GT_false)
    return len(np.where(FP_region==True)[0])

def get_FN(pred_mask, GT_mask):
    # pred_false and GT_true
    pred_false = np.logical_not(pred_mask)
    FN_region = np.logical_and(pred_false, GT_mask)
    return len(np.where(FN_region==True)[0])

def get_TN(pred_mask, GT_mask):
    # pred_false and GT_false
    GT_false = np.logical_not(GT_mask)
    pred_false = np.logical_not(pred_mask)
    TN_region = np.logical_and(GT_false, pred_false)
    return len(np.where(TN_region==True)[0])


def get_one_IoU(pred_mask, GT_mask):
    intersct = np.logical_and(pred_mask,GT_mask)
    union =  np.logical_or(pred_mask, GT_mask)
    IoU = np.sum(intersct) / np.sum(union)
    return IoU

def get_IOU(args):
    IOU_sum = 0
    IOU_list = []
    gt_mask_folder = args.gt_mask_folder
    image_path = args.image_path
    pred_mask_folder = args.pred_mask_folder
    for pred_mask_name in os.listdir(pred_mask_folder):
        pred_mask_path = os.path.join(pred_mask_folder, pred_mask_name)
        pred_mask = io.imread(pred_mask_path)
        GT_mask = io.imread(os.path.join(gt_mask_folder, pred_mask_name))
        IOU = get_one_IoU(pred_mask, GT_mask)
        IOU_sum += IOU
        raw_image_path = os.path.join(image_path, pred_mask_name.replace("png", "jpg"))
        IOU_list.append([raw_image_path,pred_mask_path,IOU ])
    IOU_list.append(["IOU_sum : " , IOU_sum / 955])
    print("IOU_sum : " , IOU_sum / 955)
    if args.output_automatically:
        save_txt_auto(args, IOU_list)
    else:
        save_txt(args.save_txt, IOU_list)
    print("get_IOU finishaed !")

# precision and recall in this funciton are measured by pixel level
def get_F_measure_pixel(args):
    f_measure_list = []
    sum_f_measure = 0
    beta = 0.3
    gt_mask_folder = args.gt_mask_folder
    image_path = args.image_path
    pred_mask_folder = args.pred_mask_folder
    for pred_mask_name in os.listdir(pred_mask_folder):
        pred_mask_path = os.path.join(pred_mask_folder, pred_mask_name)
        pred_mask = io.imread(pred_mask_path)
        raw_image_path = os.path.join(image_path, pred_mask_name.replace("png", "jpg"))
        GT_mask = io.imread(os.path.join(gt_mask_folder, pred_mask_name))
        TP = get_TP(pred_mask, GT_mask)
        FP = get_FP(pred_mask, GT_mask)
        FN = get_FN(pred_mask, GT_mask)
        try:
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f_measure = ((1+beta**2)*precision*recall )/((beta**2) * precision + recall)
        except:
            f_measure = 0
        sum_f_measure += f_measure
        f_measure_list.append([raw_image_path,pred_mask_path,f_measure])
        print(raw_image_path,pred_mask_path,f_measure)
    f_measure_list.append(["f_measure score : " , sum_f_measure / len(os.listdir(pred_mask_folder))])
    print("f_measure score : " , sum_f_measure / len(os.listdir(pred_mask_folder)))
    if args.output_automatically:
        save_txt_auto(args, f_measure_list)
    else:
        save_txt(args.save_txt, f_measure_list)
    print("get_F_measure_pixel finishaed !")

def get_BER(args):
    BER_list = []
    sum_BER = 0
    beta = 0.3
    gt_mask_folder = args.gt_mask_folder
    image_path = args.image_path
    pred_mask_folder = args.pred_mask_folder
    for pred_mask_name in os.listdir(pred_mask_folder):
        pred_mask_path = os.path.join(pred_mask_folder, pred_mask_name)
        pred_mask = io.imread(pred_mask_path)
        raw_image_path = os.path.join(image_path, pred_mask_name.replace("png", "jpg"))
        GT_mask = io.imread(os.path.join(gt_mask_folder, pred_mask_name))
        TP = get_TP(pred_mask, GT_mask)
        TN = get_TN(pred_mask, GT_mask)
        Np = get_Np(GT_mask)
        Nn = get_Nn(GT_mask)
        try:
            BER = 100*(1-0.5*(TP/Np + TN/Nn))
        except:
            BER = 0
        sum_BER += BER
        BER_list.append([raw_image_path,pred_mask_path,BER])
        print(raw_image_path,pred_mask_path,BER)
    BER_list.append(["BER score : " , sum_BER / len(os.listdir(pred_mask_folder))])
    print("BER score : " , sum_BER / len(os.listdir(pred_mask_folder)))
    if args.output_automatically:
        save_txt_auto(args, BER_list)
    else:
        save_txt(args.save_txt, BER_list)
    print("get_BER_pixel finishaed !")


def get_MAE(args):
    MAE_list = []
    sum_MAE = 0
    gt_mask_folder = args.gt_mask_folder
    image_path = args.image_path
    pred_mask_folder = args.pred_mask_folder
    for pred_mask_name in os.listdir(pred_mask_folder):
        pred_mask_path = os.path.join(pred_mask_folder, pred_mask_name)
        pred_mask = io.imread(pred_mask_path)
        raw_image_path = os.path.join(image_path, pred_mask_name.replace("png", "jpg"))
        GT_mask = io.imread(os.path.join(gt_mask_folder, pred_mask_name))
        FN = get_FN(pred_mask, GT_mask)
        FP = get_FP(pred_mask, GT_mask)
        w, h  = GT_mask.shape
        try:
            MAE = (FN + FP) / (w*h)
        except:
            MAE = 0
        sum_MAE += MAE
        MAE_list.append([raw_image_path,pred_mask_path,MAE])
        print(raw_image_path,pred_mask_path,MAE)
    MAE_list.append(["MAE score : " , sum_MAE / len(os.listdir(pred_mask_folder))])
    print("MAE score : " , sum_MAE / len(os.listdir(pred_mask_folder)))
    if args.output_automatically:
        save_txt_auto(args, MAE_list)
    else:
        save_txt(args.save_txt, MAE_list)
    print("get_BER_pixel finishaed !")

def get_Acc(args):
    Acc_list = []
    sum_Acc = 0
    gt_mask_folder = args.gt_mask_folder
    image_path = args.image_path
    pred_mask_folder = args.pred_mask_folder
    for pred_mask_name in os.listdir(pred_mask_folder):
        pred_mask_path = os.path.join(pred_mask_folder, pred_mask_name)
        pred_mask = io.imread(pred_mask_path)
        raw_image_path = os.path.join(image_path, pred_mask_name.replace("png", "jpg"))
        GT_mask = io.imread(os.path.join(gt_mask_folder, pred_mask_name))
        TP = get_TP(pred_mask, GT_mask)
        Np = get_Np(GT_mask)
        w, h  = GT_mask.shape
        try:
            Acc = TP / Np
        except:
            Acc = 0
        sum_Acc += Acc
        Acc_list.append([raw_image_path,pred_mask_path,Acc])
        print(raw_image_path,pred_mask_path,Acc)
    Acc_list.append(["Acc score : " , sum_Acc / len(os.listdir(pred_mask_folder))])
    print("Acc score : " , sum_Acc / len(os.listdir(pred_mask_folder)))
    if args.output_automatically:
        save_txt_auto(args, Acc_list)
    else:
        save_txt(args.save_txt, Acc_list)
    print("get_BER_pixel finishaed !")

def save_json(data,save_path):
    out_json = json.dumps(data, sort_keys=False, indent=4, separators=(',', ':'),
                          ensure_ascii=False)
    fo = open(save_path, "w")
    fo.write(out_json)
    fo.close()
    print("json file saved to : ",save_path )

def combine_and_save(img_list, save_path):
    # list_im = ['Test1.jpg', 'Test2.jpg', 'Test3.jpg']
    imgs = [ PIL.Image.open(i).convert('RGB') for i in img_list ]
    # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
    min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
    imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )

    # save that beautiful picture
    imgs_comb = PIL.Image.fromarray( imgs_comb)
    imgs_comb.save(save_path)    
    print("combined image saved to :", save_path)



# 获取分数低的样本
def get_low_IOU_sample(args):
    IOU_sum = 0
    IOU_list = []
    raw_path_info = dict()
    
    gt_mask_folder = args.gt_mask_folder
    image_path = args.image_path
    pred_mask_folder = args.pred_mask_folder
    for pred_mask_name in os.listdir(pred_mask_folder):
        pred_mask_path = os.path.join(pred_mask_folder, pred_mask_name)
        pred_mask = io.imread(pred_mask_path)
        GT_mask = io.imread(os.path.join(gt_mask_folder, pred_mask_name))
        IOU = get_one_IoU(pred_mask, GT_mask)
        IOU_sum += IOU
        raw_image_path = os.path.join(image_path, pred_mask_name.replace("png", "jpg"))
        IOU_list.append(IOU)
        raw_path_info[raw_image_path] = [IOU,pred_mask_path]
        
        # copy sample to folder by IOU score 
        save_folder_path = os.path.join(args.sample_save_folder, str(int(IOU*10))) 
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
        sample_save_path = os.path.join(save_folder_path, pred_mask_name)
        combine_and_save([raw_image_path,pred_mask_path],sample_save_path)
        # save_fold
        
    # IOU_list.append(["IOU_sum : " , IOU_sum / 955])
    # print("IOU_sum : " , IOU_sum / 955)
    # if args.output_automatically:
    #     save_txt_auto(args, IOU_list)
    # else:
    #     save_txt(args.save_txt, IOU_list)
    # print("get_IOU finishaed !")

    # save_json(raw_path_info, "result/mirrorNet_result.json")
    # plt.xlabel("IoU")
    # plt.ylabel("Sample number")
    # plt.title("IoU distribution")
    # plt.hist(IOU_list, bins = 10)
    # plt.show()

# 获取一开始数据的ratio分布 （分析dataset数据情况）
def get_mask_ratio_distribution(args):
    ratio_list = []
    raw_path_info = dict()
    
    gt_mask_folder = args.gt_mask_folder
    image_path = args.image_path
    pred_mask_folder = args.pred_mask_folder
    for pred_mask_name in os.listdir(pred_mask_folder):
        pred_mask_path = os.path.join(pred_mask_folder, pred_mask_name)
        pred_mask = io.imread(pred_mask_path)
        GT_mask = io.imread(os.path.join(gt_mask_folder, pred_mask_name))
        ratio = len(np.where(GT_mask>0)[0]) / (GT_mask.shape[0] *  GT_mask.shape[1])
        raw_image_path = os.path.join(image_path, pred_mask_name.replace("png", "jpg"))
        ratio_list.append(ratio)

    plt.xlabel("ratio = mirror size / image size")
    plt.ylabel("sample number")
    plt.title("mirror size (ratio) distribution")
    plt.hist(ratio_list, bins = 10)
    plt.show()

# 获取镜子大小和最后分数的关系
def get_mask_ratio_and_IOU_relation(args):
    ratio_list = []
    IOU_list = []
    raw_path_info = dict()
    
    gt_mask_folder = args.gt_mask_folder
    image_path = args.image_path
    pred_mask_folder = args.pred_mask_folder
    for pred_mask_name in os.listdir(pred_mask_folder):
        pred_mask_path = os.path.join(pred_mask_folder, pred_mask_name)
        pred_mask = io.imread(pred_mask_path)
        GT_mask = io.imread(os.path.join(gt_mask_folder, pred_mask_name))
        ratio = len(np.where(GT_mask>0)[0]) / (GT_mask.shape[0] *  GT_mask.shape[1])
        IOU = get_one_IoU(pred_mask, GT_mask)
        IOU_list.append(IOU)
        raw_image_path = os.path.join(image_path, pred_mask_name.replace("png", "jpg"))
        ratio_list.append(ratio)

    plt.xlabel("ratio = mirror size / image size")
    plt.ylabel("IOU")
    plt.title("ratio VS IOU")
    # plt.hist(ratio_list, bins = 10)
    plt.scatter(ratio_list, IOU_list, alpha=0.3)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
    plt.show()



if __name__ == "__main__":

    # ###################### INSTRUCTION ######################
    # input requirement :
    #     (1) mask type : 1 channel iamge 
    #     (2) pred_mask_name = GT_mask_name )
    # get 6 types of score together : 
    #       --output_automatically True --stage 6 
    parser = argparse.ArgumentParser(description='Get Setting :D')
    parser.add_argument(
        '--gt_mask_folder', default="C:\\Users\\yipeng\\OneDrive - sfu.ca\\Courses\\CMPT 726\\Project\\msd\\test", type=str)
    parser.add_argument(
        '--image_path', default="C:\\Users\\yipeng\\OneDrive - sfu.ca\\Courses\\CMPT 726\\Project\\msd\\test", type=str)
    parser.add_argument(
        '--pred_mask_folder', default="C:\\Users\\yipeng\\OneDrive - sfu.ca\\Courses\\CMPT 726\\Project\\results\\results_msd_6000", type=str)
    parser.add_argument(
        '--save_txt', default="result/test_result_ratio.txt", type=str, help="output txt save path")
    parser.add_argument(
        '--sample_save_folder', default="C:\\Users\\yipeng\\OneDrive - sfu.ca\\Courses\\CMPT 726\\Project", \
            type=str, help="folder to save selected sample (parameter used in selecting low score sampe)")
    parser.add_argument(
        '--output_automatically', default=True, type=bool , help="run all stages and save result automatically")
    parser.add_argument(
        '--stage', default="9", type=str , help="(1) IOU \
                                                (2)F measure\
                                                (3) BER\
                                                (4) MEA\
                                                (5) pixel accuracy\
                                                (6) get all score\
                                                (9) plot ratio VS IoU")
    args = parser.parse_args()
    

    # get IoU for MirrorNet
    if args.stage == "1":
        get_IOU(args)
    if args.stage == "2":
        get_F_measure_pixel(args)
    if args.stage == "3":
        get_BER(args)
    if args.stage == "4":
        get_MAE(args)
    if args.stage == "5":
        get_Acc(args)
    # run all together
    if args.stage == "6":
        get_IOU(args)
        get_F_measure_pixel(args)
        get_BER(args)
        get_MAE(args)
        get_Acc(args)
    if args.stage == "7":
        get_low_IOU_sample(args)
    if args.stage == "8":
        get_mask_ratio_distribution(args)
    if args.stage == "9":
        get_mask_ratio_and_IOU_relation(args)