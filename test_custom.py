
import numpy as np
import os,sys
import argparse

import torchvision
from tqdm import tqdm
from einops import rearrange, repeat

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
# from ptflops import get_model_complexity_info

import scipy.io as sio

from adapter import dataset_loader
from utils.loader import get_validation_data
import utils
import cv2
from model import UNet

from skimage import img_as_float32, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from sklearn.metrics import mean_squared_error as mse_loss

parser = argparse.ArgumentParser(description='RGB denoising evaluation on the validation set of SIDD')
parser.add_argument('--input_dir', default='../ISTD_Dataset/test/',
    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/',
    type=str, help='Directory for results')
parser.add_argument('--weights', default='./log/ShadowFormer_istd/models/model_best.pth',
    type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--arch', default='ShadowFormer', type=str, help='arch')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
parser.add_argument('--cal_metrics', action='store_true', help='Measure denoised images with GT')
parser.add_argument('--embed_dim', type=int, default=32, help='number of data loading workers')    
parser.add_argument('--win_size', type=int, default=10, help='number of data loading workers')
parser.add_argument('--token_projection', type=str, default='linear', help='linear/conv token projection')
parser.add_argument('--token_mlp', type=str,default='leff', help='ffn/leff token mlp')
# args for vit
parser.add_argument('--vit_dim', type=int, default=256, help='vit hidden_dim')
parser.add_argument('--vit_depth', type=int, default=12, help='vit depth')
parser.add_argument('--vit_nheads', type=int, default=8, help='vit hidden_dim')
parser.add_argument('--vit_mlp_dim', type=int, default=512, help='vit mlp_dim')
parser.add_argument('--vit_patch_size', type=int, default=16, help='vit patch_size')
parser.add_argument('--global_skip', action='store_true', default=False, help='global skip connection')
parser.add_argument('--local_skip', action='store_true', default=False, help='local skip connection')
parser.add_argument('--vit_share', action='store_true', default=False, help='share vit module')
parser.add_argument('--train_ps', type=int, default=320, help='patch size of training sample')
parser.add_argument('--tile', type=int, default=None, help='Tile size (e.g 720). None means testing on the original resolution image')
parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
args = parser.parse_args()
utils.mkdir(args.result_dir)

def main():
    ws_istd = "X:/ISTD_Dataset/test/test_A/*.png"
    ns_istd = "X:/ISTD_Dataset/test/test_C/*.png"
    mask_istd = "X:/ISTD_Dataset/test/test_B/*.png"

    opts = {}
    opts["num_workers"] = 12
    opts["cuda_device"] = "cuda:0"
    opts["img_to_load"] = -1
    test_loader_istd = dataset_loader.load_istd_dataset(ws_istd, ns_istd, mask_istd, 32, opts)
    save_dir_istd = "./reports/ISTD/"

    ws_istd = "X:/SRD_Test/srd/shadow/*.jpg"
    ns_istd = "X:/SRD_Test/srd/shadow_free/*.jpg"
    mask_istd = "X:/SRD_Test/srd/mask/*.jpg"
    test_loader_srd = dataset_loader.load_srd_dataset(ws_istd, ns_istd, mask_istd, 32, opts)
    save_dir_srd = "./reports/SRD/"

    # test_dataset = get_validation_data(args.input_dir)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(device)
    print("Torch CUDA version: %s" % torch.version.cuda)

    model_restoration = utils.get_arch(args)
    model_restoration = torch.nn.DataParallel(model_restoration)
    model_restoration.to(device)

    print("===>Testing using weights: ", args.weights)
    utils.load_checkpoint(model_restoration, args.weights)


    model_restoration.cuda()
    model_restoration.eval()

    with torch.no_grad():
        for ii, (file_name, rgb_ws, rgb_ns, shadow_matte) in enumerate(tqdm(test_loader_istd), 0):
            rgb_gt = rgb_ns.to(device)
            rgb_noisy = rgb_ws.to(device)
            mask = shadow_matte.to(device)

            rgb_restored = model_restoration(rgb_noisy, mask)
            rgb_restored = torch.clamp(rgb_restored, 0.0, 1.0)
            rgb_restored = torch.nn.functional.interpolate(rgb_restored, (240, 320))
            for j in range(0, np.size(file_name)):
                impath = save_dir_istd + file_name[j] + ".png"
                torchvision.utils.save_image(rgb_restored[j], impath, normalize=True)
                print("Saving " + impath)

        for ii, (file_name, rgb_ws, rgb_ns, shadow_matte) in enumerate(tqdm(test_loader_srd), 0):
            rgb_gt = rgb_ns.to(device)
            rgb_noisy = rgb_ws.to(device)
            mask = shadow_matte.to(device)

            rgb_restored = model_restoration(rgb_noisy, mask)
            rgb_restored = torch.clamp(rgb_restored, 0.0, 1.0)
            rgb_restored = torch.nn.functional.interpolate(rgb_restored, (160, 210))

            for j in range(0, np.size(file_name)):
                impath = save_dir_srd + file_name[j] + ".png"
                torchvision.utils.save_image(rgb_restored[j], impath, normalize=True)
                print("Saving " + impath)


if __name__ == "__main__":
    main()

