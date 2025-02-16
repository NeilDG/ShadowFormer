import os
import sys

import yaml
from yaml import SafeLoader

from adapter import dataset_loader

import argparse
import options


import utils
import torch
# from piqa import SSIM
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from natsort import natsorted
import glob
import random
import time
import numpy as np
from einops import rearrange, repeat
import datetime
from pdb import set_trace as stx
from utils import save_img
from losses import CharbonnierLoss

from tqdm import tqdm 
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler

from utils.loader import get_training_data, get_validation_data


def main():
    # add dir
    dir_name = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(dir_name, './auxiliary/'))
    print(dir_name)

    ######### parser ###########
    opt = options.Options().init(argparse.ArgumentParser(description='image denoising')).parse_args()
    print(opt)

    ######### Logs dir ###########
    log_dir = os.path.join(dir_name, 'log', opt.arch+opt.env)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # logname = os.path.join(log_dir, datetime.datetime.now().isoformat()+'.txt')
    logname = os.path.join(log_dir, 'train_logs.txt')
    print("Now time is : ", datetime.datetime.now().isoformat())
    result_dir = os.path.join(log_dir, 'reports')
    model_dir  = os.path.join(log_dir, 'models')
    utils.mkdir(result_dir)
    utils.mkdir(model_dir)

    # ######### Set Seeds ###########
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)



    ######### Model ###########
    model_restoration = utils.get_arch(opt)

    with open(logname,'a') as f:
        f.write(str(opt)+'\n')
        f.write(str(model_restoration)+'\n')

    ######### Optimizer ###########
    start_epoch = 0
    if opt.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
    elif opt.optimizer.lower() == 'adamw':
            optimizer = optim.AdamW(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
    else:
        raise Exception("Error optimizer...")

    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(device)
    print("Torch CUDA version: %s" % torch.version.cuda)

    ######### DataParallel ###########
    model_restoration = torch.nn.DataParallel (model_restoration)
    model_restoration.to(device)

    ######### Resume ###########
    if opt.resume:
        path_chk_rest = opt.pretrain_weights
        utils.load_checkpoint(model_restoration,path_chk_rest)
        start_epoch = utils.load_start_epoch(path_chk_rest) + 1
        print("------------------Successfully loaded ShadowFormer network: ", path_chk_rest, "------------------")
    #     lr = utils.load_optim(optimizer, path_chk_rest)
    #
    #     for p in optimizer.param_groups: p['lr'] = lr
    #     warmup = False
    #     new_lr = lr
    #     print('------------------------------------------------------------------------------')
    #     print("==> Resuming Training with learning rate:",new_lr)
    #     print('------------------------------------------------------------------------------')
    #     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-start_epoch+1, eta_min=1e-6)

    # ######### Scheduler ###########
    if opt.warmup:
        print("Using warmup and cosine strategy!")
        warmup_epochs = opt.warmup_epochs
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-warmup_epochs, eta_min=1e-6)
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
        scheduler.step()
    else:
        step = 50
        print("Using StepLR,step={}!".format(step))
        scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
        scheduler.step()

    # plot utils
    plot_loss_path = "./reports/train_test_loss.yaml"
    l1_loss = nn.L1Loss()
    if (os.path.exists(plot_loss_path)):
        with open(plot_loss_path) as f:
            losses_dict = yaml.load(f, SafeLoader)
    else:
        losses_dict = {}
        losses_dict["train"] = []
        losses_dict["test_istd"] = []

    print("Losses dict: ", losses_dict["train"])
    current_step = 0
    required_step_per_save = 500

    ######### Loss ###########
    criterion = CharbonnierLoss().cuda()

    ######### DataLoader ###########
    print('===> Loading datasets')
    # img_options_train = {'patch_size':opt.train_ps}
    # train_dataset = get_training_data(opt.train_dir, img_options_train)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True,
    #         num_workers=opt.train_workers, pin_memory=True, drop_last=False)
    #
    # val_dataset = get_validation_data(opt.val_dir)
    # val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False,
    #         num_workers=opt.eval_workers, pin_memory=False, drop_last=False)

    ### data loader
    rgb_dir_ws = "X:/SynthWeather Dataset 10/{dataset_version}/rgb/*/*.*"
    rgb_dir_ns = "X:/SynthWeather Dataset 10/{dataset_version}/rgb_noshadows/*/*.*"
    rgb_dir_ws = rgb_dir_ws.format(dataset_version="v69_places")
    rgb_dir_ns = rgb_dir_ns.format(dataset_version="v69_places")

    ws_istd = "X:/ISTD_Dataset/test/test_A/*.png"
    ns_istd = "X:/ISTD_Dataset/test/test_C/*.png"
    mask_istd = "X:/ISTD_Dataset/test/test_B/*.png"

    opts = {}
    opts["img_to_load"] = 10000
    opts["num_workers"] = 12
    opts["cuda_device"] = "cuda:0"
    train_load_size = 16
    train_loader = dataset_loader.load_shadow_train_dataset(rgb_dir_ws, rgb_dir_ns, ws_istd, ns_istd, train_load_size, opts=opts)
    val_loader = dataset_loader.load_istd_dataset(ws_istd, ns_istd, mask_istd, 8, opts)

    len_trainset = opts["img_to_load"]
    len_valset = 540
    print("Sizeof training set: ", len_trainset,", sizeof validation set: ", len_valset)

    ######### train ###########
    print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.nepoch))
    best_psnr = 0
    best_epoch = 0
    best_iter = 0
    print("\nEvaluation after every {} Iterations !!!\n".format(required_step_per_save))

    # compute total progress
    max_epochs = opt.nepoch
    dataset_count = opts["img_to_load"]
    needed_progress = int(max_epochs * (dataset_count / train_load_size))
    current_progress = int(start_epoch * (dataset_count / train_load_size))
    pbar = tqdm(total=needed_progress)
    pbar.update(current_progress)

    loss_scaler = NativeScaler()
    torch.cuda.empty_cache()
    ii=0
    index = 0
    for epoch in range(start_epoch, opt.nepoch + 1):
        epoch_start_time = time.time()
        epoch_loss = 0
        train_id = 1
        epoch_ssim_loss = 0
        for i, (_, rgb_ws, rgb_ns, shadow_map, shadow_matte) in enumerate(train_loader, 0):
            # zero_grad
            index += 1
            current_step = current_step + 1
            optimizer.zero_grad()
            input_ = rgb_ws.to(device)
            target = rgb_ns.to(device)
            mask = shadow_matte.to(device)
            if epoch > 5:
                target, input_, mask = utils.MixUp_AUG().aug(target, input_, mask)
            with torch.cuda.amp.autocast():
                restored = model_restoration(input_, mask)
                restored = torch.clamp(restored,0,1)
                loss = criterion(restored, target)
            loss_scaler(
                    loss, optimizer,parameters=model_restoration.parameters())
            epoch_loss +=loss.item()
            pbar.update(1)

            #### Evaluation ####
            if (current_step % required_step_per_save == 0):
                train_loss = float(np.round(l1_loss(restored, target).item(), 4))
                losses_dict["train"].append({current_step: train_loss})

                eval_shadow_rmse = 0
                eval_nonshadow_rmse = 0
                eval_rmse = 0
                with torch.no_grad():
                    model_restoration.eval()
                    psnr_val_rgb = []
                    for ii, (file_name, rgb_ws, rgb_ns, shadow_matte) in enumerate((val_loader), 0):
                        input_ = rgb_ws.to(device)
                        target_val = rgb_ns.to(device)
                        mask = shadow_matte.to(device)

                        with torch.cuda.amp.autocast():
                            restored_val = model_restoration(input_, mask)
                        restored_val = torch.clamp(restored_val,0,1)
                        psnr_val_rgb.append(utils.batch_PSNR(restored_val, target_val, False).item())

                    psnr_val_rgb = sum(psnr_val_rgb)/len(val_loader)
                    if psnr_val_rgb > best_psnr:
                        best_psnr = psnr_val_rgb
                        best_epoch = epoch
                        best_iter = i
                        torch.save({'epoch': epoch,
                                    'state_dict': model_restoration.state_dict(),
                                    'optimizer' : optimizer.state_dict()
                                    }, os.path.join(model_dir,"model_best.pth"))
                    print("[Ep %d it %d\t PSNR : %.4f] " % (epoch, i, psnr_val_rgb))
                    with open(logname,'a') as f:
                        f.write("[Ep %d it %d\t PSNR SIDD: %.4f\t] ----  [best_Ep_SIDD %d best_it_SIDD %d Best_PSNR_SIDD %.4f] " \
                            % (epoch, i, psnr_val_rgb,best_epoch,best_iter,best_psnr)+'\n')
                    model_restoration.train()
                    torch.cuda.empty_cache()

                    test_loss = float(np.round(l1_loss(restored_val, target_val).item(), 4))
                    losses_dict["test_istd"].append({current_step: test_loss})

        plot_loss_file = open(plot_loss_path, "w")
        yaml.dump(losses_dict, plot_loss_file)
        plot_loss_file.close()
        print("Dumped train test loss to ", plot_loss_path)

        scheduler.step()

        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss,scheduler.get_lr()[0]))
        print("------------------------------------------------------------------")
        with open(logname,'a') as f:
            f.write("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss, scheduler.get_lr()[0])+'\n')

        torch.save({'epoch': epoch,
                    'state_dict': model_restoration.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(model_dir,"model_latest.pth"))


        torch.save({'epoch': epoch,
                    'state_dict': model_restoration.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(model_dir,"model_epoch_{}.pth".format(epoch)))
    print("Now time is : ",datetime.datetime.now().isoformat())

if __name__ == "__main__":
    main()



