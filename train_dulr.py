##import models
import argparse
import faulthandler
import logging
import os
import os.path
import random
import time
from typing import Any, Dict, List

##import Third party libraries
import pandas as pd
import numpy as np
import torch
from prettytable import PrettyTable
from torch import cuda
from torch.utils.data import DataLoader
from tqdm import tqdm

#Import custom modules.
from data.dataset_polar import DatasetPolar
from data.select_dataset import select_dataset
from models.model import Model
from utils import utils_image as util
from utils import utils_logger
from utils import utils_option as option
import matplotlib.pyplot as plt

faulthandler.enable()
torch.autograd.set_detect_anomaly(True)

def main(json_path: str =
    'options/train_cpdm_s2.json'):
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt',
                        type=str,
                        default=json_path,
                        help='Path to option JSON file.')

    opt = option.parse(parser.parse_args().opt, is_train=True)
    util.makedirs(
        [path for key, path in opt['path'].items() if 'pretrained' not in key])
    option.save(opt)

    # logger
    logger_name = 'train'
    utils_logger.logger_info(
        logger_name, os.path.join(opt['path']['log'], logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    # seed
    seed = opt['train']['manual_seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda.manual_seed_all(seed)

    # data
    opt_data_train: Dict[str, Any] = opt["data"]["train"]
    train_set: DatasetPolar = select_dataset(opt_data_train, "train")

    train_loader: DataLoader[DatasetPolar] = DataLoader(
        train_set,
        batch_size=opt_data_train['batch_size'],
        shuffle=True,
        num_workers=opt_data_train['num_workers'],
        drop_last=True,
        pin_memory=True,
        persistent_workers = True,
        prefetch_factor = 2)

    opt_data_test = opt["data"]["test"]
    test_sets: List[DatasetPolar] = select_dataset(opt_data_test, "test")
    test_loaders: List[DataLoader[DatasetPolar]] = []
    for test_set in test_sets:
        test_loaders.append(
            DataLoader(test_set,
                       batch_size=1,
                       shuffle=False,
                       num_workers=1,
                       drop_last=True,
                       pin_memory=True))

    # model
    model = Model(opt)
    model.init()

    # train
    start = time.time()
    start_epoch = 0
    current_step = 0
    losses = []

    best_loss = float('inf')
    best_model_path = os.path.join(opt['path']['models'], 'best_model.pth')

    checkpoint_dir = os.path.join(opt['path']['models'], 'checkpoints')
    latest_checkpoint = os.path.join(checkpoint_dir, 'latest.pth')

    train_choice = input("Continue training? (Enter 'y' for yes, others for no): ")
    if train_choice.lower() != 'y':
        print("Training canceled.")
    else:
        if os.path.exists(latest_checkpoint):
            start_epoch, current_step, losses = model.load_checkpoint(latest_checkpoint)
            logger.info(
                f"Successfully resumed training state! Will continue from epoch {start_epoch}, step {current_step}")
            for scheduler in model.schedulers:
                scheduler.last_epoch = start_epoch
                logger.info(f"Manually set scheduler last_epoch: {scheduler.last_epoch} (matches completed epochs)")
        else:
            logger.info("Checkpoint not found or not resuming training, starting from scratch")

        start_epoch=start_epoch+1

    model.save(logger, start_epoch-1)
    total_epochs = opt['train']['total_epochs']
    for epoch in range(start_epoch, total_epochs): # keep running

        model.net.train()
        model.loss_history=[]
        model.loss_history_s1 = []
        model.loss_history_s2= []
        for train_data in tqdm(train_loader):
            current_step += 1

            model.feed_data(train_data)
            model.train()

            current_loss = model.log_dict['G_loss']
            losses.append(current_loss)
            aveloss = sum(model.loss_history) / len(model.loss_history)

            if current_step % opt['train']['checkpoint_save_interval'] == 0:
                model.save_checkpoint(epoch, current_step, opt['path']['models'],1)
                logger.info(f"Saved checkpoint at step {current_step}")


            if current_step % opt['train']['checkpoint_log'] == 0:
                model.log_train(current_step, epoch, logger,aveloss)

            if current_step % opt['train']['checkpoint_test'] == 0:
                if aveloss < best_loss:
                    best_loss = aveloss
                    model.save(logger, epoch)
                    model.save_checkpoint(epoch, current_step, opt['path']['models'],1)
                    logger.info(f"Saved checkpoint at step {current_step}")

                test_index = 0
                all_metrics_data = []

                for test_loader in tqdm(test_loaders):
                    test_set: DatasetPolar = test_loader.dataset
                    for test_data in tqdm(test_loader):
                        test_index += 1
                        model.feed_data(test_data)
                        model.test()
                        psnr_ssim_dict = model.cal_metrics(model.out_dict['y_gt'],model.out_dict['dx'])
                        all_metrics_data.append( psnr_ssim_dict)
                        model.save_visuals(epoch)


                metrics_df = pd.DataFrame(all_metrics_data)
                save_path = opt["data"]["test"]["save_path"]
                os.makedirs(save_path, exist_ok=True)
                metrics_df.to_excel(os.path.join(save_path, f"{epoch}all_metrics_data_results.xlsx"), index=False)
                logger.info(f"Time elapsed: {time.time() - start:.2f}")
                start = time.time()
      
        for scheduler in model.schedulers:
            print(f"Before calling scheduler.step() in Epoch {epoch}, last_epoch={scheduler.last_epoch}")
            scheduler.step()

        current_lr = model.optimizer.param_groups[0]['lr']
        print(f"Current LR: {current_lr:.6f}")



if __name__ == '__main__':
    main()
