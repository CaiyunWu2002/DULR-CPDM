
##import models
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import logging
import os
import os.path
import time
from typing import Any, Dict, List


import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm

#Import custom modules.
# from data.dataset_denoising import DatasetDenoising
from data.dataset_polar import DatasetPolar
from data.select_dataset import select_dataset
from models.model import Model
from utils import utils_image as util
from utils import utils_logger
from utils import utils_option as option

def main(config_path: str = 'options/test_cpdm.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt',
                        type=str,
                        default=config_path,
                        help='Path to option JSON file.')
    opt = option.parse(parser.parse_args().opt, is_train=True)
    util.makedirs(
        [path for key, path in opt['path'].items() if 'pretrained' not in key])

    option.save(opt)

    # logger
    logger_name = 'test'
    utils_logger.logger_info(
        logger_name, os.path.join(opt['path']['log'], logger_name + '.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))

    # data
    opt_data_test = opt["data"]["test"]
    test_sets: List[DatasetPolar] = select_dataset(opt_data_test, "test")
    test_loaders: List[DataLoader[DatasetPolar]] = []
    for test_set in test_sets:
        test_loaders.append(
            DataLoader(test_set,
                       batch_size=1,
                       shuffle=False,
                       num_workers=1,
                       drop_last=False,
                       pin_memory=True))

    # model
    model = Model(opt)
    model.init()
    start = time.time()
    all_infer_time = []

    test_index = 0
    epoch = 0
    for test_loader in tqdm(test_loaders, desc='DataSet'):
        for test_data in tqdm(test_loader, desc='Sample', leave=False):
            test_index += 1
            model.feed_data(test_data)

            infer_start = time.time()
            model.test()
            infer_end = time.time()
            infer_time = (infer_end - infer_start) * 1000
            all_infer_time.append(infer_time)
            
            model.save_visuals(epoch)

    total_time = time.time() - start
    if all_infer_time:
        avg_infer_time = np.mean(all_infer_time)
        min_infer_time = np.min(all_infer_time)
        max_infer_time = np.max(all_infer_time)
        std_infer_time = np.std(all_infer_time)
        total_infer_time = np.sum(all_infer_time)
        print("\n" + "=" * 60)
        print(" Inference Performance Statistics")
        print("=" * 60)
        print(f"Test count: {test_index}")
        print(f"Average inference time: {avg_infer_time:.2f} ms")
        print(f"Minimum inference time: {min_infer_time:.2f} ms")
        print(f"Maximum inference time: {max_infer_time:.2f} ms")
        print(f"Standard deviation: {std_infer_time:.2f} ms")
        print(f"Total inference time: {total_infer_time:.2f} ms ({total_infer_time / 1000:.2f} s)")
        print(f"Average FPS: {1000 / avg_infer_time:.2f} img/s")
        print(f"Total time (including data loading): {total_time:.2f} s")
        print("=" * 60 + "\n")

        logger.info(
            f"Inference completed - Samples: {test_index}, Average time: {avg_infer_time:.2f}ms, FPS: {1000 / avg_infer_time:.2f}")
        

if __name__ == '__main__':
    main()
