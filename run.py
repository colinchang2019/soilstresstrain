from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
import os
from config import cfg
from model import PhysicalLSTM2 as PhysicalLSTM
from utils.earlyStopping import EarlyStopping
from dataset import CSVDataset
from utils.loss import pyMseLoss

LR, STEP_SIZE, GAMMA = cfg.parameters

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    # cudnn.benchmark = False
    # cudnn.enabled = False

setup_seed(2019)

def _init_fn(worker_id):
    random.seed(10 + worker_id)
    np.random.seed(10 + worker_id)
    torch.manual_seed(10 + worker_id)
    torch.cuda.manual_seed(10 + worker_id)
    torch.cuda.manual_seed_all(10 + worker_id)

import logging

logging.basicConfig(filename='logger.log', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

logger = logging.getLogger(__name__) 
logger.setLevel(level=logging.INFO) 

pathlog = "log_train.txt"
filehandler = logging.FileHandler(pathlog) 
filehandler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
filehandler.setFormatter(formatter)

console = logging.StreamHandler() 
console.setLevel(logging.INFO)
console.setFormatter(formatter)

logger.addHandler(filehandler)
logger.addHandler(console)

logger.info("Start log")
logger.info("Parametes:　LR: {}, STEP_SIZE:{}, GAMMA: {}".format(LR, STEP_SIZE, GAMMA))


def main(path=cfg.pathm):
    """
    :param path: "C:/Users/chesley/Pictures/ne7/second" as default
    :return:
    """
    logger.info("Run in file: {}".format(path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # create dataloader
    print("Laoding dataset to torch.")
    trainDataset = CSVDataset(cfg.preTransform, cfg.last, cfg.trains, cfg.train_size, cfg.src_len, shuffle=False) 
    trainDataloader = DataLoader(
        dataset=trainDataset,
        batch_size=cfg.batch,
        shuffle=False,
        num_workers=cfg.num_workers,
        worker_init_fn=_init_fn
    )
    testDataset = CSVDataset(cfg.preTransform, cfg.last, cfg.tests, cfg.test_size, cfg.src_len, shuffle=False)
    testDataloader = DataLoader(
        dataset=testDataset,
        batch_size=cfg.batch,
        shuffle=False,
        num_workers=cfg.num_workers
    )
    print("Dataset prepared.")
    torch.cuda.empty_cache() 

    model = PhysicalLSTM().to(device)
    # loss_fn = nn.MSELoss()
    loss_fn = pyMseLoss()
    loss_fn.to(device)

    optimiser = optim.Adam(params=model.parameters(), lr=LR) 

    print("Start Train.")
    num_epochs = cfg.num_epochs
    total_step = len(trainDataloader)
    loss_List = []
    loss_test_list = []

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=STEP_SIZE, gamma=GAMMA)

    # save_path for model in different case
    path_m = path + "press.pth"

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=cfg.patience, verbose=False, path=path_m)

    if os.path.exists(path_m):
        print(path_m)
        model.load_state_dict(torch.load(path_m)["state_dict"])

    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        model.train()
        totalLoss = 0
        print("Epoch: ", epoch)
        for i, (x, y) in enumerate(trainDataloader):
            x, y = x.to(device), y.to(device)
            # print(x.shape, y.shape)
            outputs, yd, time1 = model(x)
            # print(outputs.shape, y.shape)
            # flags = flags_trans(flags)
            # print(flags, data[0])
            loss = loss_fn(outputs, yd, time1, y)

            # l_weight = 0.1 
            #l1_penalty = l_weight * sum([p.abs().sum() for p in model.parameters()])
            #l2_penalty = l_weight * sum([p.square().sum() for p in model.parameters()])
            #loss_with_penalty = loss + l1_penalty  # l2_penalty  #

            loss = loss.requires_grad_()  
            optimiser.zero_grad()
            loss.backward()
            # loss_with_penalty.backward() 
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=50.0)
            optimiser.step()
            lr_now = optimiser.state_dict()['param_groups'][0]['lr']
            # lr_scheduler.step(loss)

            totalLoss = totalLoss + loss.item()

            if i % 30 == 0:

                tem = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                # print(outputs[0])
                print("Time {}, Epoch [{}/{}], Step [{}/{}], loss: {:.8f}, lr: {}".format(tem, epoch+1, num_epochs, i+1, total_step, totalLoss/(i+1), lr_now))
        lr_scheduler.step()
        loss_List.append(totalLoss/(i+1))
        logger.info("Time {}, Epoch [{}/{}], Step [{}/{}], loss: {:.8f}, lr: {}".format(tem, epoch+1, num_epochs, i+1, total_step, totalLoss/(i+1), lr_now))

        model.eval()
        with torch.no_grad():
            loss_t = 0
            for j, (x, y) in enumerate(testDataloader):
                x,  y = x.to(device), y.to(device)
                outputs, yd, time1 = model(x)

                loss_test = loss_fn(outputs, yd, time1, y)
                loss_t += loss_test.item()
                # """
                if j % 30 == 0:
                    # print(outputs[0])
                    tem = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    print("Time {}, loss in test: {:.8f}".format(tem, loss_t/(j+1)))
                # """
        logger.info("Loss in Test dataset: {}".format(loss_t / (j + 1)))

        checkpoint = {
            "state_dict": model.state_dict(),
            "opt_state_dict": optimiser.state_dict(),
            "epoch": epoch
        }
        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping.checkpoint = checkpoint
        early_stopping(loss_t/(j+1), model)

        loss_test_list.append(loss_t/(j+1))
        print("_"*10)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    df = pd.DataFrame(data=np.array(loss_List), columns=["loss_train"])
    df["loss_test"] = np.array(loss_test_list)
    pathdf = path + "pile_loss.xlsx"
    df.to_excel(pathdf)

    plt.figure(figsize=(8, 8))
    plt.plot(loss_List, color='red', linewidth=1.5, linestyle='-', label="loss_train")
    plt.plot(loss_test_list, color='black', linewidth=1.5, linestyle='-', label="loss_test")
    plt.legend(loc="upper right")
    pathpic = path + "_pile_loss.jpg"
    plt.savefig(pathpic, dpi=100)
    plt.close()  


if __name__ == "__main__":
    torch.cuda.empty_cache()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    main(path="./modelResult/transform_" + str(cfg.batch) + "_")
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
