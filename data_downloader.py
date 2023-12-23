# -*- coding: UTF-8 -*- #
"""
@filename:data_downloader.py
@author:Young
@time:2023-12-23
"""
import torch
from dataset import load_init_data
import pathlib

def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    data_path = './data/'
    pathlib.Path(data_path).mkdir(parents=True, exist_ok=True)
    load_init_data('mnist', device,True, data_path)
    load_init_data('cifar10', device,True, data_path)

if __name__ == "__main__":
    main()
