# -*- coding: UTF-8 -*- #
"""
@filename:__init__.py
@author:Young
@time:2023-12-23
"""
# 负责调用具体的网络模型

import torch
from .badnet import BadNet
def load_model(model_path, model_type, input_channels, output_num, device):
    print("## load model from : %s" % model_path)
    if model_type == 'badnet':
        model = BadNet(input_channels, output_num).to(device)
    else:
        print("can't match your input model type, please check...")

    model.load_state_dict(torch.load(model_path))

    return model
