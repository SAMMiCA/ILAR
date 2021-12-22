import argparse

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18_encoder import *
from models.resnet20_cifar import *
from vissl.utils.checkpoint import replace_module_prefix
from utils import *

class MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.use_selfsup_fe = args.use_selfsup_fe
        self.new_mode = args.new_mode
        self.varsess = args.varsess
        self.not_data_init = args.not_data_init
        self.temperature = args.temperature

        self.mode = mode
        # self.num_features = 512
        model_dict = {
            'resnet18': [resnet18, 512],
            'resnet20': [resnet20, 64],
            'resnet34': [resnet34, 512],
            'resnet50': [resnet50, 2048],
            'resnet101': [resnet101, 2048],
        }

        if args.dataset in ['cifar100']:
            model_, self.num_features = model_dict['resnet20']
            # model_, self.num_features = model_dict['resnet18']
            self.encoder = model_()
        if args.dataset in ['mini_imagenet']:
            model_, self.num_features = model_dict['resnet18']
            self.encoder = model_(False, args)
        if args.dataset == 'cub200':
            model_, self.num_features = model_dict['resnet18']
            self.encoder = model_(True, args)

        if args.angle_mode is not None:
            n_sess = int((args.num_classes - args.base_class) / args.way)+1
            self.angle_w = nn.ParameterList([nn.Parameter(torch.FloatTensor(args.base_class, self.num_features))
                                         if i==0 else nn.Parameter(torch.zeros(args.way, self.num_features))
                                          for i in range(n_sess)])
            nn.init.xavier_uniform_(self.angle_w[0])

        if args.angle_mode == 'cosface':
            self.s = args.s
            self.m = args.m

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.num_features, args.num_classes, bias=False)

    def forward_cosface(self, x, label, sess, get_logit=False):
        angle_w_tot = repr_tot(sess, self.angle_w)

        if get_logit == False:
            x = self.encode(x)

        cosine = cosine_distance(x, angle_w_tot)
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        n_cls = angle_w_tot.shape[0]
        B = x.shape[0]
        one_hot = torch.arange(n_cls).expand(B,n_cls).cuda()
        label_ = label.unsqueeze(1).expand(B,n_cls)
        one_hot = torch.where(one_hot==label_,1,0)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output, cosine

    def encode(self, x):
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)

        return x

    def forward(self, input, label=None, sess=None):
        if self.mode == 'encoder':
            input = self.encode(input)
            return input
        elif self.mode == 'cosface':
            assert label is not None
            input, cos_logit = self.forward_cosface(input, label, sess)
            return input, cos_logit
        elif self.mode == 'get_logit_cos':
            input, cos_logit = self.forward_cosface(input, label, sess,True)
            return input, cos_logit

    def set_mode(self, mode):
        self.mode = mode

    def get_logits(self, x, fc):
        if 'dot' in self.new_mode:
            return F.linear(x, fc)
        elif 'cos' in self.new_mode:
            return self.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))
            # p=2, dim=-1 is as I know same as just F.linear(x),


