'''
This code is based on pytorch_ssd and RFBNet.
Details about the modules:
               TUM - Thinned U-shaped Module
               MLFPN - Multi-Level Feature Pyramid Network
               M2Det - Multi-level Multi-scale single-shot object Detector

Author:  Qijie Zhao (zhaoqijie@pku.edu.cn)
Finished Date:  01/17/2019

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import warnings
warnings.filterwarnings('ignore')
import torch.backends.cudnn as cudnn
import os,sys,time
from layers.nn_utils import *
from torch.nn import init as init
from utils.core import print_info
import pdb
#ADDED from FCOS
from backbone.resnet import resnet50
from fpn_neck import FPN

class M2Det(nn.Module):
    def __init__(self, phase, size, config = None):
        '''
        M2Det: Multi-level Multi-scale single-shot object Detector
        '''
        super(M2Det,self).__init__()
        self.phase = phase
        self.size = size
        self.init_params(config)
        print_info('===> Constructing M2Det model', ['yellow','bold'])
        self.construct_modules()

    def init_params(self, config=None): # Directly read the config
        assert config is not None, 'Error: no config'
        for key,value in config.items():
            if check_argu(key,value):
                setattr(self, key, value)

    def construct_modules(self,):
        # construct tums - not needed
        
        # construct base features - else if not needed 
        # TODO: replaced with resnet101 from FCOS
        self.base = resnet50(pretrained=True,if_include_top=False)
        self.fpn = FPN(self.planes,use_p5=True)
        # construct others
        if self.phase == 'test':
            self.softmax = nn.Softmax()
        self.Norm = nn.BatchNorm2d(256)

        # construct localization and recognition layers
        loc_ = list()
        conf_ = list()
        self.num_scales =5 #TODO: num_scales = 5 (P3, P4, P5, P6, P7)??????????????
        for i in range(self.num_scales):
            loc_.append(nn.Conv2d(self.planes, #TODO: removed levels
                                       4 * 6, # 4 is coordinates, 6 is anchors for each pixels,
                                       3, 1, 1))#?????????????
            conf_.append(nn.Conv2d(self.planes,
                                       self.num_classes * 6, #6 is anchors for each pixels,
                                       3, 1, 1))#?????????????
        self.loc = nn.ModuleList(loc_)
        self.conf = nn.ModuleList(conf_)        
    
    def forward(self,x):
        loc,conf = list(),list()
        #base_feats = list()
        #vgg not needed        
        C3,C4,C5= self.base(x)
        #base_feats = [C3, C4, C5] #TODO: replaced with resnet from FCOS
        
        sources = self.fpn([C3,C4,C5])
        #pdb.set_trace()
        # sources = [torch.cat([_fx[i-1] for _fx in tum_outs],1) for i in range(self.num_scales, 0, -1)]
        # TODO: sources should be output of fpn
        
        # forward_sfam - not needed
        # TODO: check if [0] needs batchnorm
        sources[0] = self.Norm(sources[0])
        
        for (x,l,c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            output = (
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
            )
        return output

    def init_model(self, base_model_path):
        # TODO: using pre-trained resnet so dont need this part i think
        # if self.backbone == 'vgg16':
        #     if isinstance(base_model_path, str):
        #         base_weights = torch.load(base_model_path)
        #         print_info('Loading base network...')
        #         self.base.load_state_dict(base_weights)
        # elif 'res' in self.backbone:
        #     pass # pretrained seresnet models are initially loaded when defining them.
        
        def weights_init(m):
            for key in m.state_dict():
                if key.split('.')[-1] == 'weight':
                    if 'conv' in key:
                        init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                    if 'bn' in key:
                        m.state_dict()[key][...] = 1
                elif key.split('.')[-1] == 'bias':
                    m.state_dict()[key][...] = 0
        
        print_info('Initializing weights for [tums, reduce, up_reduce, leach, loc, conf]...')
        # for i in range(self.num_levels):
            # getattr(self,'unet{}'.format(i+1)).apply(weights_init)
        # self.reduce.apply(weights_init)
        # self.up_reduce.apply(weights_init)
        # self.leach.apply(weights_init)
        self.loc.apply(weights_init)
        self.conf.apply(weights_init)

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print_info('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file))
            print_info('Finished!')
        else:
            print_info('Sorry only .pth and .pkl files supported.')

def build_net(phase='train', size=320, config = None):
    if not phase in ['test','train']:
        raise ValueError("Error: Phase not recognized")

    if not size in [320, 512, 704, 800]:
        raise NotImplementedError("Error: Sorry only M2Det320,M2Det512 M2Det704 or M2Det800 are supported!")
    
    return M2Det(phase, size, config)
