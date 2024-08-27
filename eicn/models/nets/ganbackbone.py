import torch.nn as nn
from .gan_networks import ResnetGenerator
from .gan_networks import get_norm_layer
import torch
class GANBackbone(nn.Module):
    def __init__(self,d2n):
        super(GANBackbone, self).__init__()
        input_nc=3
        output_nc=3
        norm_layer = get_norm_layer(norm_type='instance')
        self.netG = ResnetGenerator(input_nc, output_nc, 
                                         ngf=64, norm_layer=norm_layer, 
                                         use_dropout=False, n_blocks=9, 
                                         padding_type='reflect')
        
               
        print('load generator path.')
        if d2n:
            load_path = './pth/day2night_net_G.pth'
        else:
            load_path = './pth/night2day_net_G.pth'
        self.load_networks(load_path)
    def forward(self, x):
        fake_img = self.netG(x)
        return fake_img


    def load_networks(self,load_path):
        print('loading the model from %s' % load_path)
                
        state_dict = torch.load(load_path)
        # 删除metadata
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
            self.__patch_instance_norm_state_dict(state_dict, self.netG, key.split('.'))
        self.netG.load_state_dict(state_dict)


    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)



class ExtractFea(nn.Module):
    def __init__(self):
        super(ExtractFea, self).__init__()
        self.conv1 = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        
    def forward(self, input):
        x = self.conv1(input)
        output1 = x.clone() 
        
        x = self.conv2(x)
        output2 = x.clone()
        
        x = self.conv3(x)
        output3 = x.clone()
        
        x = self.conv4(x)
        output4 = x.clone()
        
        return [output1, output2, output3, output4]



