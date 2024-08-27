import torch.nn as nn
import torch

from eicn.models.registry import NETS
from ..registry import build_backbones, build_aggregator, build_heads, build_necks
from eicn.models.nets.ganbackbone import GANBackbone
from eicn.models.losses.focal_loss import FocalLoss
from eicn.models.utils.dynamic_assign import assign
from eicn.models.losses.lineiou_loss import liou_loss
import torch.nn.functional as F
from eicn.models.losses.focal_loss import FocalLoss
from eicn.models.losses.accuracy import accuracy

@NETS.register_module
class Detector(nn.Module):
    def __init__(self, cfg):
        super(Detector, self).__init__()
        self.cfg = cfg
        self.backbone = build_backbones(cfg)
        self.backbone_night = build_backbones(cfg)
        self.backbone_day = build_backbones(cfg)
        self.aggregator = build_aggregator(cfg) if cfg.haskey('aggregator') else None
        self.neck = build_necks(cfg) if cfg.haskey('neck') else None
        self.neck_img = build_necks(cfg) if cfg.haskey('neck') else None
        self.neck_night = build_necks(cfg) if cfg.haskey('neck') else None
        self.neck_day = build_necks(cfg) if cfg.haskey('neck') else None
        self.heads = build_heads(cfg)
        self.heads_img = build_heads(cfg)
        self.heads_night = build_heads(cfg)
        self.heads_day = build_heads(cfg)
        # gan
        self.GeneratorD2N = GANBackbone(d2n=True)
        self.GeneratorN2D = GANBackbone(d2n=False)
        # chanel_concate
        if  cfg.backbone.type=='ResNetWrapper' and cfg.backbone.resnet =='resnet101':
            self.conv_enhance_11 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
            self.conv_enhance_12 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
            self.conv_enhance_21 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
            self.conv_enhance_22 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
            self.conv_enhance_31 = nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1)
            self.conv_enhance_32 = nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1)
            self.conv_style_11 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
            self.conv_style_12 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
            self.conv_style_21 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
            self.conv_style_22 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
            self.conv_style_31 = nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1)
            self.conv_style_32 = nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1)
        else:
            self.conv_enhance_11 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
            self.conv_enhance_12 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
            self.conv_enhance_21 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            self.conv_enhance_22 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            self.conv_enhance_31 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
            self.conv_enhance_32 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
            self.conv_style_11 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
            self.conv_style_12 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
            self.conv_style_21 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            self.conv_style_22 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            self.conv_style_31 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
            self.conv_style_32 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        # net Generator
        for param in self.GeneratorD2N.parameters():
            param.requires_grad = False

        for param in self.GeneratorN2D.parameters():
            param.requires_grad = False
        
        self.img_h = 320
        self.img_w = 800
        self.n_strips = 72-1
        self.refine_layers =3
        weights = torch.ones(self.cfg.num_classes)
        weights[0] = self.cfg.bg_weight
        self.criterion = torch.nn.NLLLoss(ignore_index=self.cfg.ignore_label,
                                     weight=weights)

    def load_networks(self,load_path):
        print('loading the model from %s' % load_path)
                
        state_dict = torch.load(load_path)
        state_dict = state_dict['net']
        backbone_state_dict = {}
        for name, param in state_dict.items():
            if 'backbone' in name:
                new_name = name.replace('module.backbone.', '')
                backbone_state_dict[new_name] = param
        
        neck_state_dict = {}
        for name, param in state_dict.items():
            if 'neck' in name:
                new_name = name.replace('module.neck.', '')
                neck_state_dict[new_name] = param
        
        heads_state_dict = {}
        for name, param in state_dict.items():
            if 'heads' in name:
                new_name = name.replace('module.heads.', '')
                heads_state_dict[new_name] = param


        self.backbone.load_state_dict(backbone_state_dict)
        self.neck.load_state_dict(neck_state_dict)
        self.heads.load_state_dict(heads_state_dict,False)


    # loss
    def loss(self,
             output,
             batch,
             cls_loss_weight=2.,
             xyt_loss_weight=0.5,
             iou_loss_weight=2.,
             seg_loss_weight=1.):
        if self.cfg.haskey('cls_loss_weight'):
            cls_loss_weight = self.cfg.cls_loss_weight
        if self.cfg.haskey('xyt_loss_weight'):
            xyt_loss_weight = self.cfg.xyt_loss_weight
        if self.cfg.haskey('iou_loss_weight'):
            iou_loss_weight = self.cfg.iou_loss_weight
        if self.cfg.haskey('seg_loss_weight'):
            seg_loss_weight = self.cfg.seg_loss_weight

        predictions_lists = output['predictions_lists']
        targets = batch['lane_line'].clone()
        cls_criterion = FocalLoss(alpha=0.25, gamma=2.)
        cls_loss = 0
        reg_xytl_loss = 0
        iou_loss = 0
        cls_acc = []

        cls_acc_stage = []
        for stage in range(self.refine_layers):
            predictions_list = predictions_lists[stage]
            for predictions, target in zip(predictions_list, targets):
                target = target[target[:, 1] == 1]

                if len(target) == 0:
                    # If there are no targets, all predictions have to be negatives (i.e., 0 confidence)
                    cls_target = predictions.new_zeros(predictions.shape[0]).long()
                    cls_pred = predictions[:, :2]
                    cls_loss = cls_loss + cls_criterion(
                        cls_pred, cls_target).sum()
                    continue

                with torch.no_grad():
                    matched_row_inds, matched_col_inds = assign(
                        predictions, target, self.img_w, self.img_h)

                # classification targets
                cls_target = predictions.new_zeros(predictions.shape[0]).long()
                cls_target[matched_row_inds] = 1
                cls_pred = predictions[:, :2]

                # regression targets -> [start_y, start_x, theta] (all transformed to absolute values), only on matched pairs
                reg_yxtl = predictions[matched_row_inds, 2:6]
                reg_yxtl[:, 0] *= self.n_strips
                reg_yxtl[:, 1] *= (self.img_w - 1)
                reg_yxtl[:, 2] *= 180
                reg_yxtl[:, 3] *= self.n_strips

                target_yxtl = target[matched_col_inds, 2:6].clone()

                # regression targets -> S coordinates (all transformed to absolute values)
                reg_pred = predictions[matched_row_inds, 6:]
                reg_pred *= (self.img_w - 1)
                reg_targets = target[matched_col_inds, 6:].clone()

                with torch.no_grad():
                    predictions_starts = torch.clamp(
                        (predictions[matched_row_inds, 2] *
                         self.n_strips).round().long(), 0,
                        self.n_strips)  # ensure the predictions starts is valid
                    target_starts = (target[matched_col_inds, 2] *
                                     self.n_strips).round().long()
                    target_yxtl[:, -1] -= (predictions_starts - target_starts
                                           )  # reg length

                # Loss calculation
                cls_loss = cls_loss + cls_criterion(cls_pred, cls_target).sum(
                ) / target.shape[0]

                target_yxtl[:, 0] *= self.n_strips
                target_yxtl[:, 2] *= 180
                reg_xytl_loss = reg_xytl_loss + F.smooth_l1_loss(
                    reg_yxtl, target_yxtl,
                    reduction='none').mean()

                iou_loss = iou_loss + liou_loss(
                    reg_pred, reg_targets,
                    self.img_w, length=15)

                # calculate acc
                cls_accuracy = accuracy(cls_pred, cls_target)
                cls_acc_stage.append(cls_accuracy)

            cls_acc.append(sum(cls_acc_stage) / len(cls_acc_stage))

        # extra segmentation loss
        seg_loss = self.criterion(F.log_softmax(output['seg'], dim=1),
                             batch['seg'].long())

        cls_loss /= (len(targets) * self.refine_layers)
        reg_xytl_loss /= (len(targets) * self.refine_layers)
        iou_loss /= (len(targets) * self.refine_layers)

        loss = cls_loss * cls_loss_weight + reg_xytl_loss * xyt_loss_weight \
            + seg_loss * seg_loss_weight + iou_loss * iou_loss_weight

        return_value = {
            'loss': loss,
            'loss_stats': {
                'loss': loss,
                'cls_loss': cls_loss * cls_loss_weight,
                'reg_xytl_loss': reg_xytl_loss * xyt_loss_weight,
                'seg_loss': seg_loss * seg_loss_weight,
                'iou_loss': iou_loss * iou_loss_weight
            }
        }

        for i in range(self.refine_layers):
            return_value['loss_stats']['stage_{}_acc'.format(i)] = cls_acc[i]

        return return_value
    
    
    def chanel_fussion(self,fea1,fea2,fea3):
            out_fea1 = []
            out_fea2 = []
            out_fea3 = []
            
            fea21_t = self.conv_enhance_11(fea2[1])
            fea21 = self.conv_enhance_12(fea21_t)

            fea22_t = self.conv_enhance_21(fea2[2])
            fea22 = self.conv_enhance_22(fea22_t)

            fea23_t = self.conv_enhance_31(fea2[3])
            fea23 = self.conv_enhance_32(fea23_t)
            out_fea2.append(fea21)
            out_fea2.append(fea22)
            out_fea2.append(fea23)



            fea31_t = self.conv_style_11(fea3[1])
            fea31 = self.conv_style_12(fea31_t)

            fea32_t = self.conv_style_21(fea3[2])
            fea32 = self.conv_style_22(fea32_t)

            fea33_t = self.conv_style_31(fea3[3])
            fea33 = self.conv_style_32(fea33_t)
            out_fea3.append(fea31)
            out_fea3.append(fea32)
            out_fea3.append(fea33)


            fea_origin_1 = fea1[1]*fea21_t*fea31_t
            fea_origin_2 = fea1[2]*fea22_t*fea32_t
            fea_origin_3 = fea1[3]*fea23_t*fea33_t
            out_fea1.append(fea_origin_1)
            out_fea1.append(fea_origin_2)
            out_fea1.append(fea_origin_3)

            fea_fussion = out_fea1 + out_fea2 + out_fea3
            return fea_fussion
    
    def forward(self, batch):
        output = {}
        # 原始特征
        fea_img = self.backbone(batch['img'] if isinstance(batch, dict) else batch)
        # use GAN net, generate night feature
        fake_night_img = self.GeneratorD2N(batch['img'] if isinstance(batch, dict) else batch)
        fea_night = self.backbone_night(fake_night_img)
        
        # use GAN net, generate day feature.
        fake_day_img = self.GeneratorN2D(batch['img'] if isinstance(batch, dict) else batch)
        fea_day = self.backbone_day(fake_day_img)

        # 融合特征
        fea_fusion = self.chanel_fussion(fea_img,fea_day,fea_night)

        if self.neck:
            fea_fusion = self.neck(fea_fusion)
            fea_img = self.neck_img(fea_img)
            fea_night = self.neck_night(fea_night)
            fea_day = self.neck_day(fea_day)

        if self.training:
            pred1,output_fusion = self.heads(fea_fusion, batch=batch)
            pred2,output_img = self.heads_img(fea_img, batch=batch)
            pred3,output_night = self.heads_night(fea_night, batch=batch)
            pred4,output_day = self.heads_day(fea_day, batch=batch)
            pred_merged = pred1
            pred_merged['seg'] = (pred1['seg']+pred2['seg']+pred3['seg']+pred4['seg'])/4.0
            pred_merged['predictions_lists'][0] = (pred1['predictions_lists'][0] + pred2['predictions_lists'][0]+ pred3['predictions_lists'][0]+ pred4['predictions_lists'][0])/4.0
            pred_merged['predictions_lists'][1] = (pred1['predictions_lists'][1] + pred2['predictions_lists'][1]+ pred3['predictions_lists'][1]+ pred4['predictions_lists'][1])/4.0
            pred_merged['predictions_lists'][2] = (pred1['predictions_lists'][2] + pred2['predictions_lists'][2]+ pred3['predictions_lists'][2]+ pred4['predictions_lists'][2])/4.0
            output_merged = self.loss(pred_merged,batch)


            merged_dict = {}
            dict1 = output_fusion
            dict2 = output_img
            dict3 = output_night
            dict4 = output_day
            dict5 = output_merged
            for key in dict1:
                # merged_dict[key] = (dict1[key] + dict2[key] + dict3[key] + dict4[key]+ dict5[key]) / 5.0
                if isinstance(dict1[key], torch.Tensor) and isinstance(dict2[key], torch.Tensor) and isinstance(dict3[key], torch.Tensor) and isinstance(dict4[key], torch.Tensor)and isinstance(dict5[key], torch.Tensor):
                    merged_dict[key] = (dict1[key] + dict2[key] + dict3[key] + dict4[key]+ dict5[key]) / 5.0
                else:
                    merged_dict[key] = dict5[key]
            output = merged_dict
        else:
            output1 = self.heads(fea_fusion)
            output2 = self.heads_img(fea_img)
            output3 = self.heads_night(fea_night)
            output4 = self.heads_day(fea_day)
            output = (output1+output2+output3+output4)/4.0
            # output = self.heads_img(fea_img)
        return output