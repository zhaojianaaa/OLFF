# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
import random

from os.path import join as pjoin
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from sklearn.cluster import KMeans

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from pytorch_metric_learning.losses import ProxyAnchorLoss
from utils.max_patch import get_max_patch
import models.configs as configs

logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config.transformer["num_heads"]
        #self.num_attention_heads = 12
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def softmax_with_policy(self, attn, policy, eps=1e-6):
        B, N = policy.size()
        B, H, N, N = attn.size()
        attn_policy = policy.reshape(B, 1, 1, N)  # * policy.reshape(B, 1, N, 1)
        eye = torch.eye(N, dtype=attn_policy.dtype, device=attn_policy.device).view(1, 1, N, N)
        attn_policy = attn_policy + (1.0 - attn_policy) * eye
        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att
        # attn = attn.exp_() * attn_policy
        # return attn / attn.sum(dim=-1, keepdim=True)

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps/N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att), attn_policy

    def forward(self, hidden_states, policy=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # modification as TRT
        #attention_probs = self.softmax(attention_scores)
        #weights = attention_probs
        if policy == None:
            attention_probs = self.softmax(attention_scores)
            weights = attention_probs
        else:
            attention_probs, attn_policy = self.softmax_with_policy(attention_scores, policy = policy) 
            weights = [attention_probs, attn_policy]

        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        patch_size = _pair(config.patches["size"])
        if config.split == 'non-overlap':
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        elif config.split == 'overlap':
            n_patches = ((img_size[0] - patch_size[0]) // config.slide_step + 1) * ((img_size[1] - patch_size[1]) // config.slide_step + 1)
            self.patch_embeddings = Conv2d(in_channels=in_channels,
                                        out_channels=config.hidden_size,
                                        kernel_size=patch_size,
                                        stride=(config.slide_step, config.slide_step))
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)

    def forward(self, x, policy = None):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x, policy)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights


class Part_Attention(nn.Module):
    def __init__(self):
        super(Part_Attention, self).__init__()

    def forward(self, x):
        length = len(x)
        last_map = x[0]
        for i in range(1, length):
            last_map = torch.matmul(x[i], last_map)
        last_map = last_map[:,:,0,1:]

        _, max_inx = last_map.max(2)
        #_, max_inx = torch.topk(last_map,3)#3
        return _, max_inx

class CAM6FuseBlock(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_classes = num_classes
        self.cam_tr  = Block(config)
        self.cam_norm = LayerNorm(self.embed_dim, eps=1e-6)
        self.cam_head = nn.Conv2d(self.embed_dim, self.num_classes, kernel_size=3, stride=1, padding=1)
        self.cam_avgpool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self,x):
        # cam 
        bz,seq_num,feat_num = x.size()
        cam_trx,_ = self.cam_tr(x)  # x 表示倒数第二个transformer block 输出的特征
        #cam_trx = x
        cam_trx = self.cam_norm(cam_trx)
        cam_x = torch.reshape(cam_trx[:,1:], [bz, int((seq_num-1)**0.5), int((seq_num-1)**0.5), self.embed_dim])
        cam_x = cam_x.permute([0,3,1,2])
        cam_x = cam_x.contiguous()
        cam_x = self.cam_head(cam_x)
        cam_logits = self.cam_avgpool(cam_x).squeeze(3).squeeze(2)
        return cam_logits, cam_x, cam_trx

class Encoder(nn.Module):
    def __init__(self, config, num_classes):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.num_classes = num_classes
        #for _ in range(config.transformer["num_layers"] - 1):
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer))
        #self.part_select = Part_Attention()
        #self.part_layer = Block(config)
        #self.part_norm = LayerNorm(config.hidden_size, eps=1e-6)

        self.score_layer = Block(config)
        self.score_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.score_head = nn.Linear(config.hidden_size, 1)

        # cam
        self.camfuse_block = CAM6FuseBlock(config, num_classes)


    def forward(self, hidden_states):
        attn_weights = []
        for layer in self.layer[:-1]:
        #for layer in self.layer:
            hidden_states, weights = layer(hidden_states)
            attn_weights.append(weights)          
        return hidden_states, attn_weights, weights

class Transformer(nn.Module):
    def __init__(self, config, img_size, num_classes):
        super(Transformer, self).__init__()
        self.num_classes = num_classes
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, self.num_classes)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        hidden_states, attn_weights, weights = self.encoder(embedding_output)
        return hidden_states, attn_weights, weights

class RelationNet(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(feature_dim * 2, feature_dim * 2),
                                    nn.BatchNorm1d(feature_dim * 2),
                                    nn.LeakyReLU(),
                                    nn.Linear(feature_dim * 2, feature_dim))

    def forward(self, features_1, features_2):
        pair = torch.cat([features_1, features_2], dim=1)
        return self.layers(pair)

def get_bboxes(cam, cam_thr=0.2):
    """
    cam: single image with shape (h, w, 1)
    thr_val: float value (0~1)
    return estimated bounding box
    """
    bboxes = []
    graymaps = []
    for i in range(cam.shape[0]):
        cami = cam[i]
        cami = (cami * 255.).astype(np.uint8)
        map_thr = cam_thr * np.max(cami)

        _, thr_gray_heatmap = cv2.threshold(cami,
                                            int(map_thr), 255,
                                            cv2.THRESH_TOZERO)
        #thr_gray_heatmap = (thr_gray_heatmap*255.).astype(np.uint8)

        contours, _ = cv2.findContours(thr_gray_heatmap,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 0:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            estimated_bbox = [x, y, x + w, y + h]
        else:
            estimated_bbox = [0, 0, 1, 1]

        bboxes.append(estimated_bbox)
        graymaps.append(thr_gray_heatmap)

    return bboxes, graymaps #, len(contours)


def sequence_refine2(bboxes, cam_global):
    num_patches1 = [37, 37]#37/18
    stride = [12, 12]
    pp = 16#12
    part_inx = []
    for i in range(len(bboxes)):
        cam_globali = cam_global[i].reshape(-1)
        bboxj = bboxes[i]
        cam_idx = np.where(cam_globali > 0)[0]
        cam_bg_idx = np.where(cam_globali == 0)[0]
        cam_idx = cam_idx + 1
        cam_bg_idx = cam_bg_idx + 1

        img1_cam_indices_to_show = [idx for idx in cam_idx.tolist()]
        img1_bg_cam_indices_to_show = [idx for idx in cam_bg_idx.tolist()]

        patches = np.array(img1_cam_indices_to_show)
        patches_bg = np.array(img1_bg_cam_indices_to_show)

        img1_y_to_show = patches / num_patches1[1]
        img1_x_to_show = patches % num_patches1[1]
        points1 = []
        points2 = []
        idx_part = []
        idp = 0
        for y1, x1 in zip(img1_y_to_show, img1_x_to_show):
            if bboxj[0] <= x1 <= bboxj[2] and bboxj[1] <= y1 <= bboxj[3]:

                x1_show = (int(x1) - 1) * stride[1] + pp // 2
                y1_show = (int(y1) - 1) * stride[0] + pp // 2
                points1.append((y1_show, x1_show))
                points2.append((int(y1), int(x1)))
                idx_part.append(idp)
            idp += 1

        normalized = np.array(points1)
        points2 = np.array(points2)
        # clustering with feats from Transformer patches
        #normalized = torch.stack(feats,0).cpu().detach()
        num_pairs = 4#3#3/4
        num_patch = 12#12
        n_clusters = min(num_pairs, len(normalized))  # if not enough pairs, show all found pairs.
        ##### Use kmeans for clustering ####
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(normalized)
        assigmnment = kmeans.labels_

        part_inx1 = []
        #print ('****************')
        for j in range(n_clusters):
            part_inx2 = []
            #centers = kmeans.cluster_centers_[j]
            allj = np.where(assigmnment == j)[0]
            pointsj = points2[allj]
            xmin = np.min(pointsj[:,1])
            ymin = np.min(pointsj[:,0])
            xmax = np.max(pointsj[:,1])
            ymax = np.max(pointsj[:,0])

            if xmin == xmax:
                xmin -= 1
                xmax +=1
            if ymin == ymax:
                ymin -= 1
                ymax +=1
            if xmin < 0:
                xmin = 0
            if xmax >= 37:
                xmax = 36
            if ymin < 0:
                ymin = 0
            if ymax >= 37:
                ymax = 36

            part_inx1 += [xmin,ymin,xmax,ymax]
          
        # background patches here
        #patches_bg
        '''
        while len(part_inx1) < num_patch*(n_clusters+1):
            choi_p = random.choice(patches_bg)
            if choi_p not in part_inx1:
                part_inx1.append(choi_p)
        '''
        part_inx.append(part_inx1) 
            
    return np.array(part_inx)


class GCNCombiner(nn.Module):

    def __init__(self, 
                 total_num_selects: int,
                 num_classes: int, 
                 inputs: None, 
                 proj_size: None,
                 fpn_size: None):
        """
        If building backbone without FPN, set fpn_size to None and MUST give 
        'inputs' and 'proj_size', the reason of these setting is to constrain the 
        dimension of graph convolutional network input.
        """
        super(GCNCombiner, self).__init__()

        assert inputs is not None or fpn_size is not None, \
            "To build GCN combiner, you must give one features dimension."

        ### auto-proj
        self.fpn_size = fpn_size
        if fpn_size is None:
            for name in inputs:
                if len(name) == 4:
                    in_size = inputs[name].size(1)
                elif len(name) == 3:
                    in_size = inputs[name].size(2)
                else:
                    raise ValusError("The size of output dimension of previous must be 3 or 4.")
                m = nn.Sequential(
                    nn.Linear(in_size, proj_size),
                    nn.ReLU(),
                    nn.Linear(proj_size, proj_size)
                )
                self.add_module("proj_"+name, m)
            self.proj_size = proj_size
        else:
            self.proj_size = fpn_size

        ### build one layer structure (with adaptive module)
        num_joints = total_num_selects // 2#12#32

        self.param_pool0 = nn.Linear(total_num_selects, num_joints)
        
        A = torch.eye(num_joints)/100 + 1/100
        self.adj1 = nn.Parameter(copy.deepcopy(A))
        self.conv1 = nn.Conv1d(self.proj_size, self.proj_size, 1)
        self.batch_norm1 = nn.BatchNorm1d(self.proj_size)
        
        self.conv_q1 = nn.Conv1d(self.proj_size, self.proj_size//4, 1)
        self.conv_k1 = nn.Conv1d(self.proj_size, self.proj_size//4, 1)
        self.alpha1 = nn.Parameter(torch.zeros(1))

        ### merge information
        self.param_pool1 = nn.Linear(num_joints, 1)
        
        #### class predict
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(self.proj_size, num_classes)

        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        """
        '''
        hs = []
        for name in x:
            if self.fpn_size is None:
                hs.append(getattr(self, "proj_"+name)(x[name]))
            else:
                hs.append(x[name])
        '''
        hs = x.transpose(1, 2)
        #hs = torch.cat(hs, dim=1).transpose(1, 2).contiguous() # B, S', C --> B, C, S
        hs = self.param_pool0(hs)
        ### adaptive adjacency
        q1 = self.conv_q1(hs).mean(1)
        k1 = self.conv_k1(hs).mean(1)
        A1 = self.tanh(q1.unsqueeze(-1) - k1.unsqueeze(1))
        A1 = self.adj1 + A1 * self.alpha1
        ### graph convolution
        hs = self.conv1(hs)
        hs = torch.matmul(hs, A1)
        hs = self.batch_norm1(hs)
        ### predict
        hs = self.param_pool1(hs)
        hs = self.dropout(hs)
        hs = hs.flatten(1)
        hs = self.classifier(hs)

        return hs

class VisionTransformer_TRT(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, smoothing_value=0, zero_head=False):
        super(VisionTransformer_TRT, self).__init__()
        self.num_classes = num_classes
        self.smoothing_value = smoothing_value
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, self.num_classes)
        self.part_head = Linear(config.hidden_size, num_classes)
        # adding norm here as TRT does
        self.part_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.thresh = 0.6

    def forward(self, x_in, pathi=None, labels=None):
        x, attn_weights, weights = self.transformer(x_in)

        # TRT here
        bz,seq_num,feat_num = x.size()

        weight_plc = torch.zeros(bz,seq_num-1,dtype=weights.dtype, device=weights.device)
        for weights in attn_weights:
            weight_plc += weights[:,:,0,1:].mean(dim=1) #+ weights[:,:,1:,1:].mean(dim=1).mean(dim=1)
        weight_plc = weight_plc/weight_plc.sum(dim=1, keepdim=True)

        for i in range(bz):
            weight_plci = weight_plc[i].reshape(37,37).cpu().detach().numpy()
            cv2.imwrite(pathi[ik].replace('.jpg', 'mhsa.png'), cv2.resize((weight_plci*255).astype(np.uint8),(448,448)))

        # CAM_fuse here
        cam_logits, cam_x, cam_tokens = self.transformer.encoder.camfuse_block.forward(x)
                
        mes_loss = torch.nn.MSELoss(reduction='sum')
        #############loc_loss = mes_loss(part_policy, F.normalize(torch.sum(cam_x, dim=1).reshape(bz,-1), p=1,dim=1))
        loc_loss = mes_loss(weight_plc, F.normalize(torch.sum(cam_x, dim=1).reshape(bz,-1), p=1,dim=1))
        #########loc_loss = mes_loss(weight_plc*part_policy, F.normalize(torch.sum(cam_x, dim=1).reshape(bz,-1), p=1,dim=1))
        
        #cumsum
        wval, wind = weight_plc.sort(dim=1, descending=True)
        wval_cum   = wval.cumsum(dim=1)
        # 占据80%重要性的token进行re-attention，剩余20%重要性的token不进行舍弃 
        wval_policy= (wval_cum < self.thresh) .to(torch.float16).view(bz,seq_num-1)
        part_policy= torch.zeros(bz,seq_num-1,dtype=wval_policy.dtype, device=wval_policy.device)
        # 挑选出wval_policy为1的wind系数，对应构建part_policy
        part_policy = part_policy.scatter(dim=1,index=wind,src=wval_policy)

        feat_sz = int(math.sqrt(seq_num-1))
        trt_x = (weight_plc*part_policy).reshape([bz,feat_sz,feat_sz]).unsqueeze(dim=1).expand(bz, self.num_classes, feat_sz,feat_sz)
        cam = cam_x * trt_x
        cam = torch.max(cam,1)[0]
        #cam = torch.max(trt_x,1)[0]
        #cam = np.array(cam.cpu().detach().numpy())

        #wval_policy1= (wval_cum < self.thresh) .to(torch.float16).view(bz,seq_num-1)
        #part_policy1= torch.zeros(bz,seq_num-1,dtype=wval_policy.dtype, device=wval_policy.device)
        # 挑选出wval_policy为1的wind系数，对应构建part_policy
        #part_policy1 = part_policy1.scatter(dim=1,index=wind,src=wval_policy)
        part_policy1 = (cam > 0.).view(bz,seq_num-1).to(wval_policy.dtype).to(wval_policy.device)

        score_tokens, _ = self.transformer.encoder.score_layer(x[:,1:], part_policy1)   
        score_tokens= self.transformer.encoder.score_norm(score_tokens)
        scores      = self.transformer.encoder.score_head(score_tokens)
        scores      = scores.exp_() * part_policy.unsqueeze(-1)
        scores      = (scores) / (scores.sum(dim=1, keepdim=True) + 1e-6)

        select_tokens = (x[:,1:] * scores.expand(bz, seq_num-1, feat_num)).sum(1)
        #total_tokens, _  = self.blocks[-1](torch.stack([x[:,0],select_tokens],dim=1))
        total_tokens, _  = self.transformer.encoder.layer[-1](torch.stack([x[:,0],select_tokens],dim=1))
        total_encoded = self.part_norm(total_tokens)  
        trt_logits = self.part_head(total_encoded[:,0])
        

               
        #feat_sz = int(math.sqrt(seq_num-1))
        #trt_x = (weight_plc*part_policy).reshape([bz,feat_sz,feat_sz]).unsqueeze(dim=1).expand(bz, self.num_classes, feat_sz,feat_sz)
        #cam = cam_x * trt_x
        #cam = torch.max(cam,1)[0]
        #cam = torch.max(trt_x,1)[0]
        cam = np.array(cam.cpu().detach().numpy())
        bboxes1, graymaps1 = get_bboxes(cam, cam_thr=0.2)
        bboxes2, graymaps2 = get_bboxes(cam, cam_thr=0.5)
        bboxes3, graymaps3 = get_bboxes(cam, cam_thr=0.8)

        bboxes = []
        for ik in range(len(cam)):
            #yi,xi = np.where(graymaps[ik] > 0)
            #xmin,ymin,xmax,ymax = np.min(xi),np.min(yi),np.max(xi),np.max(yi)
            xmin,ymin,xmax,ymax, mask_new = get_max_patch(cam[ik])
            bboxes.append([xmin,ymin,xmax,ymax])
            img0 = x_in[ik].cpu().detach().numpy().transpose(1,2,0).astype(np.uint8)
            img0 = cv2.resize(img0, (37,37))
            img0[ymin, xmin:xmax] = 255,0,0
            img0[ymax, xmin:xmax] = 255,0,0
            img0[ymin:ymax, xmin] = 255,0,0
            img0[ymin:ymax, xmax] = 255,0,0
            img0 = cv2.resize(img0,(448,448))
            cv2.imwrite(pathi[ik].replace('.jpg', '_bbox.jpg'), img0)
            cv2.imwrite(pathi[ik].replace('.jpg', '_bbox_mask.png'), cv2.resize((cam[ik]*255).astype(np.uint8),(448,448)))
            cv2.imwrite(pathi[ik].replace('.jpg', '_bbox_mask_new.png'), cv2.resize((mask_new*255).astype(np.uint8),(448,448)))

        bboxes = []
        for ik in range(len(graymaps1)):
            #yi,xi = np.where(graymaps[ik] > 0)
            #xmin,ymin,xmax,ymax = np.min(xi),np.min(yi),np.max(xi),np.max(yi)
            xmin,ymin,xmax,ymax, mask_new = get_max_patch(graymaps1[ik])
            bboxes.append([xmin,ymin,xmax,ymax])
            img0 = x_in[ik].cpu().detach().numpy().transpose(1,2,0).astype(np.uint8)
            img0 = cv2.resize(img0, (37,37))
            img0[ymin, xmin:xmax] = 255,0,0
            img0[ymax, xmin:xmax] = 255,0,0
            img0[ymin:ymax, xmin] = 255,0,0
            img0[ymin:ymax, xmax] = 255,0,0
            img0 = cv2.resize(img0,(448,448))
            cv2.imwrite(pathi[ik].replace('.jpg', '_bbox1.jpg'), img0)
            cv2.imwrite(pathi[ik].replace('.jpg', '_bbox1_mask.png'), cv2.resize((graymaps1[ik]*255).astype(np.uint8),(448,448)))
            cv2.imwrite(pathi[ik].replace('.jpg', '_bbox1_mask_new.png'), cv2.resize((mask_new*255).astype(np.uint8),(448,448)))

        bboxes = []
        for ik in range(len(graymaps2)):
            #yi,xi = np.where(graymaps[ik] > 0)
            #xmin,ymin,xmax,ymax = np.min(xi),np.min(yi),np.max(xi),np.max(yi)
            xmin,ymin,xmax,ymax, mask_new = get_max_patch(graymaps2[ik])
            bboxes.append([xmin,ymin,xmax,ymax])
            img0 = x_in[ik].cpu().detach().numpy().transpose(1,2,0).astype(np.uint8)
            img0 = cv2.resize(img0, (37,37))
            img0[ymin, xmin:xmax] = 255,0,0
            img0[ymax, xmin:xmax] = 255,0,0
            img0[ymin:ymax, xmin] = 255,0,0
            img0[ymin:ymax, xmax] = 255,0,0
            img0 = cv2.resize(img0,(448,448))
            cv2.imwrite(pathi[ik].replace('.jpg', '_bbox2.jpg'), img0)
            cv2.imwrite(pathi[ik].replace('.jpg', '_bbox2_mask.png'), cv2.resize((graymaps2[ik]*255).astype(np.uint8),(448,448)))
            cv2.imwrite(pathi[ik].replace('.jpg', '_bbox2_mask_new.png'), cv2.resize((mask_new*255).astype(np.uint8), (448,448)))

        bboxes = []
        for ik in range(len(graymaps3)):
            #yi,xi = np.where(graymaps[ik] > 0)
            #xmin,ymin,xmax,ymax = np.min(xi),np.min(yi),np.max(xi),np.max(yi)
            xmin,ymin,xmax,ymax, mask_new = get_max_patch(graymaps3[ik])
            bboxes.append([xmin,ymin,xmax,ymax])
            img0 = x_in[ik].cpu().detach().numpy().transpose(1,2,0).astype(np.uint8)
            img0 = cv2.resize(img0, (37,37))
            img0[ymin, xmin:xmax] = 255,0,0
            img0[ymax, xmin:xmax] = 255,0,0
            img0[ymin:ymax, xmin] = 255,0,0
            img0[ymin:ymax, xmax] = 255,0,0
            img0 = cv2.resize(img0,(448,448))
            cv2.imwrite(pathi[ik].replace('.jpg', '_bbox3.jpg'), img0)
            cv2.imwrite(pathi[ik].replace('.jpg', '_bbox3_mask.png'), cv2.resize((graymaps3[ik]*255).astype(np.uint8),(448,448)))
            cv2.imwrite(pathi[ik].replace('.jpg', '_bbox3_mask_new.png'), cv2.resize((mask_new*255).astype(np.uint8),(448,448)))
        
            

        '''
        #
        #############################
        part_num, part_inx = self.transformer.encoder.part_select(attn_weights)
        part_inx = part_inx + 1
        B, num = part_inx.shape
        #B, num, _ = part_inx.shape
        #part_inx = part_inx.reshape(B,-1)

        part_inx_gcn = sequence_refine2(bboxes, graymaps)

        parts = []
        for i in range(B):
            parts.append(x[i, part_inx[i,:]])

        x_parts = x[:,1:,:]#bs*1369*768
        bs,Np,Dim = x_parts.shape#bs*768*37*37
        x_parts = x_parts.permute(0,2,1).reshape(bs,Dim,np.int(np.sqrt(Np)),np.int(np.sqrt(Np)))
        parts_gcn = []
        x_partsjs = []
        parts_j = []
        for i in range(B):
            #parts_gcn.append(x[i, part_inx_gcn[i,:]])
            for j in range(4):
                if j == 3:
                    xminj,yminj,xmaxj,ymaxj = part_inx_gcn[i,j*4:]
                else:
                    xminj,yminj,xmaxj,ymaxj = part_inx_gcn[i,j*4:(j+1)*4]
                x_partsj = x_parts[i,:,yminj:ymaxj,xminj:xmaxj]
                x_partsj = self.up1(x_partsj.unsqueeze(0)).squeeze(0)
                x_partsj = x_partsj.reshape(Dim,self.N_p*self.N_p).permute(1,0)
                x_partsjs.append(x_partsj)
            #parts_j.append(torch.cat([x_partsjs[0],x_partsjs[1],x_partsjs[2]]))
            parts_j.append(torch.cat([x_partsjs[0],x_partsjs[1],x_partsjs[2],x_partsjs[3]]))

        parts = torch.stack(parts).squeeze(1)
        concat = torch.cat((x[:,0].unsqueeze(1), parts), dim=1)
        parts_gcn = torch.stack(parts_j).squeeze(1)
        concat_gcn = torch.cat((x[:,0].unsqueeze(1), parts_gcn), dim=1)
        #concat += self.position_embeddings1

        #B = parts.shape[0]
        #cls_tokens = self.cls_token1.expand(B, -1, -1)
        #concat = torch.cat((cls_tokens, parts), dim=1)
        part_states, part_weights = self.transformer.encoder.part_layer(concat)
        # wo
        part_tokens = self.transformer.encoder.part_norm(part_states)  
        # w
        #part_states, _  = self.transformer.encoder.layer[-1](part_states)
        #part_tokens = self.part_norm(part_states)  

        c_gcn1 = self.conv1(self.conv01(concat_gcn[:,1:self.N_p*self.N_p+1,:]))
        c_gcn2 = self.conv2(self.conv02(concat_gcn[:,self.N_p*self.N_p+1:2*self.N_p*self.N_p+1,:]))
        c_gcn3 = self.conv3(self.conv03(concat_gcn[:,2*self.N_p*self.N_p+1:3*self.N_p*self.N_p+1,:]))
        c_gcn4 = self.conv4(self.conv04(concat_gcn[:,3*self.N_p*self.N_p+1:,:]))
        #c_gcn4 = self.conv4(concat_gcn[:,37:,:])
        #concat_gcn = torch.cat([concat_gcn[:,0,:].unsqueeze(1),c_gcn1,c_gcn2,c_gcn3],1)
        concat_gcn = torch.cat([concat_gcn[:,0,:].unsqueeze(1),c_gcn1,c_gcn2,c_gcn3,c_gcn4],1)

        gcn_logits = self.combiner(concat_gcn)
        ######################################
        # TFG here
        part_logits = self.part_head(part_tokens[:, 0])
        if labels is not None:
            if self.smoothing_value == 0:
                loss_fct = CrossEntropyLoss()
            else:
                loss_fct = LabelSmoothing(self.smoothing_value)
            part_loss = loss_fct(part_logits.view(-1, self.num_classes), labels.view(-1))
            contrast_loss = con_loss(part_tokens[:, 0], labels.view(-1))
            #contrast_loss = con_loss_new(part_tokens[:, 0], labels.view(-1))

            # RP here
            relation_repr_trt = self.relation_net(total_encoded[:,0], part_tokens[:,0])
            relation_repr_cam = self.relation_net(cam_tokens[:,0], part_tokens[:,0])
            loss_rp_trt = self.proxy_criterion(relation_repr_trt, labels)
            loss_rp_cam = self.proxy_criterion(relation_repr_cam, labels)

            #loss = part_loss + contrast_loss + loss_rp
            return part_loss, contrast_loss, loss_rp_trt, loss_rp_cam, part_logits, logits, cam_logits, gcn_logits
        else:

            return part_logits, logits, cam_logits, gcn_logits, cam_x * trt_x
        '''
        #return trt_logits, cam_logits, loc_loss
        return trt_logits, cam_logits, torch.tensor(0.0)

def con_loss_new(features, labels):
    eps = 1e-6
    B, _ = features.shape
    features = F.normalize(features)
    cos_matrix = features.mm(features.t())

    pos_label_matrix = torch.stack([labels == labels[i] for i in range(B)]).float()
    neg_label_matrix = 1 - pos_label_matrix
    neg_label_matrix_new = 1 - pos_label_matrix

    pos_cos_matrix = 1 - cos_matrix
    neg_cos_matrix =1 + cos_matrix
  
    margin = 0.3
    sim = (1 + cos_matrix)/2.0
    scores = 1 - sim
    positive_scores = torch.where(pos_label_matrix == 1.0, scores, scores-scores)
    mask = torch.eye(features.size(0)).cuda()
    positive_scores = torch.where(mask == 1.0, positive_scores - positive_scores, positive_scores)

    #print(positive_scores)
    #print(torch.sum(positive_scores, dim=1, keepdim=True))
    positive_scores = torch.sum(positive_scores, dim=1, keepdim=True)/((torch.sum(pos_label_matrix, dim=1, keepdim=True)-1)+eps)
    positive_scores = torch.repeat_interleave(positive_scores, B, dim=1)
    
    #print(positive_scores)
    relative_dis1 = margin + positive_scores -scores
    neg_label_matrix_new[relative_dis1 < 0] = 0
    neg_label_matrix = neg_label_matrix*neg_label_matrix_new
    
    #print(neg_label_matrix)
    loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
    loss /= B*B
    #print(loss)
    return loss

def con_loss(features, labels):
    B, _ = features.shape
    features = F.normalize(features)
    cos_matrix = features.mm(features.t())
    pos_label_matrix = torch.stack([labels == labels[i] for i in range(B)]).float()
    neg_label_matrix = 1 - pos_label_matrix
    pos_cos_matrix = 1 - cos_matrix
    neg_cos_matrix = cos_matrix - 0.4
    neg_cos_matrix[neg_cos_matrix < 0] = 0
    loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
    loss /= (B * B)
    return loss

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'testing': configs.get_testing(),
}
