# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np
import time

from datetime import timedelta

import torch
import torch.distributed as dist

from tqdm import tqdm
#from torch.utils.tensorboard import SummaryWriter
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

from models.modeling_fuse_cub_step2_v10 import VisionTransformer_TRT, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
from utils.dist_util import get_world_size
from utils.checkpoints_w11 import check_load_fuse, resultss

logger = logging.getLogger(__name__)

os.environ['CUDA_VISIBLE_DEVICES']='0'

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def save_model(args, model, global_step, acs):
    model_to_save = model.module if hasattr(model, 'module') else model
    #model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    model_checkpoint = os.path.join(args.output_dir, "checkpoint_cub_step2_v10_" + str(global_step) + '_' + str(acs[0].item())[0:7] + '_' + str(acs[1].item())[0:7] + '_' + str(acs[2].item())[0:7] + '_' + str(acs[3].item())[0:7] + '_' + str(acs[4].item())[0:7] + ".pth")
    if args.fp16:
        checkpoint = {
            'model': model_to_save.state_dict(),
            'amp': amp.state_dict()
        }
    else:
        checkpoint = {
            'model': model_to_save.state_dict(),
        }
    torch.save(checkpoint, model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)

def save_model0(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    #model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.bin" % args.name)
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint0.pth" % args.name)
    #if args.fp16:
    if False:
        checkpoint = {
            'model': model_to_save.state_dict(),
            'amp': amp.state_dict()
        }
    else:
        checkpoint = {
            'model': model_to_save.state_dict(),
        }
    torch.save(checkpoint, model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)

def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]
    config.split = args.split
    config.slide_step = args.slide_step

    if args.dataset == "CUB_200_2011":
        num_classes = 200
    elif args.dataset == "car":
        num_classes = 196
    elif args.dataset == "nabirds":
        num_classes = 555
    elif args.dataset == "dog":
        num_classes = 120
    elif args.dataset == "INat2017":
        num_classes = 5089

    args.num_labels = num_classes
    model = VisionTransformer_TRT(config, args.img_size, zero_head=True, num_classes=num_classes,                                                   smoothing_value=args.smoothing_value)

    #model.load_from(np.load(args.pretrained_dir))
    #save_model0(args, model)
    if args.pretrained_model is not None:
        pretrained_model = torch.load(args.pretrained_model)['model']
        #model.load_state_dict(pretrained_model)
        model.load_state_dict(pretrained_model, strict = False)
        #model = check_load_fuse(model, args.pretrained_model)

    for name, p in model.named_parameters():
        if 'step2_' not in name:
            p.requires_grad = False
            #pdb.set_trace()
    for name, p in model.named_parameters():
        print(name,' grad-', p.requires_grad)

    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def valid(args, model, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("\n")
    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    #c51
    GCN_preds, TFG_preds, TRT_preds, CAM_preds = [],[],[],[]
    # c52
    GCN_TFG_preds, GCN_TRT_preds, GCN_CAM_preds, TFG_TRT_preds, TFG_CAM_preds, TRT_CAM_preds = [],[],[],[],[],[]
    # c52_nor
    nor_GCN_TFG_preds, nor_GCN_TRT_preds, nor_GCN_CAM_preds, nor_TFG_TRT_preds, nor_TFG_CAM_preds, nor_TRT_CAM_preds = [],[],[],[],[],[]
    # c53
    GCN_TFG_TRT_preds, GCN_TFG_CAM_preds, GCN_TRT_CAM_preds, TFG_TRT_CAM_preds = [],[],[],[]
    # c53_nor
    nor_GCN_TFG_TRT_preds, nor_GCN_TFG_CAM_preds, nor_GCN_TRT_CAM_preds, nor_TFG_TRT_CAM_preds = [],[],[],[]
    # c54
    GCN_TFG_TRT_CAM_preds = []
    # c54_nor
    nor_GCN_TFG_TRT_CAM_preds = []
    # c55, c55_nor, all_label
    all_preds, nor_all_preds, all_label = [],[],[]
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        #batch = tuple(t.to(args.device) for t in batch)
        x, y, pathi = batch[0].to(args.device), batch[1].to(args.device), batch[2]
        #try:
        if True:
            with torch.no_grad():
                loc_loss, logits, TRT_logits, CAM_logits, GCN_logits = model(x, pathi)

                eval_loss = loss_fct(logits, y) + torch.nn.CrossEntropyLoss()(TRT_logits, y) + torch.nn.CrossEntropyLoss()(CAM_logits, y) + torch.nn.CrossEntropyLoss()(GCN_logits, y) + loc_loss
                eval_loss = eval_loss.mean()
                eval_losses.update(eval_loss.item())

                # leave for GCN_logits
                nor_GCN_logits = (GCN_logits - torch.min(GCN_logits,1)[0].unsqueeze(1).repeat(1,args.num_labels))/(torch.max(GCN_logits,1)[0].unsqueeze(1).repeat(1,args.num_labels) - torch.min(GCN_logits,1)[0].unsqueeze(1).repeat(1,args.num_labels))
                nor_GCN_logits = nor_GCN_logits/torch.sum(nor_GCN_logits,1).unsqueeze(1).repeat(1,args.num_labels)

                nor_logits = (logits - torch.min(logits,1)[0].unsqueeze(1).repeat(1, args.num_labels))/(torch.max(logits,1)[0].unsqueeze(1).repeat(1,args.num_labels) - torch.min(logits,1)[0].unsqueeze(1).repeat(1,args.num_labels))
                nor_logits = nor_logits/torch.sum(nor_logits,1).unsqueeze(1).repeat(1,args.num_labels)

                nor_TRT_logits = (TRT_logits - torch.min(TRT_logits,1)[0].unsqueeze(1).repeat(1,args.num_labels))/(torch.max(TRT_logits,1)[0].unsqueeze(1).repeat(1,args.num_labels) - torch.min(TRT_logits,1)[0].unsqueeze(1).repeat(1,args.num_labels))
                nor_TRT_logits = nor_TRT_logits/torch.sum(nor_TRT_logits,1).unsqueeze(1).repeat(1,args.num_labels)

                nor_CAM_logits = (CAM_logits - torch.min(CAM_logits,1)[0].unsqueeze(1).repeat(1,args.num_labels))/(torch.max(CAM_logits,1)[0].unsqueeze(1).repeat(1,args.num_labels) - torch.min(CAM_logits,1)[0].unsqueeze(1).repeat(1,args.num_labels))
                nor_CAM_logits = nor_CAM_logits/torch.sum(nor_CAM_logits,1).unsqueeze(1).repeat(1,args.num_labels)

                # c51
                preds_GCN = torch.argmax(GCN_logits, dim=-1)
                preds_TFG = torch.argmax(logits, dim=-1)
                preds_TRT = torch.argmax(TRT_logits, dim=-1)
                preds_CAM = torch.argmax(CAM_logits, dim=-1)

                # c52
                preds_GCN_TFG = torch.argmax(GCN_logits + logits, dim=-1)
                preds_GCN_TRT = torch.argmax(GCN_logits + TRT_logits, dim=-1)
                preds_GCN_CAM = torch.argmax(GCN_logits + CAM_logits, dim=-1)
                preds_TFG_TRT = torch.argmax(logits + TRT_logits, dim=-1)
                preds_TFG_CAM = torch.argmax(logits + CAM_logits, dim=-1)
                preds_TRT_CAM = torch.argmax(TRT_logits + CAM_logits, dim=-1)
                # c52_nor
                nor_preds_GCN_TFG = torch.argmax(nor_GCN_logits + nor_logits, dim=-1)
                nor_preds_GCN_TRT = torch.argmax(nor_GCN_logits + nor_TRT_logits, dim=-1)
                nor_preds_GCN_CAM = torch.argmax(nor_GCN_logits + nor_CAM_logits, dim=-1)
                nor_preds_TFG_TRT = torch.argmax(nor_logits + nor_TRT_logits, dim=-1)
                nor_preds_TFG_CAM = torch.argmax(nor_logits + nor_CAM_logits, dim=-1)
                nor_preds_TRT_CAM = torch.argmax(nor_TRT_logits + nor_CAM_logits, dim=-1)

                # c53
                preds_GCN_TFG_TRT = torch.argmax(GCN_logits + logits + TRT_logits, dim=-1)
                preds_GCN_TFG_CAM = torch.argmax(GCN_logits + logits + CAM_logits, dim=-1)
                preds_GCN_TRT_CAM = torch.argmax(GCN_logits + TRT_logits + CAM_logits, dim=-1)
                preds_TFG_TRT_CAM = torch.argmax(logits + TRT_logits + CAM_logits, dim=-1)

                # c53_nor
                nor_preds_GCN_TFG_TRT = torch.argmax(nor_GCN_logits + nor_logits + nor_TRT_logits, dim=-1)
                nor_preds_GCN_TFG_CAM = torch.argmax(nor_GCN_logits + nor_logits + nor_CAM_logits, dim=-1)
                nor_preds_GCN_TRT_CAM = torch.argmax(nor_GCN_logits + nor_TRT_logits + nor_CAM_logits, dim=-1)
                nor_preds_TFG_TRT_CAM = torch.argmax(nor_logits + nor_TRT_logits + nor_CAM_logits, dim=-1)

                # c54
                preds_GCN_TFG_TRT_CAM = torch.argmax(logits + TRT_logits + GCN_logits + CAM_logits, dim=-1)

                # c54_nor
                nor_preds_GCN_TFG_TRT_CAM = torch.argmax(nor_logits + nor_TRT_logits + nor_GCN_logits + nor_CAM_logits, dim=-1)


            if len(all_label) == 0:
                # c51
                GCN_preds.append(preds_GCN.detach().cpu().numpy())
                TFG_preds.append(preds_TFG.detach().cpu().numpy())
                TRT_preds.append(preds_TRT.detach().cpu().numpy())
                CAM_preds.append(preds_CAM.detach().cpu().numpy())

                # c52
                GCN_TFG_preds.append(preds_GCN_TFG.detach().cpu().numpy())
                GCN_TRT_preds.append(preds_GCN_TRT.detach().cpu().numpy())
                GCN_CAM_preds.append(preds_GCN_CAM.detach().cpu().numpy())
                TFG_TRT_preds.append(preds_TFG_TRT.detach().cpu().numpy())
                TFG_CAM_preds.append(preds_TFG_CAM.detach().cpu().numpy())
                TRT_CAM_preds.append(preds_TRT_CAM.detach().cpu().numpy())

                # c52_nor
                nor_GCN_TFG_preds.append(nor_preds_GCN_TFG.detach().cpu().numpy())
                nor_GCN_TRT_preds.append(nor_preds_GCN_TRT.detach().cpu().numpy())
                nor_GCN_CAM_preds.append(nor_preds_GCN_CAM.detach().cpu().numpy())
                nor_TFG_TRT_preds.append(nor_preds_TFG_TRT.detach().cpu().numpy())
                nor_TFG_CAM_preds.append(nor_preds_TFG_CAM.detach().cpu().numpy())
                nor_TRT_CAM_preds.append(nor_preds_TRT_CAM.detach().cpu().numpy())

                # c53
                GCN_TFG_TRT_preds.append(preds_GCN_TFG_TRT.detach().cpu().numpy())
                GCN_TFG_CAM_preds.append(preds_GCN_TFG_CAM.detach().cpu().numpy())
                GCN_TRT_CAM_preds.append(preds_GCN_TRT_CAM.detach().cpu().numpy())
                TFG_TRT_CAM_preds.append(preds_TFG_TRT_CAM.detach().cpu().numpy())

                # c53_nor
                nor_GCN_TFG_TRT_preds.append(nor_preds_GCN_TFG_TRT.detach().cpu().numpy())
                nor_GCN_TFG_CAM_preds.append(nor_preds_GCN_TFG_CAM.detach().cpu().numpy())
                nor_GCN_TRT_CAM_preds.append(nor_preds_GCN_TRT_CAM.detach().cpu().numpy())
                nor_TFG_TRT_CAM_preds.append(nor_preds_TFG_TRT_CAM.detach().cpu().numpy())

                # c54
                GCN_TFG_TRT_CAM_preds.append(preds_GCN_TFG_TRT_CAM.detach().cpu().numpy())
                # c54_nor
                nor_GCN_TFG_TRT_CAM_preds.append(nor_preds_GCN_TFG_TRT_CAM.detach().cpu().numpy())

                # c55 and c55_no4 
                all_label.append(y.detach().cpu().numpy())
            else:
                # c51
                GCN_preds[0] = np.append(GCN_preds[0], preds_GCN.detach().cpu().numpy(), axis=0)
                TFG_preds[0] = np.append(TFG_preds[0], preds_TFG.detach().cpu().numpy(), axis=0)
                TRT_preds[0] = np.append(TRT_preds[0], preds_TRT.detach().cpu().numpy(), axis=0)
                CAM_preds[0] = np.append(CAM_preds[0], preds_CAM.detach().cpu().numpy(), axis=0)

                # c52
                GCN_TFG_preds[0] = np.append(GCN_TFG_preds[0], preds_GCN_TFG.detach().cpu().numpy(), axis=0)
                GCN_TRT_preds[0] = np.append(GCN_TRT_preds[0], preds_GCN_TRT.detach().cpu().numpy(), axis=0)
                GCN_CAM_preds[0] = np.append(GCN_CAM_preds[0], preds_GCN_CAM.detach().cpu().numpy(), axis=0)
                TFG_TRT_preds[0] = np.append(TFG_TRT_preds[0], preds_TFG_TRT.detach().cpu().numpy(), axis=0)
                TFG_CAM_preds[0] = np.append(TFG_CAM_preds[0], preds_TFG_CAM.detach().cpu().numpy(), axis=0)
                TRT_CAM_preds[0] = np.append(TRT_CAM_preds[0], preds_TRT_CAM.detach().cpu().numpy(), axis=0)

                # c52_nor
                nor_GCN_TFG_preds[0] = np.append(nor_GCN_TFG_preds[0], nor_preds_GCN_TFG.detach().cpu().numpy(), axis=0)
                nor_GCN_TRT_preds[0] = np.append(nor_GCN_TRT_preds[0], nor_preds_GCN_TRT.detach().cpu().numpy(), axis=0)
                nor_GCN_CAM_preds[0] = np.append(nor_GCN_CAM_preds[0], nor_preds_GCN_CAM.detach().cpu().numpy(), axis=0)
                nor_TFG_TRT_preds[0] = np.append( nor_TFG_TRT_preds[0], nor_preds_TFG_TRT.detach().cpu().numpy(), axis=0)
                nor_TFG_CAM_preds[0] = np.append(nor_TFG_CAM_preds[0], nor_preds_TFG_CAM.detach().cpu().numpy(), axis=0)
                nor_TRT_CAM_preds[0] = np.append(nor_TRT_CAM_preds[0], nor_preds_TRT_CAM.detach().cpu().numpy(), axis=0)

                # c53
                GCN_TFG_TRT_preds[0] = np.append(GCN_TFG_TRT_preds[0], preds_GCN_TFG_TRT.detach().cpu().numpy(), axis=0)
                GCN_TFG_CAM_preds[0] = np.append(GCN_TFG_CAM_preds[0], preds_GCN_TFG_CAM.detach().cpu().numpy(), axis=0)
                GCN_TRT_CAM_preds[0] = np.append(GCN_TRT_CAM_preds[0], preds_GCN_TRT_CAM.detach().cpu().numpy(), axis=0)
                TFG_TRT_CAM_preds[0] = np.append(TFG_TRT_CAM_preds[0], preds_TFG_TRT_CAM.detach().cpu().numpy(), axis=0)

                # c53_nor
                nor_GCN_TFG_TRT_preds[0] = np.append(nor_GCN_TFG_TRT_preds[0], nor_preds_GCN_TFG_TRT.detach().cpu().numpy(), axis=0)
                nor_GCN_TFG_CAM_preds[0] = np.append(nor_GCN_TFG_CAM_preds[0], nor_preds_GCN_TFG_CAM.detach().cpu().numpy(), axis=0)
                nor_GCN_TRT_CAM_preds[0] = np.append(nor_GCN_TRT_CAM_preds[0], nor_preds_GCN_TRT_CAM.detach().cpu().numpy(), axis=0)
                nor_TFG_TRT_CAM_preds[0] = np.append(nor_TFG_TRT_CAM_preds[0], nor_preds_TFG_TRT_CAM.detach().cpu().numpy(), axis=0)

                # c54
                GCN_TFG_TRT_CAM_preds[0] = np.append(GCN_TFG_TRT_CAM_preds[0], preds_GCN_TFG_TRT_CAM.detach().cpu().numpy(), axis=0)

                # c54_nor
                nor_GCN_TFG_TRT_CAM_preds[0] = np.append(nor_GCN_TFG_TRT_CAM_preds[0], nor_preds_GCN_TFG_TRT_CAM.detach().cpu().numpy(), axis=0)

                # c55 and c55_no4 
                all_label[0] = np.append(all_label[0], y.detach().cpu().numpy(), axis=0)
            epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)
        #except:
        #    print ("Valid Error")

    accuracy_GCN, accuracy_TFG, accuracy_TRT, accuracy_CAM, accuracy_GCN_TFG, accuracy_GCN_TRT, accuracy_GCN_CAM, accuracy_TFG_TRT, accuracy_TFG_CAM, accuracy_TRT_CAM, nor_accuracy_GCN_TFG, nor_accuracy_GCN_TRT, nor_accuracy_GCN_CAM, nor_accuracy_TFG_TRT, nor_accuracy_TFG_CAM, nor_accuracy_TRT_CAM, accuracy_GCN_TFG_TRT, accuracy_GCN_TFG_CAM, accuracy_GCN_TRT_CAM, accuracy_TFG_TRT_CAM, nor_accuracy_GCN_TFG_TRT, nor_accuracy_GCN_TFG_CAM, nor_accuracy_GCN_TRT_CAM, nor_accuracy_TFG_TRT_CAM, accuracy_GCN_TFG_TRT_CAM, nor_accuracy_GCN_TFG_TRT_CAM = resultss(args, all_label, GCN_preds, TFG_preds, TRT_preds, CAM_preds, GCN_TFG_preds, GCN_TRT_preds, GCN_CAM_preds, TFG_TRT_preds, TFG_CAM_preds, TRT_CAM_preds, nor_GCN_TFG_preds, nor_GCN_TRT_preds, nor_GCN_CAM_preds, nor_TFG_TRT_preds, nor_TFG_CAM_preds, nor_TRT_CAM_preds, GCN_TFG_TRT_preds, GCN_TFG_CAM_preds, GCN_TRT_CAM_preds, TFG_TRT_CAM_preds, nor_GCN_TFG_TRT_preds, nor_GCN_TFG_CAM_preds, nor_GCN_TRT_CAM_preds, nor_TFG_TRT_CAM_preds, GCN_TFG_TRT_CAM_preds, nor_GCN_TFG_TRT_CAM_preds)
  
    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)

    logger.info("c21 TFG: %2.5f, TRT: %2.5f, CAM: %2.5f, GCN: %2.5f" % (accuracy_TFG, accuracy_TRT, accuracy_CAM, accuracy_GCN))
    logger.info("c22 %2.5f, %2.5f, %2.5f, %2.5f, %2.5f, %2.5f" % (accuracy_GCN_TFG, accuracy_GCN_TRT, accuracy_GCN_CAM, accuracy_TFG_TRT, accuracy_TFG_CAM, accuracy_TRT_CAM))
    logger.info("c42_nor %2.5f, %2.5f, %2.5f, %2.5f, %2.5f, %2.5f" % (nor_accuracy_GCN_TFG, nor_accuracy_GCN_TRT, nor_accuracy_GCN_CAM, nor_accuracy_TFG_TRT, nor_accuracy_TFG_CAM, nor_accuracy_TRT_CAM))
    logger.info("c43 %2.5f, %2.5f, %2.5f, %2.5f" % (accuracy_GCN_TFG_TRT, accuracy_GCN_TFG_CAM, accuracy_GCN_TRT_CAM, accuracy_TFG_TRT_CAM))
    logger.info("c43_nor %2.5f, %2.5f, %2.5f, %2.5f" % (nor_accuracy_GCN_TFG_TRT, nor_accuracy_GCN_TFG_CAM, nor_accuracy_GCN_TRT_CAM, nor_accuracy_TFG_TRT_CAM))
    logger.info("c44: %2.5f and c44_nor: %2.5f" % (accuracy_GCN_TFG_TRT_CAM, nor_accuracy_GCN_TFG_TRT_CAM))
    return [accuracy_GCN, accuracy_TFG, accuracy_TRT, accuracy_CAM, accuracy_GCN_TFG, accuracy_GCN_TRT, accuracy_GCN_CAM, accuracy_TFG_TRT, accuracy_TFG_CAM, accuracy_TRT_CAM, nor_accuracy_GCN_TFG, nor_accuracy_GCN_TRT, nor_accuracy_GCN_CAM, nor_accuracy_TFG_TRT, nor_accuracy_TFG_CAM, nor_accuracy_TRT_CAM, accuracy_GCN_TFG_TRT, accuracy_GCN_TFG_CAM, accuracy_GCN_TRT_CAM, accuracy_TFG_TRT_CAM, nor_accuracy_GCN_TFG_TRT, nor_accuracy_GCN_TFG_CAM, nor_accuracy_GCN_TRT_CAM, nor_accuracy_TFG_TRT_CAM, nor_accuracy_GCN_TFG_TRT_CAM, accuracy_GCN_TFG_TRT_CAM]

def train(args, model):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
    #writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    train_loader, test_loader = get_loader(args)

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    print (args.fp16)
    print (args.local_rank)
    if args.fp16:
        model, optimizer = amp.initialize(models=model,
                                          optimizers=optimizer,
                                          opt_level=args.fp16_opt_level)
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20

    # Distributed training
    if args.local_rank != -1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses_part = AverageMeter()
    losses_cons = AverageMeter()
    losses_rp_trt = AverageMeter()
    losses_rp_cam = AverageMeter()
    losses_trt = AverageMeter()
    losses_cam = AverageMeter()
    losses_gcn = AverageMeter()
    losses_loc = AverageMeter()
    losses = AverageMeter()
    global_step, best_acc, best_acc_all = 0, 0, 0
    start_time = time.time()
    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        all_preds_TFG, all_preds_TRT, all_preds_CAM, all_preds_GCN, all_preds, all_label = [], [], [], [], [], []
        for step, batch in enumerate(epoch_iterator):
            #batch = tuple(t.to(args.device) for t in batch)
            x, y, pathi = batch[0].to(args.device), batch[1].to(args.device), batch[2]
            
            #accuracy = valid(args, model, test_loader, global_step)
            #try:
            if True:
                loss_loc, part_loss, contrast_loss, loss_rp_trt, loss_rp_cam, logits, TRT_logits, CAM_logits, GCN_logits = model(x, pathi, y)
                loss_trt = torch.nn.CrossEntropyLoss()(TRT_logits, y)
                loss_cam = torch.nn.CrossEntropyLoss()(CAM_logits, y)
                loss_gcn = torch.nn.CrossEntropyLoss()(GCN_logits, y)
                loss = part_loss + contrast_loss + loss_rp_trt + loss_rp_cam + loss_trt + loss_cam + loss_gcn + loss_loc
                loss = loss.mean()

                preds_TFG = torch.argmax(logits, dim=-1)
                preds_TRT = torch.argmax(TRT_logits, dim=-1)
                preds_CAM = torch.argmax(CAM_logits, dim=-1)
                preds_GCN = torch.argmax(GCN_logits, dim=-1)
                preds = torch.argmax(logits + TRT_logits + CAM_logits + GCN_logits, dim=-1)

                if len(all_preds) == 0:
                    all_preds_TFG.append(preds_TFG.detach().cpu().numpy())
                    all_preds_TRT.append(preds_TRT.detach().cpu().numpy())
                    all_preds_CAM.append(preds_CAM.detach().cpu().numpy())
                    all_preds_GCN.append(preds_GCN.detach().cpu().numpy())
                    all_preds.append(preds.detach().cpu().numpy())
                    all_label.append(y.detach().cpu().numpy())
                else:
                    all_preds_TFG[0] = np.append(
                        all_preds_TFG[0], preds_TFG.detach().cpu().numpy(), axis=0
                    )
                    all_preds_TRT[0] = np.append(
                        all_preds_TRT[0], preds_TRT.detach().cpu().numpy(), axis=0
                    )
                    all_preds_CAM[0] = np.append(
                        all_preds_CAM[0], preds_CAM.detach().cpu().numpy(), axis=0
                    )
                    all_preds_GCN[0] = np.append(
                        all_preds_GCN[0], preds_GCN.detach().cpu().numpy(), axis=0
                    )
                    all_preds[0] = np.append(
                        all_preds[0], preds.detach().cpu().numpy(), axis=0
                    )
                    all_label[0] = np.append(
                        all_label[0], y.detach().cpu().numpy(), axis=0
                    )

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    losses_part.update(part_loss.item()*args.gradient_accumulation_steps)
                    losses_cons.update(contrast_loss.item()*args.gradient_accumulation_steps)
                    losses_rp_trt.update(loss_rp_trt.item()*args.gradient_accumulation_steps)
                    losses_rp_cam.update(loss_rp_cam.item()*args.gradient_accumulation_steps)
                    losses_trt.update(loss_trt.item()*args.gradient_accumulation_steps)
                    losses_cam.update(loss_cam.item()*args.gradient_accumulation_steps)
                    losses_gcn.update(loss_gcn.item()*args.gradient_accumulation_steps)
                    losses_loc.update(loss_loc.item()*args.gradient_accumulation_steps)
                    losses.update(loss.item()*args.gradient_accumulation_steps)
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scheduler.step()
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    epoch_iterator.set_description(
                        "Training (%d/%d Steps) loss=%2.5f(loc=%2.5f,part=%2.5f,cons=%2.5f,rptrt=%2.5f,rpcam=%2.5f,trt=%2.5f,cam=%2.5f,gcn=%2.5f)" % (global_step, t_total, losses.val, losses_loc.val, losses_part.val, losses_cons.val, losses_rp_trt.val, losses_rp_cam.val, losses_trt.val, losses_cam.val, losses_gcn.val)
                    )
                    if global_step % args.eval_every == 0:
                        with torch.no_grad():
                            accuracy = valid(args, model, test_loader, global_step)
                        if args.local_rank in [-1, 0]:
                            if best_acc < accuracy[-1]:
                                save_model(args, model, global_step, accuracy)
                                best_acc = accuracy[-1]
                            logger.info("best accuracy so far: %f" % best_acc)
                            if best_acc_all < np.max(accuracy):
                                save_model(args, model, global_step, accuracy)
                                best_acc_all = np.max(accuracy)
                            logger.info("best accuracy all so far: %f" % best_acc_all)

                        model.train()

                    if global_step % t_total == 0:
                        break
            #except:
            #    print ('Train Error:')

        all_preds_TFG, all_label = all_preds_TFG[0], all_label[0]
        accuracy_TFG = simple_accuracy(all_preds_TFG, all_label)
        accuracy_TFG = torch.tensor(accuracy_TFG).to(args.device)

        all_preds_TRT = all_preds_TRT[0]
        accuracy_TRT = simple_accuracy(all_preds_TRT, all_label)
        accuracy_TRT = torch.tensor(accuracy_TRT).to(args.device)

        all_preds_CAM = all_preds_CAM[0]
        accuracy_CAM = simple_accuracy(all_preds_CAM, all_label)
        accuracy_CAM = torch.tensor(accuracy_CAM).to(args.device)

        all_preds_GCN = all_preds_GCN[0]
        accuracy_GCN = simple_accuracy(all_preds_GCN, all_label)
        accuracy_GCN = torch.tensor(accuracy_GCN).to(args.device)

        all_preds = all_preds[0]
        accuracy = simple_accuracy(all_preds, all_label)
        accuracy = torch.tensor(accuracy).to(args.device)

        accuracy_TFG = accuracy_TFG.detach().cpu().numpy()
        accuracy_TRT = accuracy_TRT.detach().cpu().numpy()
        accuracy_CAM = accuracy_CAM.detach().cpu().numpy()
        accuracy_GCN = accuracy_GCN.detach().cpu().numpy()
        accuracy = accuracy.detach().cpu().numpy()
        logger.info("train acc_TFG: %f, acc_TRT: %f, acc_CAM: %f, acc_GCN: %f, acc: %f" % (accuracy_TFG, accuracy_TRT, accuracy_CAM, accuracy_GCN, accuracy))
        losses.reset()
        if global_step % t_total == 0:
            break

    #writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("End Training!")
    end_time = time.time()
    logger.info("Total Training Time: \t%f" % ((end_time - start_time) / 3600))

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", default = 'sample_run',
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["CUB_200_2011", "car", "dog", "nabirds", "INat2017"], default="CUB_200_2011",
                        help="Which dataset.")
    parser.add_argument('--data_root', type=str, default='./opt/tiger/minist')
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="./opt/tiger/minist/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--pretrained_model", type=str, default='./checkpoints/cub_200_2011/ViT-B_16_overlap.pth',
    #parser.add_argument("--pretrained_model", type=str, default='./outputs/cub_200_2011_v10/checkpoint_cub_step1_v10_36000_0.90403_0.90248_0.90645.pth',
                        help="load pretrained model")
    parser.add_argument("--output_dir", default="./outputs/cub_200_2011_v10", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--img_size", default=448, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=2, type=int,
                        help="Total batch size for training.")#16
    parser.add_argument("--eval_batch_size", default=2, type=int,
                        help="Total batch size for eval.")#8
    parser.add_argument("--eval_every", default=1000, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=20000*8, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true', default=True,
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    parser.add_argument('--smoothing_value', type=float, default=0.0,
                        help="Label smoothing value\n")

    parser.add_argument('--split', type=str, default='overlap', #non-overlap
                        help="Split method")
    parser.add_argument('--slide_step', type=int, default=12,
                        help="Slide step for overlap split")

    args = parser.parse_args()

    # if args.fp16 and args.smoothing_value != 0:
    #     raise NotImplementedError("label smoothing not supported for fp16 training now")
    args.data_root = '{}/{}'.format(args.data_root, args.dataset)
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device
    args.nprocs = torch.cuda.device_count()

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)
    # Training
    train(args, model)

if __name__ == "__main__":
    main()
