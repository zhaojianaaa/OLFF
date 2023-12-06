import torch

def check_load(model, pretrained_model):
    checkpoint = torch.load(pretrained_model, map_location="cpu")['model']
    model_dict = model.state_dict()
    for k, v in model_dict.items():
        if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
            del checkpoint[k]
            print(k + ' delete')
        elif k not in checkpoint and 'layer.11' in k:
            checkpoint[k] = checkpoint[k.replace('layer.11','layer.10')]
            print(k + ' copy from ' + k.replace('layer.11','layer.10'))
        elif k not in checkpoint and 'part_layer' in k:
            checkpoint[k] = checkpoint[k.replace('part_layer','layer.11')]
            print(k + ' copy from ' + k.replace('part_layer','layer.11'))
        #elif k not in checkpoint and 'score_layer' in k:
        #    checkpoint[k] = checkpoint[k.replace('score_layer','layer.11')]
        #    print(k + ' copy from ' + k.replace('score_layer','layer.11'))
        #elif k not in checkpoint and 'part_layer' in k:
        #    checkpoint[k] = checkpoint[k.replace('part_layer','layer.10')]
        #    print(k + ' copy from ' + k.replace('part_layer','layer.10'))
        #elif k not in checkpoint and 'score_norm' in k:
        #    checkpoint[k] = checkpoint[k.replace('score_norm','part_norm')]
        #    print(k + ' copy from ' + k.replace('score_layer','part_norm'))

    model_dict.update(checkpoint)
    model.load_state_dict(model_dict)
    return model

def check_load_fuse(model, pretrained_model):
    checkpoint = torch.load(pretrained_model, map_location="cpu")['model']
    model_dict = model.state_dict()
    for k, v in model_dict.items():
        if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
            del checkpoint[k]
            print(k + ' delete')
        elif k not in checkpoint and 'layer.11' in k:
            checkpoint[k] = checkpoint[k.replace('layer.11','layer.10')]
            print(k + ' copy from ' + k.replace('layer.11','layer.10'))
        elif k not in checkpoint and 'part_layer' in k:
            checkpoint[k] = checkpoint[k.replace('part_layer','layer.11')]
            print(k + ' copy from ' + k.replace('part_layer','layer.11'))
        elif k not in checkpoint and 'camfuse_block.cam_tr' in k:
            checkpoint[k] = checkpoint[k.replace('camfuse_block.cam_tr','layer.11')]
            print(k + ' copy from ' + k.replace('camfuse_block.cam_tr','layer.11'))
        elif k not in checkpoint and 'camfuse_block_obj.cam_tr' in k:
            checkpoint[k] = checkpoint[k.replace('camfuse_block_obj.cam_tr','layer.11')]
            print(k + ' copy from ' + k.replace('camfuse_block_obj.cam_tr','layer.11'))
        elif k not in checkpoint and 'score_layer' in k:
            checkpoint[k] = checkpoint[k.replace('score_layer','layer.11')]
            print(k + ' copy from ' + k.replace('score_layer','layer.11'))
        elif k not in checkpoint and 'obj_layer' in k:
            checkpoint[k] = checkpoint[k.replace('obj_layer','layer.11')]
            print(k + ' copy from ' + k.replace('obj_layer','layer.11'))


    model_dict.update(checkpoint)
    model.load_state_dict(model_dict, strict=False)
    return model

def simple_accuracy(preds, labels):
    try:
        return (preds == labels).mean()
    except:
        return 0.0

def resultss(args, all_label, GCN_preds, TFG_preds, TRT_preds, CAM_preds, GCN_TFG_preds, GCN_TRT_preds, GCN_CAM_preds, TFG_TRT_preds, TFG_CAM_preds, TRT_CAM_preds, nor_GCN_TFG_preds, nor_GCN_TRT_preds, nor_GCN_CAM_preds, nor_TFG_TRT_preds, nor_TFG_CAM_preds, nor_TRT_CAM_preds, GCN_TFG_TRT_preds, GCN_TFG_CAM_preds, GCN_TRT_CAM_preds, TFG_TRT_CAM_preds, nor_GCN_TFG_TRT_preds, nor_GCN_TFG_CAM_preds, nor_GCN_TRT_CAM_preds, nor_TFG_TRT_CAM_preds, GCN_TFG_TRT_CAM_preds, nor_GCN_TFG_TRT_CAM_preds):
 
    #c51
    all_label = all_label[0]

    GCN_preds = GCN_preds[0]
    accuracy_GCN = simple_accuracy(GCN_preds, all_label)
    accuracy_GCN = torch.tensor(accuracy_GCN).to(args.device)

    TFG_preds = TFG_preds[0]
    accuracy_TFG = simple_accuracy(TFG_preds, all_label)
    accuracy_TFG = torch.tensor(accuracy_TFG).to(args.device)

    TRT_preds = TRT_preds[0]
    accuracy_TRT = simple_accuracy(TRT_preds, all_label)
    accuracy_TRT = torch.tensor(accuracy_TRT).to(args.device)

    CAM_preds = CAM_preds[0]
    accuracy_CAM = simple_accuracy(CAM_preds, all_label)
    accuracy_CAM = torch.tensor(accuracy_CAM).to(args.device)

    # c52
    GCN_TFG_preds = GCN_TFG_preds[0]
    accuracy_GCN_TFG = simple_accuracy(GCN_TFG_preds, all_label)
    accuracy_GCN_TFG = torch.tensor(accuracy_GCN_TFG).to(args.device)

    GCN_TRT_preds = GCN_TRT_preds[0]
    accuracy_GCN_TRT = simple_accuracy(GCN_TRT_preds, all_label)
    accuracy_GCN_TRT = torch.tensor(accuracy_GCN_TRT).to(args.device)

    GCN_CAM_preds = GCN_CAM_preds[0]
    accuracy_GCN_CAM = simple_accuracy(GCN_CAM_preds, all_label)
    accuracy_GCN_CAM = torch.tensor(accuracy_GCN_CAM).to(args.device)

    TFG_TRT_preds = TFG_TRT_preds[0]
    accuracy_TFG_TRT = simple_accuracy(TFG_TRT_preds, all_label)
    accuracy_TFG_TRT = torch.tensor(accuracy_TFG_TRT).to(args.device)

    TFG_CAM_preds = TFG_CAM_preds[0]
    accuracy_TFG_CAM = simple_accuracy(TFG_CAM_preds, all_label)
    accuracy_TFG_CAM = torch.tensor(accuracy_TFG_CAM).to(args.device)

    TRT_CAM_preds = TRT_CAM_preds[0]
    accuracy_TRT_CAM = simple_accuracy(TRT_CAM_preds, all_label)
    accuracy_TRT_CAM = torch.tensor(accuracy_TRT_CAM).to(args.device)

    # c52_nor
    nor_GCN_TFG_preds = nor_GCN_TFG_preds[0]
    nor_accuracy_GCN_TFG = simple_accuracy(nor_GCN_TFG_preds, all_label)
    nor_accuracy_GCN_TFG = torch.tensor(nor_accuracy_GCN_TFG).to(args.device)

    nor_GCN_TRT_preds = nor_GCN_TRT_preds[0]
    nor_accuracy_GCN_TRT = simple_accuracy(nor_GCN_TRT_preds, all_label)
    nor_accuracy_GCN_TRT = torch.tensor(nor_accuracy_GCN_TRT).to(args.device)

    nor_GCN_CAM_preds = nor_GCN_CAM_preds[0]
    nor_accuracy_GCN_CAM = simple_accuracy(nor_GCN_CAM_preds, all_label)
    nor_accuracy_GCN_CAM = torch.tensor(nor_accuracy_GCN_CAM).to(args.device)

    nor_TFG_TRT_preds = nor_TFG_TRT_preds[0]
    nor_accuracy_TFG_TRT = simple_accuracy(nor_TFG_TRT_preds, all_label)
    nor_accuracy_TFG_TRT = torch.tensor(nor_accuracy_TFG_TRT).to(args.device)

    nor_TFG_CAM_preds = nor_TFG_CAM_preds[0]
    nor_accuracy_TFG_CAM = simple_accuracy(nor_TFG_CAM_preds, all_label)
    nor_accuracy_TFG_CAM = torch.tensor(nor_accuracy_TFG_CAM).to(args.device)

    nor_TRT_CAM_preds = nor_TRT_CAM_preds[0]
    nor_accuracy_TRT_CAM = simple_accuracy(nor_TRT_CAM_preds, all_label)
    nor_accuracy_TRT_CAM = torch.tensor(nor_accuracy_TRT_CAM).to(args.device)

   # c53
    GCN_TFG_TRT_preds = GCN_TFG_TRT_preds[0]
    accuracy_GCN_TFG_TRT = simple_accuracy(GCN_TFG_TRT_preds, all_label)
    accuracy_GCN_TFG_TRT = torch.tensor(accuracy_GCN_TFG_TRT).to(args.device)

    GCN_TFG_CAM_preds = GCN_TFG_CAM_preds[0]
    accuracy_GCN_TFG_CAM = simple_accuracy(GCN_TFG_CAM_preds, all_label)
    accuracy_GCN_TFG_CAM = torch.tensor(accuracy_GCN_TFG_CAM).to(args.device)

    GCN_TRT_CAM_preds = GCN_TRT_CAM_preds[0]
    accuracy_GCN_TRT_CAM = simple_accuracy(GCN_TRT_CAM_preds, all_label)
    accuracy_GCN_TRT_CAM = torch.tensor(accuracy_GCN_TRT_CAM).to(args.device)

    TFG_TRT_CAM_preds = TFG_TRT_CAM_preds[0]
    accuracy_TFG_TRT_CAM = simple_accuracy(TFG_TRT_CAM_preds, all_label)
    accuracy_TFG_TRT_CAM = torch.tensor(accuracy_TFG_TRT_CAM).to(args.device)

   # c53_nor
    nor_GCN_TFG_TRT_preds = nor_GCN_TFG_TRT_preds[0]
    nor_accuracy_GCN_TFG_TRT = simple_accuracy(nor_GCN_TFG_TRT_preds, all_label)
    nor_accuracy_GCN_TFG_TRT = torch.tensor(nor_accuracy_GCN_TFG_TRT).to(args.device)

    nor_GCN_TFG_CAM_preds = nor_GCN_TFG_CAM_preds[0]
    nor_accuracy_GCN_TFG_CAM = simple_accuracy(nor_GCN_TFG_CAM_preds, all_label)
    nor_accuracy_GCN_TFG_CAM = torch.tensor(nor_accuracy_GCN_TFG_CAM).to(args.device)

    nor_GCN_TRT_CAM_preds = nor_GCN_TRT_CAM_preds[0]
    nor_accuracy_GCN_TRT_CAM = simple_accuracy(nor_GCN_TRT_CAM_preds, all_label)
    nor_accuracy_GCN_TRT_CAM = torch.tensor(nor_accuracy_GCN_TRT_CAM).to(args.device)

    nor_TFG_TRT_CAM_preds = nor_TFG_TRT_CAM_preds[0]
    nor_accuracy_TFG_TRT_CAM = simple_accuracy(nor_TFG_TRT_CAM_preds, all_label)
    nor_accuracy_TFG_TRT_CAM = torch.tensor(nor_accuracy_TFG_TRT_CAM).to(args.device)

    # c54
    GCN_TFG_TRT_CAM_preds = GCN_TFG_TRT_CAM_preds[0]
    accuracy_GCN_TFG_TRT_CAM = simple_accuracy(GCN_TFG_TRT_CAM_preds, all_label)
    accuracy_GCN_TFG_TRT_CAM = torch.tensor(accuracy_GCN_TFG_TRT_CAM).to(args.device)

    # c54_nor
    nor_GCN_TFG_TRT_CAM_preds = nor_GCN_TFG_TRT_CAM_preds[0]
    nor_accuracy_GCN_TFG_TRT_CAM = simple_accuracy(nor_GCN_TFG_TRT_CAM_preds, all_label)
    nor_accuracy_GCN_TFG_TRT_CAM = torch.tensor(nor_accuracy_GCN_TFG_TRT_CAM).to(args.device)

    # c55 and c55_nor

    return accuracy_GCN, accuracy_TFG, accuracy_TRT, accuracy_CAM, accuracy_GCN_TFG, accuracy_GCN_TRT, accuracy_GCN_CAM, accuracy_TFG_TRT, accuracy_TFG_CAM, accuracy_TRT_CAM, nor_accuracy_GCN_TFG, nor_accuracy_GCN_TRT, nor_accuracy_GCN_CAM, nor_accuracy_TFG_TRT, nor_accuracy_TFG_CAM, nor_accuracy_TRT_CAM, accuracy_GCN_TFG_TRT, accuracy_GCN_TFG_CAM, accuracy_GCN_TRT_CAM, accuracy_TFG_TRT_CAM, nor_accuracy_GCN_TFG_TRT, nor_accuracy_GCN_TFG_CAM, nor_accuracy_GCN_TRT_CAM, nor_accuracy_TFG_TRT_CAM, nor_accuracy_GCN_TFG_TRT_CAM, accuracy_GCN_TFG_TRT_CAM

















