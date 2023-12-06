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
        elif k not in checkpoint and 'score_layer' in k:
            checkpoint[k] = checkpoint[k.replace('score_layer','layer.11')]
            print(k + ' copy from ' + k.replace('score_layer','layer.11'))
        #elif k not in checkpoint and 'part_layer' in k:
        #    checkpoint[k] = checkpoint[k.replace('part_layer','layer.10')]
        #    print(k + ' copy from ' + k.replace('part_layer','layer.10'))
        elif k not in checkpoint and 'score_norm' in k:
            checkpoint[k] = checkpoint[k.replace('score_norm','part_norm')]
            print(k + ' copy from ' + k.replace('score_layer','part_norm'))

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
        elif k not in checkpoint and 'score_layer' in k:
            checkpoint[k] = checkpoint[k.replace('score_layer','layer.11')]
            print(k + ' copy from ' + k.replace('score_layer','layer.11'))
        elif k not in checkpoint and 'part_sim_layer' in k:
            checkpoint[k] = checkpoint[k.replace('part_sim_layer','layer.11')]
            print(k + ' copy from ' + k.replace('part_sim_layer','layer.11'))
        elif k not in checkpoint and 'cam_tr' in k:
            checkpoint[k] = checkpoint[k.replace('camfuse_block.cam_tr','layer.11')]
            print(k + ' copy from ' + k.replace('camfuse_block.cam_tr','layer.11'))
        elif k not in checkpoint and 'score_norm' in k:
            checkpoint[k] = checkpoint[k.replace('score_norm','part_norm')]
            print(k + ' copy from ' + k.replace('score_norm','part_norm'))
        elif k not in checkpoint and 'cam_norm' in k:
            checkpoint[k] = checkpoint[k.replace('camfuse_block.cam_norm','part_norm')]
            print(k + ' copy from ' + k.replace('camfuse_block.cam_norm','part_norm'))

    model_dict.update(checkpoint)
    model.load_state_dict(model_dict, strict=False)
    return model
