from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
import time

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes_mix, Cityscapes
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
from utils.visualizer import Visualizer

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

consistency_loss = 'CE'
consistency_weight = 1

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=['deeplabv3_resnet50',  'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet18', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)
    
    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    # Mixing strategy
    parser.add_argument("--mix", type=str, default='watershedmix',
                        choices=['watershedmix',  'cut'], help='mixing strategy name choose between cutmix -> cut and watershedmix')

    return parser

class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, confidences, predictions, labels):

        accuracies = predictions.eq(labels)
        ece = torch.zeros(1, device=confidences.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

ECE = _ECELoss()

def get_dataset(opts):
    """ Dataset And Augmentation
    """
    mix_mask = opts.mix
    watershed=False
    if mix_mask == 'watershedmix': watershed =True
    if opts.dataset == 'voc':
        train_transform = et.ExtCompose([
            #et.ExtResize(size=opts.crop_size),
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        train_dst_labelled = VOCSegmentation(root=opts.data_root, year=opts.year,
                                    image_set='train', download=opts.download, transform=train_transform)
        train_dst_unlabelled = VOCSegmentation(root=opts.data_root, year=opts.year,
                                    image_set='train', download=opts.download, transform=train_transform)
        val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                  image_set='val', download=False, transform=val_transform)
    if opts.dataset == 'cityscapes':
        train_transform_simple = et.ExtCompose([
            #et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter( brightness=0.5, contrast=0.5, saturation=0.5 ),
            et.ExtRandomHorizontalFlip(),
        ])
        train_transform_tensor = et.ExtCompose([

            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        val_transform = et.ExtCompose([
            #et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        train_dst_labelled = Cityscapes_mix(root=opts.data_root,
                               split='train', transform1=train_transform_simple, transform2=train_transform_tensor,watershed=False)
        train_dst_unlabelled = Cityscapes_mix(root=opts.data_root,
                               split='train', transform1=train_transform_simple, transform2=train_transform_tensor,watershed=watershed)
        val_dst = Cityscapes(root=opts.data_root,
                             split='val', transform=val_transform)
    return train_dst_labelled,train_dst_unlabelled, val_dst


def validate(opts, model1,model2,model3, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    softmax = torch.nn.Softmax2d()
    ret_samples = []
    ece=[]
    NLL=[]
    criterion_NLL = nn.NLLLoss(ignore_index=255, reduction='mean')
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):
            
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs1 = model1(images)
            outputs2 = model2(images)
            outputs3 = model3(images)
            outputs=(softmax(outputs1)+softmax(outputs2)+softmax(outputs3))/3.0
            nll_out=criterion_NLL(outputs, labels)
            NLL.append(nll_out.cpu().item())
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
            #outputs=softmax(outputs)
            conf, preds_val  = torch.max(outputs,dim=1)

            ece_out = ECE.forward(conf.squeeze(), preds_val.squeeze(),labels)
            ece.append(ece_out.cpu().item())


            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)



                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score, ret_samples,ece,NLL

def create_ema_model(model,modelname,num_classes,output_stride,gpus):
    model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet,
        'deeplabv3plus_resnet18' : network.deeplabv3plus_resnet18
    }

    ema_model = model_map[modelname](num_classes=num_classes, output_stride=output_stride)

    for param in ema_model.parameters():
        param.detach_()
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()
    if len(gpus)>1:
        ema_model = torch.nn.DataParallel(ema_model)
    return ema_model

def update_ema_variables(ema_model, model, alpha_teacher, iteration,gpus):
    # Use the "true" average until the exponential average is more correct
    alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)
    if len(gpus)>1:
        for ema_param, param in zip(ema_model.module.parameters(), model.module.parameters()):
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    else:
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

def mix(mask, data = None, target = None):
    #Mix
    if not (data is None):
        if mask.shape[0] == data.shape[0]:
            data = torch.cat([(mask[i] * data[i] + (1 - mask[i]) * data[(i + 1) % data.shape[0]]).unsqueeze(0) for i in range(data.shape[0])])
        elif mask.shape[0] == data.shape[0] / 2:
            data = torch.cat((torch.cat([(mask[i] * data[2 * i] + (1 - mask[i]) * data[2 * i + 1]).unsqueeze(0) for i in range(int(data.shape[0] / 2))]),
                              torch.cat([((1 - mask[i]) * data[2 * i] + mask[i] * data[2 * i + 1]).unsqueeze(0) for i in range(int(data.shape[0] / 2))])))
    if not (target is None):
        target = torch.cat([(mask[i] * target[i] + (1 - mask[i]) * target[(i + 1) % target.shape[0]]).unsqueeze(0) for i in range(target.shape[0])])
    return data, target

def generate_cutout_mask(img_size, seed = None):
    np.random.seed(seed)

    cutout_area = img_size[0] * img_size[1] / 2

    w = np.random.randint(img_size[1] / 2, img_size[1])
    h = np.amin((np.round(cutout_area / w),img_size[0]))

    x_start = np.random.randint(0, img_size[1] - w + 1)
    y_start = np.random.randint(0, img_size[0] - h + 1)

    x_end = int(x_start + w)
    y_end = int(y_start + h)

    mask = np.ones(img_size)
    mask[y_start:y_end, x_start:x_end] = 0
    return mask.astype(float)

class CrossEntropyLoss2dPixelWiseWeighted(nn.Module):
    def __init__(self, weight=None, ignore_index=250, reduction='none'):
        super(CrossEntropyLoss2dPixelWiseWeighted, self).__init__()
        self.CE =  nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target, pixelWiseWeight):
        loss = self.CE(output, target)
        loss = torch.mean(loss * pixelWiseWeight)
        return loss

class MSELoss2d(nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean', ignore_index=255):
        super(MSELoss2d, self).__init__()
        self.MSE = nn.MSELoss(size_average=size_average, reduce=reduce, reduction=reduction)

    def forward(self, output, target):
        loss = self.MSE(torch.softmax(output, dim=1), target)
        return loss

def main():

    opts = get_argparser().parse_args()
    mix_mask = opts.mix
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19

    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    if opts.dataset=='voc' and not opts.crop_val:
        opts.val_batch_size = 1
    
    train_dst_labelled, train_dst_unlabelled, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dst_labelled, batch_size=opts.batch_size, shuffle=True, num_workers=2)
    train_loader_unlabelled = data.DataLoader(
        train_dst_unlabelled, batch_size=opts.batch_size, shuffle=True, num_workers=2)
    train_loader_unlabelled_iter = iter(train_loader_unlabelled)
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
    #interp = nn.Upsample(size=(opts.crop_size, opts.crop_size), mode='bilinear', align_corners=True)
    
    if consistency_loss == 'CE':
         unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted().cuda()
    elif consistency_loss == 'MSE':
        unlabeled_loss =  MSELoss2d().cuda()

    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst_labelled), len(val_dst)))

    # Set up model
    model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet,
        'deeplabv3plus_resnet18' : network.deeplabv3plus_resnet18
    }

    model1 = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    model2 = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    model3 = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)


    if opts.separable_conv and 'plus' in opts.model1:
        network.convert_to_separable_conv(model1.classifier)
    utils.set_bn_momentum(model1.backbone, momentum=0.01)
    if opts.separable_conv and 'plus' in opts.model2:
        network.convert_to_separable_conv(model2.classifier)
    utils.set_bn_momentum(model2.backbone, momentum=0.01)

    if opts.separable_conv and 'plus' in opts.model3:
        network.convert_to_separable_conv(model3.classifier)
    utils.set_bn_momentum(model3.backbone, momentum=0.01)
    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model1.backbone.parameters(), 'lr': 0.1*opts.lr},
        {'params': model1.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    #optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    #torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    if opts.lr_policy=='poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy=='step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    #criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model1.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)
    
    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load('/home/soumik/workspace/workspacegianni/deeplabv3-watershedmix/checkpoints/R101_b16_o8_cr768_lr0_1_seed0/best80_deeplabv3plus_resnet101_cityscapes_os8.pth', map_location=torch.device('cpu'))
        model1.load_state_dict(checkpoint["model_state"])
        model1 = nn.DataParallel(model1)
        model1.to(device)
        checkpoint = torch.load('/home/soumik/workspace/workspacegianni/deeplabv3-watershedmix/checkpoints/R101_b16_o8_cr768_lr0_1_seed1/best80_deeplabv3plus_resnet101_cityscapes_os8.pth', map_location=torch.device('cpu'))
        model2.load_state_dict(checkpoint["model_state"])
        model2 = nn.DataParallel(model2)
        model2.to(device)
        checkpoint = torch.load('/home/soumik/workspace/workspacegianni/deeplabv3-watershedmix/checkpoints/R101_b16_o8_cr768_lr0_1_seed2/best80_deeplabv3plus_resnet101_cityscapes_os8.pth', map_location=torch.device('cpu'))
        model3.load_state_dict(checkpoint["model_state"])
        model3 = nn.DataParallel(model3)
        model3.to(device)
    else:
        print("[!] Retrain")
        model1 = nn.DataParallel(model1)
        model1.to(device)


    #==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images


    model1.eval()
    model2.eval()
    model3.eval()
    val_score, ret_samples,ece,NLL = validate(
        opts=opts, model1=model1, model2=model2, model3=model3, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
    print(metrics.to_str(val_score))
    print('--------------------------------------------------------------------')
    print('ECE!!!!! mean ECE = ', np.mean(np.asarray(ece)))
    print('NLL!!!!! mean ECE = ', np.mean(np.asarray(NLL)))
    return




        
if __name__ == '__main__':
    main()
