import argparse
import os

import pandas as pd
import torch
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from model import Model

try:
    # from apex.parallel import DistributedDataParallel as DDP
    # from apex.fp16_utils import *
    # from apex import amp, optimizers
    # from apex.multi_tensor_apply import multi_tensor_applier
    from apex import amp
except ImportError:
    print("AMP is not installed. If --amp is True, code will fail.")  

import wandb
# wandb.login()

import random
import numpy as np

from io_utils import set_seed, parse_args
args = parse_args('main')

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["WANDB_API_KEY"] = 'ae69f83abb637683132c012cd248d4a14177cd36'
os.environ['WANDB_MODE'] = args.wandb

# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer, example_ct, batch_ct):

    wandb.watch(net, log="all", log_freq=10)

    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader, desc='Data loading')
    for pos_1, pos_2, target in train_bar:
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        train_optimizer.zero_grad()

        ####### apex ######
        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

        example_ct +=  len(pos_1)
        batch_ct += 1
        if ((batch_ct + 1) % 20) == 0:
            wandb.log({"epoch": epoch, "loss": total_loss / total_num}, step=example_ct)
            # print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")

    return total_loss / total_num


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))
        
        wandb.log({"top1": total_top1 / total_num * 100, "top5": total_top5 / total_num * 100})
    
    batch_size = 1
    input_shape = (3, 32, 32)
    x = torch.randn(batch_size,*input_shape).cuda()
    torch.onnx.export(net,x, "SimCLR1.onnx")
    wandb.save("SimCLR1.onnx")

    return total_top1 / total_num * 100, total_top5 / total_num * 100


if __name__ == '__main__':

    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, epochs = args.batch_size, args.epochs

    path, path_d = args.results_path, args.datasets_path
    use_checkpoint = args.use_checkpoint
    start_epoch = args.start_epoch


    ###### seed ######
    set_seed(args.seed)

    ###### wandb ######
    wandb.init(project="simclr1")
    wandb.config.update(args)

    # data prepare
    train_data = utils.CIFAR10Pair(root=path_d, train=True, transform=utils.train_transform, download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,drop_last=True)

    memory_data = utils.CIFAR10Pair(root=path_d, train=True, transform=utils.test_transform, download=True)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    test_data = utils.CIFAR10Pair(root=path_d, train=False, transform=utils.test_transform, download=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)


    # model setup and optimizer config
    model = Model(feature_dim, use_checkpoint).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    ####### apex ######
    if args.amp:
        opt_level = args.amp_level
        model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level) 

    ####### checkpoint ######
    if args.resume:
        resume = torch.load(args.resume_path, map_location='cpu')
        model.load_state_dict(resume['model'], False)
        optimizer.load_state_dict(resume['optimizer'])
        amp.load_state_dict(resume['amp'])
        start_epoch = resume['epoch'] +  1


    # 计算模型的 参数量 和 浮点数
    flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))
    c = len(memory_data.classes)

    # training loop
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    save_name_pre = '{}_{}_{}_{}_{}'.format(feature_dim, temperature, k, batch_size, epochs)
    if not os.path.exists(path):
        os.mkdir(path)
    best_acc = 0.0

    example_ct = 0  # number of examples seen
    batch_ct = 0

    for epoch in range(start_epoch, epochs + 1):
        train_loss = train(model, train_loader, optimizer, example_ct, batch_ct)
        results['train_loss'].append(train_loss)
        test_acc_1, test_acc_5 = test(model, memory_loader, test_loader)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('{}/{}_statistics.csv'.format(path, save_name_pre), index_label='epoch')
        # Save checkpoint
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'amp': amp.state_dict(),
            'epoch': epoch
        }
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            torch.save(checkpoint, '{}/{}_model.pth'.format(path, save_name_pre))
    wandb.save('{}/{}_model.pth'.format(path, save_name_pre))
