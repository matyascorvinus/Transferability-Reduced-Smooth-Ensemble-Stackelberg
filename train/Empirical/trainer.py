

from utils.Empirical.attack import arc_attack, apgd_attack
from utils.Empirical.datasets import get_dataset, get_normalize_layer
from utils.Empirical.utils_ensemble import AverageMeter, accuracy, test, copy_code, requires_grad_, evaltrans, arr_to_str, proj_onto_simplex
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from utils.Empirical.third_party.distillation import Linf_distillation
from utils.Empirical.utils_ensemble import Cosine, Magnitude, Combinator
from models.ensemble import Ensemble
from utils.Empirical.utils_ensemble import AverageMeter, accuracy, test, copy_code, requires_grad_
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import time
import torch.nn.functional as F
import torch.autograd as autograd
import sys
import os

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir)


# PGD - projected gradient descent
# The PGD algorithm works by iteratively applying a small perturbation to the input that
# maximizes the loss of the models on the target label. In each iteration, the perturbation is clipped to ensure that
# it stays within a certain range of values. This helps to prevent the perturbation from becoming too large and
# making the adversarial example too obvious. The perturbation is initialized to zero, and the number of iterations


def osp_iter(epoch, alpha, lr_scheduler, osp_loader, model_ls, normalize, osp_lr_max, attack='apgd', debug_mode=False):

    M = len(alpha)
    err = torch.zeros(M, device='cuda')
    n = 0
    pbar = tqdm(osp_loader)
    osp_lr_init = osp_lr_max * lr_scheduler.get_lr()[0]
    curr_lr = osp_lr_init/(1 + epoch)
    model_ls[-1].eval()
    pbar.set_description("OSP:{:3d} epoch lr {:.4f}".format(epoch, curr_lr))
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.cuda(), targets.cuda()
        # adv_inp = arc_attack(model_ls, inputs, targets, alpha,
        #                      8 / 255.0, 8 / 255.0, 10, normalize=normalize, g=2)
        if attack == 'apgd':
            adv_inp = apgd_attack(model_ls, inputs, targets, alpha,
                                    8 / 255.0, 2 / 255.0, 10, normalize=normalize)
        if attack == 'arc':
            adv_inp = arc_attack(model_ls, inputs, targets, alpha,
                                    8 / 255.0, 8 / 255.0, 10, normalize=normalize, g=2)
        for m in range(M):
            t_m = model_ls[m](normalize(adv_inp))
            err[m] += (t_m.max(1)[1] != targets).sum().item()

        n += targets.size(0)
        pbar_dic = OrderedDict()
        pbar_dic['Adv Acc'] = '{:2.2f}'.format(
            100. * ( 1 - torch.sum(err*alpha) / n))
        pbar.set_postfix(pbar_dic)
        # alpha = alpha if isinstance(alpha, torch.Tensor)  else torch.from_numpy(alpha)

    grad = err / n
    return grad


def tuning_alpha(epoch, total_epochs, lr_scheduler, alpha, osp_loader, models, dataset, osp_freq=10, osp_lr_max=10, debug_mode=False, attack='apgd'):
    if epoch >= (total_epochs // 2) and ((epoch - total_epochs // 2 + 1) % osp_freq == 0 or epoch == total_epochs - 1) or debug_mode:
        for iteration in range(0, len(models)):
            if iteration > 2:
                eta_best = 1
                osp_lr_init = osp_lr_max * lr_scheduler.get_lr()[0]
                print('==> Begin OSP routine, starting alpha=' + arr_to_str(alpha))
                for t in range(total_epochs):
                    osp_lr = osp_lr_init / (t + 1)
                    normalize = get_normalize_layer(dataset)
                    # sub-gradient of eta(alpha_t)
                    g_t = osp_iter(t, alpha, lr_scheduler,
                                   osp_loader, models, normalize, osp_lr_max, attack = attack, debug_mode=debug_mode)
                    eta_t = sum(g_t * alpha)  # eta(alpha_t)
                    if eta_t <= eta_best:
                        t_best = t
                        prob_best = alpha.clone()
                        eta_best = eta_t
                    print(
                        "best acc = {:2.2f} @ alpha_best = ".format(100.*(1-eta_best)) + arr_to_str(prob_best))
                    alpha = proj_onto_simplex(
                        alpha - osp_lr * g_t)
                print('==> End OSP routine, final alpha=' + arr_to_str(prob_best))
                # alpha = torch.from_numpy(np.copy(prob_best))  # update alpha
                alpha = prob_best.clone() # update alpha


def PGD(models, inputs, labels, eps):
    steps = 6
    alpha = eps / 3.

    adv = inputs.detach() + torch.FloatTensor(inputs.shape).uniform_(-eps, eps).cuda()
    adv = torch.clamp(adv, 0, 1)
    criterion = nn.CrossEntropyLoss()

    adv.requires_grad = True
    for _ in range(steps):
        # adv.requires_grad_()
        grad_loss = 0
        for i, m in enumerate(models):
            loss = criterion(m(adv), labels)
            grad = autograd.grad(loss, adv, create_graph=True)[0]
            grad = grad.flatten(start_dim=1)
            grad_loss += Magnitude(grad)

        grad_loss /= 3
        grad_loss.backward()
        sign_grad = adv.grad.data.sign()
        with torch.no_grad():
            adv.data = adv.data + alpha * sign_grad
            adv.data = torch.max(
                torch.min(adv.data, inputs + eps), inputs - eps)
            adv.data = torch.clamp(adv.data, 0., 1.)

    adv.grad = None
    return adv.detach()


def TRS_Trainer(args, loader: DataLoader, models, criterion, optimizer: Optimizer,
                epoch: int, device: torch.device, osp_loader, scheduler, writer=None, alpha=1/3):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    cos_losses = AverageMeter()
    smooth_losses = AverageMeter()
    # cos01_losses = AverageMeter()
    # cos02_losses = AverageMeter()
    # cos12_losses = AverageMeter()
    cos_losses_array = []
    list_of_combination = Combinator(args.num_models)
    for index, combination in list_of_combination:
        cos_losses_array.append(AverageMeter())
    end = time.time()

    for i in range(args.num_models):
        models[i].train()
        requires_grad_(models[i], True)

    for i, (inputs, targets) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)
        inputs.requires_grad = True
        grads = []
        loss_std = 0
        for j in range(args.num_models):
            logits = models[j](inputs)
            loss = criterion(logits, targets)
            grad = autograd.grad(loss, inputs, create_graph=True)[0]
            grad = grad.flatten(start_dim=1)
            grads.append(grad)
            loss_std += loss

        cos_loss, smooth_loss = 0, 0

        # cos01 = Cosine(grads[0], grads[1])
        # cos02 = Cosine(grads[0], grads[2])
        # cos12 = Cosine(grads[1], grads[2])

        # cos_loss = (cos01 + cos02 + cos12) / 3.

        cos_array = []
        for combination in list_of_combination:
            cos_array.append(
                Cosine(grads[combination[0]], grads[combination[1]]))
        cos_loss = sum(cos_array) / len(cos_array)

        N = inputs.shape[0] // 2
        cureps = (args.adv_eps - args.init_eps) * \
            epoch / args.epochs + args.init_eps
        # PGD(self.models, inputs[:N], targets[:N])
        clean_inputs = inputs[:N].detach()
        adv_inputs = PGD(models, inputs[N:], targets[N:], cureps).detach()

        adv_x = torch.cat([clean_inputs, adv_inputs])

        adv_x.requires_grad = True

        if (args.plus_adv):
            for j in range(args.num_models):
                outputs = models[j](adv_x)
                loss = criterion(outputs, targets)
                grad = autograd.grad(loss, adv_x, create_graph=True)[0]
                grad = grad.flatten(start_dim=1)
                smooth_loss += Magnitude(grad)

        else:
            # grads = []
            for j in range(args.num_models):
                outputs = models[j](inputs)
                loss = criterion(outputs, targets)
                grad = autograd.grad(loss, inputs, create_graph=True)[0]
                grad = grad.flatten(start_dim=1)
                smooth_loss += Magnitude(grad)

        smooth_loss /= 3

        loss = loss_std + args.scale * \
            (args.coeff * cos_loss + args.lamda * smooth_loss)
        # alpha = 1 / num_models
        ensemble = Ensemble(models, alpha, True)
        # OSP Algorithm - This caused the performance problem
        tuning_alpha(epoch, args.epochs, scheduler, alpha, osp_loader, models,
                     args.dataset, osp_freq=args.osp_freq, attack = args.attack, debug_mode=args.debug == 1)

        logits = ensemble(inputs)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
        losses.update(loss.item(), batch_size)
        top1.update(acc1.item(), batch_size)
        top5.update(acc5.item(), batch_size)
        cos_losses.update(cos_loss.item(), batch_size)
        smooth_losses.update(smooth_loss.item(), batch_size)
        # cos01_losses.update(cos01.item(), batch_size)
        # cos02_losses.update(cos02.item(), batch_size)
        # cos12_losses.update(cos12.item(), batch_size)
        for index, combination in list_of_combination:
            cos_losses_array[index].update(cos_array[index].item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f}\t'
                  'Acc@1 {top1.avg:.3f}\t'
                  'Acc@5 {top5.avg:.3f}'.format(
                      epoch, i, len(loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1, top5=top5))
            if i >= 10 and args.debug == 1:
                break

    writer.add_scalar('train/batch_time', batch_time.avg, epoch)
    writer.add_scalar('train/acc@1', top1.avg, epoch)
    writer.add_scalar('train/acc@5', top5.avg, epoch)
    writer.add_scalar('train/loss', losses.avg, epoch)
    writer.add_scalar('train/cos_loss', cos_losses.avg, epoch)
    writer.add_scalar('train/smooth_loss', smooth_losses.avg, epoch)
    # writer.add_scalar('train/cos01', cos01_losses.avg, epoch)
    # writer.add_scalar('train/cos02', cos02_losses.avg, epoch)
    # writer.add_scalar('train/cos12', cos12_losses.avg, epoch)
    for index, combination in list_of_combination:
        writer.add_scalar('train/cos_{}'.format(combination),
                          cos_losses_array[index].avg, epoch)


def Naive_Trainer(args, loader: DataLoader, models, criterion, optimizer: Optimizer,
                  epoch: int, device: torch.device, writer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()

    for i in range(args.num_models):
        models[i].train()
        requires_grad_(models[i], True)

    for i, (inputs, targets) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)
        inputs.requires_grad = True
        grads = []
        loss_std = 0
        for j in range(args.num_models):
            logits = models[j](inputs)
            loss = criterion(logits, targets)
            grad = autograd.grad(loss, inputs, create_graph=True)[0]
            grad = grad.flatten(start_dim=1)
            grads.append(grad)
            loss_std += loss

        loss = loss_std

        ensemble = Ensemble(models)
        logits = ensemble(inputs)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
        losses.update(loss.item(), batch_size)
        top1.update(acc1.item(), batch_size)
        top5.update(acc5.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f}\t'
                  'Acc@1 {top1.avg:.3f}\t'
                  'Acc@5 {top5.avg:.3f}'.format(
                      epoch, i, len(loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1, top5=top5))

    writer.add_scalar('train/batch_time', batch_time.avg, epoch)
    writer.add_scalar('train/acc@1', top1.avg, epoch)
    writer.add_scalar('train/acc@5', top5.avg, epoch)
    writer.add_scalar('train/loss', losses.avg, epoch)


def GAL_Trainer(args, loader: DataLoader, models, criterion, optimizer: Optimizer,
                epoch: int, device: torch.device, writer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    gal_losses = AverageMeter()

    end = time.time()

    for i in range(args.num_models):
        models[i].train()
        requires_grad_(models[i], True)

    for i, (inputs, targets) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)
        inputs.requires_grad = True
        grads = []
        loss_std = 0
        for j in range(args.num_models):
            logits = models[j](inputs)
            loss = criterion(logits, targets)
            grad = autograd.grad(loss, inputs, create_graph=True)[0]
            grad = grad.flatten(start_dim=1)
            grads.append(grad)
            loss_std += loss

        cos_sim = []
        for ii in range(len(models)):
            for j in range(ii + 1, len(models)):
                cos_sim.append(F.cosine_similarity(
                    grads[ii], grads[j], dim=-1))
        cos_sim = torch.stack(cos_sim, dim=-1)
        gal_loss = torch.log(cos_sim.exp().sum(dim=-1) + 1e-20).mean()

        loss = loss_std + args.coeff * gal_loss

        ensemble = Ensemble(models)
        logits = ensemble(inputs)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
        losses.update(loss.item(), batch_size)
        top1.update(acc1.item(), batch_size)
        top5.update(acc5.item(), batch_size)
        gal_losses.update(gal_loss.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f}\t'
                  'Acc@1 {top1.avg:.3f}\t'
                  'Acc@5 {top5.avg:.3f}'.format(
                      epoch, i, len(loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1, top5=top5))

    writer.add_scalar('train/batch_time', batch_time.avg, epoch)
    writer.add_scalar('train/acc@1', top1.avg, epoch)
    writer.add_scalar('train/acc@5', top5.avg, epoch)
    writer.add_scalar('train/loss', losses.avg, epoch)
    writer.add_scalar('train/gal_loss', gal_losses.avg, epoch)


def ADP_Trainer(args, loader: DataLoader, models, criterion, optimizer: Optimizer,
                epoch: int, device: torch.device, writer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    entropy_losses = AverageMeter()
    det_losses = AverageMeter()
    end = time.time()

    for i in range(args.num_models):
        models[i].train()
        requires_grad_(models[i], True)

    for i, (inputs, targets) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)
        inputs.requires_grad = True

        num_classes = 10
        y_true = torch.zeros(inputs.size(0), num_classes).cuda()
        y_true.scatter_(1, targets.view(-1, 1), 1)

        loss_std = 0
        mask_non_y_pred = []
        ensemble_probs = 0

        for j in range(args.num_models):
            outputs = models[j](inputs)
            loss_std += criterion(outputs, targets)

            # for log_det
            y_pred = F.softmax(outputs, dim=-1)
            bool_R_y_true = torch.eq(torch.ones_like(y_true) - y_true,
                                     torch.ones_like(y_true))  # batch_size X (num_class X num_models), 2-D
            mask_non_y_pred.append(torch.masked_select(y_pred, bool_R_y_true).
                                   reshape(-1, num_classes - 1))  # batch_size X (num_class-1) X num_models, 1-D

            # for ensemble entropy
            ensemble_probs += y_pred

        ensemble_probs = ensemble_probs / len(models)
        ensemble_entropy = torch.sum(-torch.mul(ensemble_probs, torch.log(ensemble_probs + 1e-20)),
                                     dim=-1).mean()

        mask_non_y_pred = torch.stack(mask_non_y_pred, dim=1)
        assert mask_non_y_pred.shape == (
            inputs.size(0), len(models), num_classes - 1)
        mask_non_y_pred = mask_non_y_pred / torch.norm(mask_non_y_pred, p=2, dim=-1,
                                                       keepdim=True)  # batch_size X num_model X (num_class-1), 3-D
        matrix = torch.matmul(mask_non_y_pred,
                              mask_non_y_pred.permute(0, 2, 1))  # batch_size X num_model X num_model, 3-D
        log_det = torch.logdet(matrix + 1e-6 * torch.eye(len(models),
                               device=matrix.device).unsqueeze(0)).mean()  # batch_size X 1, 1-D

        loss = loss_std - args.alpha * ensemble_entropy - args.beta * log_det

        ensemble = Ensemble(models)
        logits = ensemble(inputs)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
        losses.update(loss.item(), batch_size)
        top1.update(acc1.item(), batch_size)
        top5.update(acc5.item(), batch_size)
        entropy_losses.update(ensemble_entropy.item(), batch_size)
        det_losses.update(log_det.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f}\t'
                  'Acc@1 {top1.avg:.3f}\t'
                  'Acc@5 {top5.avg:.3f}'.format(
                      epoch, i, len(loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1, top5=top5))

    writer.add_scalar('train/batch_time', batch_time.avg, epoch)
    writer.add_scalar('train/acc@1', top1.avg, epoch)
    writer.add_scalar('train/acc@5', top5.avg, epoch)
    writer.add_scalar('train/loss', losses.avg, epoch)
    writer.add_scalar('train/entropy_loss', entropy_losses.avg, epoch)
    writer.add_scalar('train/det_loss', det_losses.avg, epoch)


def DVERGE_Trainer(args, loader, models, criterion, optimizers,
                   epoch: int, distill_cfg, device: torch.device, writer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    for i in range(args.num_models):
        models[i].train()
        requires_grad_(models[i], True)

    losses = [AverageMeter() for i in range(args.num_models)]

    for i, (si, sl, ti, tl) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        si, sl, ti, tl = si.to(device), sl.to(
            device), ti.to(device), tl.to(device)
        batch_size = si.size(0)

        distilled_data_list = []
        for j in range(args.num_models):
            temp = Linf_distillation(models[j], si, ti, **distill_cfg)
            distilled_data_list.append(temp)

        for j in range(args.num_models):
            loss = 0
            for k, distilled_data in enumerate(distilled_data_list):
                if (j == k):
                    continue
                outputs = models[j](distilled_data)
                loss += criterion(outputs, sl)

            losses[j].update(loss.item(), batch_size)
            optimizers[j].zero_grad()
            loss.backward()
            optimizers[j].step()

        ensemble = Ensemble(models)
        logits = ensemble(si)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(logits, sl, topk=(1, 5))
        top1.update(acc1.item(), batch_size)
        top5.update(acc5.item(), batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f}\t'
                  'Acc@1 {top1.avg:.3f}\t'
                  'Acc@5 {top5.avg:.3f}'.format(
                      epoch, i, len(loader), batch_time=batch_time,
                      data_time=data_time, loss=losses[0], top1=top1, top5=top5))

    writer.add_scalar('train/batch_time', batch_time.avg, epoch)
    writer.add_scalar('train/acc@1', top1.avg, epoch)
    writer.add_scalar('train/acc@5', top5.avg, epoch)
    for i in range(args.num_models):
        writer.add_scalar('train/loss%d' % (i), losses[i].avg, epoch)
