import os
import cv2 as cv
import numpy as np
import torch
import torchvision
from tqdm import tqdm
from .utils import get_lr


# ----------------------------------------------------#
#   训练
# ----------------------------------------------------#
def train_epoch(model, device, train_dataloader, criterion, optimizer, epoch, num_Epoches):
    model.train()
    train_epoch_loss = []
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    for index, (under_patch, over_patch) in enumerate(pbar, start=1):
        # 清空梯度  reset gradient
        optimizer.zero_grad()
        # 载入批量图像
        under_patch, over_patch = under_patch.to(device), over_patch.to(device)
        under_patch_lum = under_patch[:, 0:1]
        over_patch_lum = over_patch[:, 0:1]
        # 前向传播
        fusion_outputs = model(under_patch_lum, over_patch_lum)
        # 计算损失
        loss, y_hat = criterion(y_1=under_patch_lum, y_2=over_patch_lum, y_f=fusion_outputs)
        # 反向传播
        loss.backward()
        # 参数更新
        optimizer.step()

        train_epoch_loss.append(loss.item())

        pbar.set_description(f'Epoch [{epoch + 1}/{num_Epoches}]')
        pbar.set_postfix(
            loss=loss.item(),
            learning_rate=get_lr(optimizer),
        )
        # pbar.set_postfix(**{'loss': loss.item(),
        #                     'lr': get_lr(optimizer),
        #                     })

    return np.average(train_epoch_loss)


# ----------------------------------------------------#
#   验证
# ----------------------------------------------------#


# ----------------------------------------------------#
#   权重保存
# ----------------------------------------------------#
def checkpoint_save(epoch, model, optimizer, lr_scheduler, checkpoints_path, best_loss):
    if not os.path.exists(checkpoints_path):
        os.mkdir(checkpoints_path)
    checkpoints = {'epoch': epoch,
                   'model': model.state_dict(),
                   'optimizer': optimizer.state_dict(),
                   'lr': lr_scheduler.state_dict(),
                   'best_loss': best_loss,
                   }
    checkpoints_name = '/epoch%03d-loss%.3f.pth' % (epoch, best_loss)
    save_path = checkpoints_path + checkpoints_name
    torch.save(checkpoints, save_path)


# ----------------------------------------------------#
#   tensorboard
# ----------------------------------------------------#
def tensorboard_load(writer, model, train_loss, test_image, device, epoch):
    def weightedFusion(Cr1, Cr2, Cb1, Cb2):
        L1_NORM = lambda b: torch.sum(torch.abs(b))
        tau = 128
        # Fuse Cr channel
        cr_up = (Cr1 * L1_NORM(Cr1 - tau) + Cr2 * L1_NORM(Cr2 - tau))
        cr_down = L1_NORM(Cr1 - tau) + L1_NORM(Cr2 - tau)
        cr_fuse = cr_up / cr_down

        # Fuse Cb channel
        cb_up = (Cb1 * L1_NORM(Cb1 - tau) + Cb2 * L1_NORM(Cb2 - tau))
        cb_down = L1_NORM(Cb1 - tau) + L1_NORM(Cb2 - tau)
        cb_fuse = cb_up / cb_down
        return cr_fuse, cb_fuse
    with torch.no_grad():
        writer.add_scalar('loss', train_loss.item(), global_step=epoch)
        test_under_patch, test_over_patch= test_image
        test_under_patch, test_over_patch = test_under_patch.to(device), test_over_patch.to(device)

        under_patch_lum = test_under_patch[:, 0:1]  # [B,1,H,W]
        under_patch_Cr = test_under_patch[:, 1:2]  # [B,1,H,W]
        under_patch_Cb = test_under_patch[:, 2:3]  # [B,1,H,W]
        over_patch_lum = test_over_patch[:, 0:1]  # [B,1,H,W]
        over_patch_Cr = test_over_patch[:, 1:2]  # [B,1,H,W]
        over_patch_Cb = test_over_patch[:, 2:3]  # [B,1,H,W]
        # 对lum通道进行DeepFuse融合
        fused_img_lum = model(under_patch_lum, over_patch_lum)
        # 对Cb和Cr通道进行加权融合
        fused_img_cr, fused_img_cb = weightedFusion(under_patch_Cr, over_patch_Cr, under_patch_Cb, over_patch_Cb)  # [B,1,H,W]
        fused_img = torch.cat((fused_img_lum.cpu(), fused_img_cr.cpu(), fused_img_cb.cpu()), dim=1)
        # grid
        img_grid_under = torchvision.utils.make_grid(test_under_patch, normalize=True, nrow=2)
        img_grid_over = torchvision.utils.make_grid(test_over_patch, normalize=True, nrow=2)
        img_grid_fuse = torchvision.utils.make_grid(fused_img, normalize=True, nrow=2)

        writer.add_image('test_under_patch', img_grid_under, global_step=1)
        writer.add_image('test_over_patch', img_grid_over, global_step=1)
        writer.add_image('fused_img', img_grid_fuse, global_step=epoch)