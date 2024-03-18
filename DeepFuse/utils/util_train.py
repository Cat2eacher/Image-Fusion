import os
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
def checkpoint(epoch, model, optimizer, lr_scheduler, checkpoints_path, best_loss):
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
    with torch.no_grad():
        writer.add_scalar('loss', train_loss.item(), global_step=epoch)
        test_under_patch, test_over_patch = test_image
        test_under_patch, test_over_patch = test_under_patch.to(device), test_over_patch.to(device)
        fused_img = model(test_under_patch, test_over_patch)
        img_grid_under = torchvision.utils.make_grid(test_under_patch, normalize=True, nrow=4)
        img_grid_over = torchvision.utils.make_grid(test_over_patch, normalize=True, nrow=4)
        img_grid_fuse = torchvision.utils.make_grid(fused_img, normalize=True, nrow=4)
        writer.add_image('test_under_patch', img_grid_under, global_step=1)
        writer.add_image('test_over_patch', img_grid_over, global_step=1)
        writer.add_image('fused_img', img_grid_fuse, global_step=epoch)
