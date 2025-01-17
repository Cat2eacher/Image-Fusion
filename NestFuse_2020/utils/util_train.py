import os
import numpy as np
import torch
import torchvision
from tqdm import tqdm
from .utils import get_lr


# ----------------------------------------------------#
#   训练
# ----------------------------------------------------#
def train_epoch(model, device, train_dataloader, criterion, optimizer, epoch, num_Epoches, deepsupervision):
    model.train()
    train_epoch_loss = {"mse_loss": [],
                        "ssim_loss": [],
                        "total_loss": [],
                        }
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    for batch_idx, image_batch in enumerate(pbar, start=1):
        # 清空梯度  reset gradient
        optimizer.zero_grad()
        # 载入批量图像
        inputs = image_batch.to(device)
        # 复制图像作为标签
        labels = image_batch.data.clone().to(device)
        # 前向传播
        if deepsupervision:
            outputs = model(inputs)
            ssim_loss_value = 0.
            pixel_loss_value = 0.
            for output in outputs:
                pixel_loss_temp = criterion["mse_loss"](output, labels)
                ssim_loss_temp = 1 - criterion["ssim_loss"](output, labels, normalize=True)
                pixel_loss_value += pixel_loss_temp
                ssim_loss_value += ssim_loss_temp
            ssim_loss_value /= len(outputs)
            pixel_loss_value /= len(outputs)
            loss = pixel_loss_value + criterion["lambda"] * ssim_loss_value
        else:
            outputs = model(inputs)[0]
            # 计算损失
            pixel_loss_value = criterion["mse_loss"](outputs, labels)
            ssim_loss_value = 1 - criterion["ssim_loss"](outputs, labels, normalize=True)
            loss = pixel_loss_value + criterion["lambda"] * ssim_loss_value
        # 反向传播
        loss.backward()
        # 参数更新
        optimizer.step()

        train_epoch_loss["mse_loss"].append(pixel_loss_value.item())
        train_epoch_loss["ssim_loss"].append(ssim_loss_value.item())
        train_epoch_loss["total_loss"].append(loss.item())

        pbar.set_description(f'Epoch [{epoch + 1}/{num_Epoches}]')
        # pbar.set_postfix(
        #     pixel_loss=pixel_loss_value.item(),
        #     ssim_loss=ssim_loss_value.item(),
        #     learning_rate=get_lr(optimizer),
        # )
        # 更新进度条信息
        pbar.set_postfix({
            'pixel_loss': f'{pixel_loss_value.item():.4f}',
            'ssim_loss': f'{ssim_loss_value.item():.4f}',
            'lr': f'{get_lr(optimizer):.6f}'
        })

    return {"mse_loss": np.mean(train_epoch_loss["mse_loss"]),
            "ssim_loss": np.mean(train_epoch_loss["ssim_loss"]),
            "total_loss": np.mean(train_epoch_loss["total_loss"]),
            }


# ----------------------------------------------------#
#   验证
# ----------------------------------------------------#
def valid_epoch(model, device, valid_dataloader, criterion):
    model.eval()
    valid_epoch_loss = []
    # valid_epoch_accuracy = []
    pbar = tqdm(valid_dataloader, total=len(valid_dataloader))
    # for index, (inputs, targets) in enumerate(train_dataloader, start=1):
    for index, image_batch in enumerate(pbar, start=1):
        # 载入批量图像
        inputs = image_batch.to(device)
        # 复制图像作为标签
        labels = image_batch.data.clone().to(device)
        # 前向传播
        outputs = model(inputs)
        # 计算损失
        pixel_loss_value = criterion["mse_loss"](outputs, labels)
        ssim_loss_value = 1 - criterion["ssim_loss"](outputs, labels, normalize=True)
        loss = pixel_loss_value + criterion["lambda"] * ssim_loss_value
        valid_epoch_loss.append(loss.item())

        pbar.set_description('valid')
        pbar.set_postfix(
            pixel_loss=pixel_loss_value.item(),
            ssim_loss=ssim_loss_value.item(),
        )
    return np.average(valid_epoch_loss)


# ----------------------------------------------------#
#   权重保存
# ----------------------------------------------------#
def checkpoint_save(epoch, model, optimizer, lr_scheduler, checkpoints_path, best_loss):
    if not os.path.exists(checkpoints_path):
        os.mkdir(checkpoints_path)
    checkpoints = {'epoch': epoch,
                   'model': model.state_dict(),
                   'encoder_state_dict': model.encoder.state_dict(),
                   'decoder_state_dict': model.decoder.state_dict(),
                   'optimizer': optimizer.state_dict(),
                   'lr': lr_scheduler.state_dict(),
                   'best_loss': best_loss,
                   }
    checkpoints_name = f'epoch{epoch:03d}-loss{best_loss:.3f}.pth'
    save_path = os.path.join(checkpoints_path, checkpoints_name)
    torch.save(checkpoints, save_path)


# ----------------------------------------------------#
#   tensorboard
# ----------------------------------------------------#
def tensorboard_log(writer, model, train_loss, test_image, epoch, deepsupervision):
    with torch.no_grad():
        # 记录损失值
        for loss_name, loss_value in train_loss.items():
            writer.add_scalar(loss_name, loss_value, global_step=epoch)
        # writer.add_scalar('pixel_loss', train_loss["mse_loss"].item(), global_step=epoch)
        # writer.add_scalar('ssim_loss', train_loss["ssim_loss"].item(), global_step=epoch)
        # writer.add_scalar('total_loss', train_loss["total_loss"].item(), global_step=epoch)
        if deepsupervision:
            rebuild_img = model(test_image)
            img_grid_real = torchvision.utils.make_grid(test_image, normalize=True, nrow=4)
            img_grid_rebuild_1 = torchvision.utils.make_grid(rebuild_img[0], normalize=True, nrow=4)
            img_grid_rebuild_2 = torchvision.utils.make_grid(rebuild_img[1], normalize=True, nrow=4)
            img_grid_rebuild_3 = torchvision.utils.make_grid(rebuild_img[2], normalize=True, nrow=4)

            writer.add_image('Real image', img_grid_real, global_step=1)
            writer.add_image('Rebuild image_1', img_grid_rebuild_1, global_step=epoch)
            writer.add_image('Rebuild image_2', img_grid_rebuild_2, global_step=epoch)
            writer.add_image('Rebuild image_3', img_grid_rebuild_3, global_step=epoch)

        else:
            # 生成重建图像
            rebuild_img = model(test_image)
            # 创建图像网格
            img_grid_real = torchvision.utils.make_grid(test_image, normalize=True, nrow=4)
            img_grid_rebuild = torchvision.utils.make_grid(rebuild_img, normalize=True, nrow=4)
            # 记录图像
            writer.add_image('Real image', img_grid_real, global_step=1)
            writer.add_image('Rebuild image', img_grid_rebuild, global_step=epoch)
