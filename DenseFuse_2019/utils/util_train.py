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
    """训练一个epoch
    Args:
        model: 待训练的模型
        device: 计算设备
        train_dataloader: 训练数据加载器
        criterion: 损失函数字典，包含'mse_loss'和'ssim_loss'
        optimizer: 优化器
        epoch: 当前epoch
        num_epochs: 总epoch数
    Returns:
        包含平均损失值的字典
    """
    model.train()
    train_epoch_loss = {"mse_loss": [],
                        "ssim_loss": [],
                        "total_loss": [],
                        }

    # 创建进度条
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    for batch_idx, image_batch in enumerate(pbar, start=1):
        # 清空梯度  reset gradient
        optimizer.zero_grad()
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
        # 反向传播
        loss.backward()
        # 参数更新
        optimizer.step()
        # 记录损失值
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
    """验证模型性能
    Args:
        model: 待验证的模型
        device: 计算设备
        valid_dataloader: 验证数据加载器
        criterion: 损失函数字典
    Returns:
        平均验证损失
    """
    model.eval()
    valid_epoch_loss = []

    with torch.no_grad():
        pbar = tqdm(valid_dataloader, desc='Validation')
        # for index, (inputs, targets) in enumerate(train_dataloader, start=vis):
        for index, image_batch in enumerate(pbar, start=1):
            inputs = image_batch.to(device)
            labels = image_batch.data.clone().to(device)
            outputs = model(inputs)

            # 计算损失
            pixel_loss_value = criterion["mse_loss"](outputs, labels)
            ssim_loss_value = 1 - criterion["ssim_loss"](outputs, labels, normalize=True)
            loss = pixel_loss_value + criterion["lambda"] * ssim_loss_value

            valid_epoch_loss.append(loss.item())

            pbar.set_postfix({
                'pixel_loss': f'{pixel_loss_value.item():.4f}',
                'ssim_loss': f'{ssim_loss_value.item():.4f}'
            })
    return np.mean(valid_epoch_loss)


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
def tensorboard_log(writer, model, train_loss, test_image, epoch):
    """记录训练信息到TensorBoard
    Args:
        writer: TensorBoard写入器
        model: 模型
        train_loss: 训练损失字典
        test_image: 测试图像
        epoch: 当前epoch
    """
    with torch.no_grad():
        # 记录损失值
        for loss_name, loss_value in train_loss.items():
            writer.add_scalar(loss_name, loss_value, global_step=epoch)
        # writer.add_scalar('pixel_loss', train_loss["mse_loss"].item(), global_step=epoch)
        # writer.add_scalar('ssim_loss', train_loss["ssim_loss"].item(), global_step=epoch)
        # writer.add_scalar('total_loss', train_loss["total_loss"].item(), global_step=epoch)

        # 生成重建图像
        rebuild_img = model(test_image)
        # 创建图像网格
        img_grid_real = torchvision.utils.make_grid(test_image, normalize=True, nrow=4)
        img_grid_rebuild = torchvision.utils.make_grid(rebuild_img, normalize=True, nrow=4)
        # 记录图像
        writer.add_image('Real image', img_grid_real, global_step=1)
        writer.add_image('Rebuild image', img_grid_rebuild, global_step=epoch)