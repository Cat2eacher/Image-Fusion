import os
import numpy as np
import torch
import torchvision
from tqdm import tqdm
from .utils import get_lr, AdaptiveWeights

# 这里信息保留度的计算要单独指定一下环境
# 写在 train_epoch 函数外是为了避免每次运行一次训练函数都重新初始化一次
# 后续优化可以直接封装进一个Trainer类中
adaptive_weights = AdaptiveWeights(device="cuda")


# ----------------------------------------------------#
#   训练
# ----------------------------------------------------#
def train_epoch(model, device, train_dataloader, criterion, optimizer, epoch, num_Epoches):
    model.train()
    train_epoch_loss = {"mse_loss": [],
                        "ssim_loss": [],
                        "total_loss": [],
                        }
    # 创建进度条
    pbar = tqdm(train_dataloader, total=len(train_dataloader))

    for batch_index, (over_patch, under_patch) in enumerate(pbar, start=1):
        # 清空梯度  reset gradient
        optimizer.zero_grad()
        # 将数据移至设备
        over_patch, under_patch = over_patch.to(device), under_patch.to(device)
        # 计算自适应权重
        weights_preserve = adaptive_weights.calculate(over_patch, under_patch).to(device)
        # print(f"{epoch}:{batch_index}-{weights_preserve}")
        # 前向传播
        outputs = model(over_patch, under_patch)
        outputs = (outputs + 1) / 2  # [-1,1]->[0,1]  归一化到[0,1]范围

        # 计算像素损失（MSE）
        pixel_loss_value = (
                weights_preserve[:, 0] * criterion["mse_loss"](outputs, over_patch) +
                weights_preserve[:, 1] * criterion["mse_loss"](outputs, under_patch)
        )
        pixel_loss_value = torch.mean(pixel_loss_value)

        # 计算结构相似度损失（SSIM）
        ssim_loss_value = (
                weights_preserve[:, 0] * (1 - criterion["ssim_loss"](outputs, over_patch, normalize=True)) +
                weights_preserve[:, 1] * (1 - criterion["ssim_loss"](outputs, under_patch, normalize=True))
        )
        ssim_loss_value = torch.mean(ssim_loss_value)
        # 总损失
        loss = pixel_loss_value + criterion["lambda"] * ssim_loss_value
        # 反向传播
        loss.backward()
        # 参数更新
        optimizer.step()

        # 记录损失值
        train_epoch_loss["mse_loss"].append(pixel_loss_value.item())
        train_epoch_loss["ssim_loss"].append(ssim_loss_value.item())
        train_epoch_loss["total_loss"].append(loss.item())

        # 更新进度条信息
        pbar.set_description(f'Epoch [{epoch + 1}/{num_Epoches}]')
        pbar.set_postfix(
            pixel_loss=f'{pixel_loss_value.item():.4f}',
            ssim_loss=f'{ssim_loss_value.item():.4f}',
            learning_rate=f'{get_lr(optimizer):.6f}'
        )
        # pbar.set_postfix(
        #     pixel_loss=pixel_loss_value.item(),
        #     ssim_loss=ssim_loss_value.item(),
        #     learning_rate=get_lr(optimizer),
        # )
        # pbar.set_postfix(**{'loss': loss.item(),
        #                     'lr': get_lr(optimizer),
        #                     })

    # 计算平均损失
    return {"mse_loss": np.mean(train_epoch_loss["mse_loss"]),
            "ssim_loss": np.mean(train_epoch_loss["ssim_loss"]),
            "total_loss": np.mean(train_epoch_loss["total_loss"]),
            }


# ----------------------------------------------------#
#   权重保存
# ----------------------------------------------------#
def checkpoint_save(epoch, model, optimizer, lr_scheduler, checkpoints_path, best_loss):
    # if not os.path.exists(checkpoints_path):
    #     os.mkdir(checkpoints_path)
    # 确保保存目录存在
    os.makedirs(checkpoints_path, exist_ok=True)
    # 构建检查点数据
    checkpoints = {'epoch': epoch,
                   'model': model.state_dict(),
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
def tensorboard_log(writer, model, train_loss, test_image, device, epoch):
    with torch.no_grad():
        # 记录损失值
        writer.add_scalar('pixel_loss', train_loss["mse_loss"].item(), global_step=epoch)
        writer.add_scalar('ssim_loss', train_loss["ssim_loss"].item(), global_step=epoch)
        writer.add_scalar('total_loss', train_loss["total_loss"].item(), global_step=epoch)

        # 处理测试图像
        test_over, test_under = test_image
        test_over = test_over.to(device)
        test_under = test_under.to(device)
        # 生成融合图像
        fused_img = model(test_over, test_under)
        fused_img = (fused_img + 1) / 2  # 归一化到[0,1]范围

        # 创建图像网格并记录
        grid_config = {'normalize': True, 'nrow': 4}
        img_grid_over = torchvision.utils.make_grid(test_over,  **grid_config)
        img_grid_under = torchvision.utils.make_grid(test_under,  **grid_config)
        img_grid_fuse = torchvision.utils.make_grid(fused_img,  **grid_config)
        writer.add_image('test_over_patch', img_grid_over, global_step=1, dataformats='CHW')
        writer.add_image('test_under_patch', img_grid_under, global_step=1, dataformats='CHW')
        writer.add_image('fused_img', img_grid_fuse, global_step=epoch, dataformats='CHW')
