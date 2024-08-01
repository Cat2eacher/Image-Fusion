import os
import numpy as np
import torch
import torchvision
from tqdm import tqdm
from .util import get_lr, clamp


# ----------------------------------------------------#
#   训练 分类模型
# ----------------------------------------------------#
def train_epoch_cls(model, device, train_dataloader, criterion, optimizer, epoch, num_Epoches):
    model.train()
    train_epoch_loss = []
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    # 白天one-hot label[1,0] ,夜晚label[0,1]
    for index, (images, labels) in enumerate(pbar, start=1):
        images = images.to(device)
        labels = labels.to(device)
        # 清空梯度  reset gradient
        optimizer.zero_grad()
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        # 参数更新
        optimizer.step()

        train_epoch_loss.append(loss.item())
        pbar.set_description(f'Epoch [{epoch + 1}/{num_Epoches}]')

        pbar.set_postfix(
            loss_total=loss.item(),
            learning_rate=get_lr(optimizer),
        )

    return np.average(train_epoch_loss)


# ----------------------------------------------------#
#   验证 分类模型
# ----------------------------------------------------#
def valid_epoch_cls(model, device, valid_dataloader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    pbar = tqdm(valid_dataloader, total=len(valid_dataloader))
    with torch.no_grad():
        for index, (images, labels) in enumerate(pbar, start=1):
            images = images.to(device)
            labels = labels.to(device)
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            # 获取最大概率对应的类别索引，pred是预测结果
            # predicts = outputs.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            _, predicts = torch.max(outputs, dim=1)
            # 将预测结果与真实标签比较，eq返回布尔值矩阵，然后对CPU上的元素求和得到该batch中正确的预测数
            # correct += predicts.eq(labels.data.view_as(pred)).cpu().sum()
            correct += (predicts == labels).sum()

        total_loss /= len(valid_dataloader.dataset)
        prec = correct / float(len(valid_dataloader.dataset))

        print('\nValid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            total_loss, correct, len(valid_dataloader.dataset), 100. * prec))
    return total_loss, prec


# ----------------------------------------------------#
#   第二阶段训练
# ----------------------------------------------------#
def train_epoch_fusion(cls_model, fusion_model, device, train_dataloader, criterion, optimizer, epoch, num_Epoches):
    cls_model.eval()
    fusion_model.train()
    train_epoch_loss = {"illum_loss": [],
                        "aux_loss": [],
                        "texture_loss": [],
                        "total_loss": [],
                        }
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    for batch_index, (vis_image, vis_y_image, _, _, inf_image, _) in enumerate(pbar, start=1):
        vis_y_image = vis_y_image.to(device)
        vis_image = vis_image.to(device)
        inf_image = inf_image.to(device)
        # 清空梯度  reset gradient
        optimizer.zero_grad()
        # 前向传播
        fused_image = fusion_model(vis_y_image, inf_image)
        # 强制约束范围在[0,1], 以免越界值使得最终生成的图像产生异常斑点
        fused_image = clamp(fused_image)

        # 使用预训练的分类模型，得到可见光图片属于白天还是夜晚的概率
        cls_preds = cls_model(vis_image)
        illum_loss_value = criterion["illum_loss"](cls_preds, vis_y_image, inf_image, fused_image)
        aux_loss_value = criterion["aux_loss"](vis_y_image, inf_image, fused_image)
        texture_loss_value = criterion["texture_loss"](vis_y_image, inf_image, fused_image, device)
        lambda1, lambda2, lambda3 = criterion["lambda"][0], criterion["lambda"][1], criterion["lambda"][2]
        loss = lambda1 * illum_loss_value + lambda2 * aux_loss_value + lambda3 * texture_loss_value
        # 反向传播
        loss.backward()
        # 参数更新
        optimizer.step()

        train_epoch_loss["illum_loss"].append(lambda1 * illum_loss_value.item())
        train_epoch_loss["aux_loss"].append(lambda2 * aux_loss_value.item())
        train_epoch_loss["texture_loss"].append(lambda3 * texture_loss_value.item())
        train_epoch_loss["total_loss"].append(loss.item())

        pbar.set_description(f'Epoch [{epoch + 1}/{num_Epoches}]')

        pbar.set_postfix(
            illum_loss=lambda1 * illum_loss_value.item(),
            aux_loss=lambda2 * aux_loss_value.item(),
            texture_loss=lambda3 * texture_loss_value.item(),
            total_loss=loss.item(),
            learning_rate=get_lr(optimizer),
        )

    return {"illum_loss": np.average(train_epoch_loss["illum_loss"]),
            "aux_loss": np.average(train_epoch_loss["aux_loss"]),
            "texture_loss": np.average(train_epoch_loss["texture_loss"]),
            "total_loss": np.average(train_epoch_loss["total_loss"]),
            }


# ----------------------------------------------------#
#   权重保存
# ----------------------------------------------------#
def checkpoint_save_cls(epoch, model, checkpoints_path, best_prec):
    if not os.path.exists(checkpoints_path):
        os.mkdir(checkpoints_path)
    # torch.save(model.state_dict(), f'{checkpoints_path}/best_cls.pth')
    checkpoints = {'epoch': epoch,
                   'model': model.state_dict(),
                   }
    save_name = '/epoch%03d-prec%.3f.pth' % (epoch, best_prec)
    save_path = checkpoints_path + save_name
    torch.save(checkpoints, save_path)


def checkpoint_save_fusion(epoch, model, checkpoints_path, best_loss):
    if not os.path.exists(checkpoints_path):
        os.mkdir(checkpoints_path)
    checkpoints = {'epoch': epoch,
                   'model': model.state_dict(),
                   # 'best_loss': best_loss,
                   }
    save_name = '/epoch%03d-loss%.3f.pth' % (epoch, best_loss)
    save_path = checkpoints_path + save_name
    # torch.save(model.state_dict(), f'{save_path}/fusion_model_epoch_{epoch}.pth')
    torch.save(checkpoints, save_path)


# ----------------------------------------------------#
#   tensorboard
# ----------------------------------------------------#
def tensorboard_load(writer, model, train_loss, test_image, device, epoch):
    with torch.no_grad():
        writer.add_scalar('illum_loss', train_loss["illum_loss"].item(), global_step=epoch)
        writer.add_scalar('aux_loss', train_loss["aux_loss"].item(), global_step=epoch)
        writer.add_scalar('texture_loss', train_loss["texture_loss"].item(), global_step=epoch)
        writer.add_scalar('total_loss', train_loss["total_loss"].item(), global_step=epoch)

        vis_image, vis_y_image, _, _, inf_image, _ = test_image
        vis_y_image = vis_y_image.to(device)
        vis_image = vis_image.to(device)
        inf_image = inf_image.to(device)
        # 前向传播
        fused_image = model(vis_y_image, inf_image)
        # 强制约束范围在[0,1], 以免越界值使得最终生成的图像产生异常斑点
        fused_image = clamp(fused_image)
        img_grid_vis = torchvision.utils.make_grid(vis_image, normalize=True, nrow=4)
        img_grid_inf = torchvision.utils.make_grid(inf_image, normalize=True, nrow=4)
        img_grid_fuse = torchvision.utils.make_grid(fused_image, normalize=True, nrow=4)
        writer.add_image('test_vis', img_grid_vis, global_step=1, dataformats='CHW')
        writer.add_image('test_inf', img_grid_inf, global_step=1, dataformats='CHW')
        writer.add_image('fused_img', img_grid_fuse, global_step=epoch, dataformats='CHW')
