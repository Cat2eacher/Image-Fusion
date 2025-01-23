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
    model.encoder.train()
    model.decoder_train.train()
    model.decoder_eval.eval()

    train_epoch_loss = {"mse_loss": [],
                        "ssim_loss": [],
                        "total_loss": [],
                        }
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    for index, image_batch in enumerate(pbar, start=1):
        # 清空梯度  reset gradient
        optimizer.zero_grad()
        # 载入批量图像
        inputs = image_batch.to(device)
        # 复制图像作为标签
        labels = image_batch.data.clone().to(device)
        # 前向传播
        if deepsupervision:
            feature_encoded = model.encoder(inputs)
            outputs = model.decoder_train(feature_encoded)
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
            feature_encoded = model.encoder(inputs)
            outputs = model.decoder_train(feature_encoded)[0]
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
        # pbar.set_postfix(loss=loss.item(), train_acc)
        pbar.set_postfix(
            pixel_loss=pixel_loss_value.item(),
            ssim_loss=ssim_loss_value.item(),
            learning_rate=get_lr(optimizer),
        )
        # pbar.set_postfix(**{'loss': loss.item(),
        #                     'lr': get_lr(optimizer),
        #                     })

    return {"mse_loss": np.average(train_epoch_loss["mse_loss"]),
            "ssim_loss": np.average(train_epoch_loss["ssim_loss"]),
            "total_loss": np.average(train_epoch_loss["total_loss"]),
            }


# ----------------------------------------------------#
#   第二阶段训练
# ----------------------------------------------------#
def train_epoch_rfn(model, device, train_dataloader, criterion, optimizer, epoch, num_Epoches):
    model["nest_model"].eval()
    model["fusion_model"].train()
    train_epoch_loss = {"detail_loss": [],
                        "feature_loss": [],
                        "total_loss": [],
                        }
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    for batch_index, (vi_batch, ir_batch) in enumerate(pbar, start=1):
        # 清空梯度  reset gradient
        optimizer.zero_grad()
        # 载入批量图像
        vi_batch, ir_batch = vi_batch.to(device), ir_batch.to(device)
        # 前向传播
        # encoder
        en_vi = model["nest_model"].encoder(vi_batch)
        en_ir = model["nest_model"].encoder(ir_batch)
        # fusion
        f = model["fusion_model"](en_vi, en_ir)
        # decoder
        outputs = model["nest_model"].decoder_train(f)
        detail_loss_value = criterion["detail_loss"](outputs, vi_batch)
        feature_loss_value = criterion["feature_loss"](f, en_vi, en_ir)
        loss = feature_loss_value + criterion["alpha"] * detail_loss_value
        # 反向传播
        loss.backward()
        # 参数更新
        optimizer.step()

        train_epoch_loss["detail_loss"].append(detail_loss_value.item())
        train_epoch_loss["feature_loss"].append(feature_loss_value.item())
        train_epoch_loss["total_loss"].append(loss.item())

        pbar.set_description(f'Epoch [{epoch + 1}/{num_Epoches}]')
        # pbar.set_postfix(loss=loss.item(), train_acc)
        pbar.set_postfix(
            detail_loss=detail_loss_value.item(),
            feature_loss=feature_loss_value.item(),
            learning_rate=get_lr(optimizer),
        )
        # pbar.set_postfix(**{'loss': loss.item(),
        #                     'lr': get_lr(optimizer),
        #                     })

    return {"detail_loss": np.average(train_epoch_loss["detail_loss"]),
            "feature_loss": np.average(train_epoch_loss["feature_loss"]),
            "total_loss": np.average(train_epoch_loss["total_loss"]),
            }


# ----------------------------------------------------#
#   权重保存
# ----------------------------------------------------#
def checkpoint_save(epoch, model, optimizer, lr_scheduler, checkpoints_path, best_loss):
    if not os.path.exists(checkpoints_path):
        os.mkdir(checkpoints_path)
    checkpoints = {'epoch': epoch,
                   # 'model': model.state_dict(),
                   'encoder': model.encoder.state_dict(),
                   'decoder': model.decoder_train.state_dict(),
                   'optimizer': optimizer.state_dict(),
                   # 'lr': lr_scheduler.state_dict(),
                   # 'best_loss': best_loss,
                   }
    checkpoints_name = '/epoch%03d-loss%.3f.pth' % (epoch, best_loss)
    save_path = checkpoints_path + checkpoints_name
    torch.save(checkpoints, save_path)


def checkpoint_save_rfn(epoch, model, optimizer, lr_scheduler, checkpoints_path, best_loss):
    if not os.path.exists(checkpoints_path):
        os.mkdir(checkpoints_path)
    checkpoints = {'epoch': epoch,
                   'model': model.state_dict(),
                   'optimizer': optimizer.state_dict(),
                   # 'lr': lr_scheduler.state_dict(),
                   # 'best_loss': best_loss,
                   }
    checkpoints_name = '/epoch%03d-loss%.3f.pth' % (epoch, best_loss)
    save_path = checkpoints_path + checkpoints_name
    torch.save(checkpoints, save_path)


# ----------------------------------------------------#
#   tensorboard
# ----------------------------------------------------#
def tensorboard_load(writer, model, train_loss, test_image, epoch, deepsupervision):
    with torch.no_grad():
        writer.add_scalar('pixel_loss', train_loss["mse_loss"].item(), global_step=epoch)
        writer.add_scalar('ssim_loss', train_loss["ssim_loss"].item(), global_step=epoch)
        writer.add_scalar('total_loss', train_loss["total_loss"].item(), global_step=epoch)
        if deepsupervision:
            feature_encoded = model.encoder(test_image)
            rebuild_img = model.decoder_train(feature_encoded)
            img_grid_real = torchvision.utils.make_grid(test_image, normalize=True, nrow=4)
            img_grid_rebuild_1 = torchvision.utils.make_grid(rebuild_img[0], normalize=True, nrow=4)
            img_grid_rebuild_2 = torchvision.utils.make_grid(rebuild_img[1], normalize=True, nrow=4)
            img_grid_rebuild_3 = torchvision.utils.make_grid(rebuild_img[2], normalize=True, nrow=4)

            writer.add_image('Real image', img_grid_real, global_step=1)
            writer.add_image('Rebuild image_1', img_grid_rebuild_1, global_step=epoch)
            writer.add_image('Rebuild image_2', img_grid_rebuild_2, global_step=epoch)
            writer.add_image('Rebuild image_3', img_grid_rebuild_3, global_step=epoch)

        else:
            feature_encoded = model.encoder(test_image)
            rebuild_img = model.decoder_train(feature_encoded)
            img_grid_real = torchvision.utils.make_grid(test_image, normalize=True, nrow=4)
            img_grid_rebuild = torchvision.utils.make_grid(rebuild_img[0], normalize=True, nrow=4)
            writer.add_image('Real image', img_grid_real, global_step=1)
            writer.add_image('Rebuild image', img_grid_rebuild, global_step=epoch)


def tensorboard_load_rfn(writer, model, train_loss, test_image, device, epoch, deepsupervision):
    with torch.no_grad():
        writer.add_scalar('detail_loss', train_loss["detail_loss"].item(), global_step=epoch)
        writer.add_scalar('feature_loss', train_loss["feature_loss"].item(), global_step=epoch)
        writer.add_scalar('total_loss', train_loss["total_loss"].item(), global_step=epoch)

        test_vi, test_ir = test_image
        test_vi, test_ir = test_vi.to(device), test_ir.to(device)
        # 前向传播
        # encoder
        en_vi = model["nest_model"].encoder(test_vi)
        en_ir = model["nest_model"].encoder(test_ir)
        # fusion
        f = model["fusion_model"](en_vi, en_ir)
        # decoder
        fused_img = model["nest_model"].decoder_train(f)

        if deepsupervision:
            img_grid_vi = torchvision.utils.make_grid(test_vi, normalize=True, nrow=4)
            img_grid_ir = torchvision.utils.make_grid(test_ir, normalize=True, nrow=4)
            img_grid_fuse_1 = torchvision.utils.make_grid(fused_img[0], normalize=True, nrow=4)
            img_grid_fuse_2 = torchvision.utils.make_grid(fused_img[1], normalize=True, nrow=4)
            img_grid_fuse_3 = torchvision.utils.make_grid(fused_img[2], normalize=True, nrow=4)

            writer.add_image('test_vi', img_grid_vi, global_step=1, dataformats='CHW')
            writer.add_image('test_ir', img_grid_ir, global_step=1, dataformats='CHW')
            writer.add_image('fused_img_1', img_grid_fuse_1, global_step=epoch, dataformats='CHW')
            writer.add_image('fused_img_2', img_grid_fuse_2, global_step=epoch, dataformats='CHW')
            writer.add_image('fused_img_3', img_grid_fuse_3, global_step=epoch, dataformats='CHW')

        else:
            img_grid_vi = torchvision.utils.make_grid(test_vi, normalize=True, nrow=4)
            img_grid_ir = torchvision.utils.make_grid(test_ir, normalize=True, nrow=4)
            img_grid_fuse = torchvision.utils.make_grid(fused_img[0], normalize=True, nrow=4)
            writer.add_image('test_vi', img_grid_vi, global_step=1, dataformats='CHW')
            writer.add_image('test_ir', img_grid_ir, global_step=1, dataformats='CHW')
            writer.add_image('fused_img', img_grid_fuse, global_step=epoch, dataformats='CHW')
