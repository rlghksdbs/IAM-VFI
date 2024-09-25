import os
import sys
sys.path.append('.')
sys.path.append('..')
import math
import time
import torch
import numpy as np
import random

from argument import add_argparse
from model.RIFE import Model
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb    
from tqdm import tqdm

exp = os.path.abspath('.').split('/')[-1]

def get_learning_rate(step, epoch, step_per_epoch, init_lr=0., min_lr=0., warm_epoch=5):
    warmup_duration = step_per_epoch * warm_epoch
    if step < warmup_duration:
        mul = step / warmup_duration
        return init_lr * mul
    else:
        mul = np.cos((step - warmup_duration) / (epoch * step_per_epoch - warmup_duration) * math.pi) * 0.5 + 0.5
        return (init_lr - min_lr) * mul + min_lr

def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())

    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)

def train(model, args, arbitrary=False):
    writer = SummaryWriter(args.log_path + '/train_tensorboard')
    writer_val = SummaryWriter(args.log_path + '/val_tensorboard')

    step = 0
    nr_eval = 0
    best_psnr = 0.0    
    
    ### dataloader setting
    from dataloader import DataLoader as customDataLoader
    dataset = customDataLoader('train', args, arbitrary=arbitrary)
    dataset_val = customDataLoader('validation', args, arbitrary=arbitrary)
    train_data = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, shuffle=True, drop_last=True)
    val_data = DataLoader(dataset_val, batch_size=args.val_batch_size, pin_memory=True, num_workers=8)

    args.step_per_epoch = train_data.__len__()
        
    ### scheduler setting
    scheduler = get_learning_rate
    print('training...')

    time_stamp = time.time()
    print_freq = args.step_per_epoch / args.print_freq

    if args.wandb_name is not None and args.wandb_model:
        wandb.watch(model.flownet, log="all", log_freq=args.step_per_epoch, idx=None, log_graph=False)

    if args.classifier_intra_v2:
        model.load_class_module(args.easy_pkl_model_path, args.medium_pkl_model_path, args.hard_pkl_model_path)
    
    start_epoch = 0
    
    for epoch in range(start_epoch, args.epoch):
        for i, data in enumerate(tqdm(train_data, desc='training')):            
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            
            if arbitrary == True:
                data_gpu, timestep = data
                timestep = timestep.to(device, non_blocking=True)
            else:
                data_gpu = data
                timestep = 0

            data_gpu = data_gpu.to(device, non_blocking=True) / 255.
            imgs = data_gpu[:, :6]
            gt = data_gpu[:, 6:9]

            learning_rate = scheduler(step, args.epoch, args.step_per_epoch, args.init_lr, args.min_lr, args.warm_epoch)            
            
            if args.classifier_intra_v2:
                pred, infos = model.update_classifier_intra_v2(imgs, gt, learning_rate, training=True, timestep=timestep)
            else:
                pred, infos = model.update(imgs, gt, learning_rate, training=True, timestep=timestep)   # when the model is IFNet(), the 'timestep' will be ignored internally.

            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            if step != 0 and step % int(print_freq) == 0:
                writer.add_scalar('learning_rate', learning_rate, step)
                writer.add_scalar('loss/l1', infos['loss_l1'], step)
                writer.add_scalar('loss/tea', infos['loss_tea'], step)
                writer.add_scalar('loss/distill', infos['loss_distill'], step)
                if args.wandb_name is not None:
                    if args.classifier_intra_v2 == True:
                        wandb.log({"train_time_interval": train_time_interval,
                                   "learning_rate": learning_rate,
                                   "loss/error_map1": infos['loss_errormap1'],
                                   "loss/error_map2": infos['loss_errormap2'],
                                   "loss/error_map3": infos['loss_errormap3'],
                                   "loss/pred_gt" : infos['loss_pred_gt'],
                                   "loss/pred_loss": infos['loss_pred']}, step=step)
                    else:
                        wandb.log({"train_time_interval": train_time_interval,
                                   "learning_rate": learning_rate,
                                   "loss/tea": infos['loss_tea'],
                                   "loss/distill": infos['loss_distill']}, step=step)
                ### image #######
                gt = (gt.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                mask = (torch.cat((infos['mask'], infos['mask_tea']), 3).permute(0, 2, 3,
                                                                               1).detach().cpu().numpy() * 255).astype(
                    'uint8')
                pred = (pred.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                merged_img = (infos['merged_tea'].permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                flow0 = infos['flow'].permute(0, 2, 3, 1).detach().cpu().numpy()
                flow1 = infos['flow_tea'].permute(0, 2, 3, 1).detach().cpu().numpy()
                wandb_images = [[]]
                for j in range(1):
                    imgs = np.concatenate((merged_img[j], pred[j], gt[j]), 1)[:, :, ::-1]
                    flows_fw = np.concatenate((flow2rgb(flow0[j][:, :, :2]), flow2rgb(flow1[j][:, :, :2])), 1)
                    flows_bw = np.concatenate((flow2rgb(flow0[j][:, :, 2:]), flow2rgb(flow1[j][:, :, 2:])), 1)
                    writer.add_image(str(j) + '/img', imgs, step, dataformats='HWC')
                    writer.add_image(str(j) + '/flow', np.concatenate((flow2rgb(flow0[j]), flow2rgb(flow1[j])), 1),
                                     step, dataformats='HWC')
                    writer.add_image(str(j) + '/mask', mask[j], step, dataformats='HWC')
                    if args.wandb_name is not None and args.wandb_image:
                        timestep_ = timestep[j] if arbitrary else 0.5
                        wandb_images[j].append(wandb.Image(imgs, caption='step_' + str(step) + '<tea_img|pred_img|gt|timestep({:.2f})>'.format(timestep_)))
                        wandb_images[j].append(wandb.Image(flows_fw, caption='step_' + str(step) + '<fw_flow|fw_flow_tea|timestep({:.2f})>'.format(timestep_)))
                        wandb_images[j].append(wandb.Image(flows_bw, caption='step_' + str(step) + '<bw_flow|bw_flow_tea|timestep({:.2f})>'.format(timestep_)))
                        wandb_images[j].append(wandb.Image(mask[j], caption='step_' + str(step) + '<mask>'))
                        wandb.log({"training images": wandb_images[j]}, step=step)
                writer.flush()
            step += 1            
        print('epoch:{} {}/{} time:{:.2f}+{:.2f} loss_l1:{:.4f} lr:{:.5f}'.format(epoch, i, args.step_per_epoch, data_time_interval, train_time_interval, infos['loss_l1'], learning_rate))

        nr_eval += 1
        if nr_eval % 1 == 0:
            eval_psnr = evaluate(model, val_data, step, args, writer_val, arbitrary)

            if eval_psnr > best_psnr:
                best_psnr = eval_psnr
                model.save_best_model(args.log_path, 0, epoch, best_psnr)
                if args.wandb_name is not None:
                    wandb.log({"Best PSNR": best_psnr, "Best Epoch": epoch}, step=step)
                    
        model.save_checkpoint(args.log_path, epoch, infos['loss_G'])
        if nr_eval % 10 == 0:
            model.save_model(args.log_path, 0, epoch, eval_psnr)
    model.save_last_model(args.log_path, 0, args.epoch, eval_psnr)




def evaluate(model, val_data, nr_eval, args, writer_val, arbitrary=False):
    loss_l1_list = []
    loss_distill_list = []
    loss_tea_list = []
    psnr_list = []
    psnr_no_refine_list = []
    time_stamp = time.time()

    pader = None
    with torch.no_grad():
        torch.cuda.empty_cache() 
        for i, data in enumerate(tqdm(val_data, desc='validation')):
            if arbitrary == True:
                data_gpu, timestep = data
                timestep = timestep.to(device, non_blocking=True)
            else:
                data_gpu = data
                timestep = 0
            data_gpu = data_gpu.to(device, non_blocking=True) / 255.
            
            n, c, h, w = data_gpu.shape
            if h == 720:
                pad = 24
                if pader == None:
                    pader = torch.nn.ReplicationPad2d([0, 0, pad, pad])
                data_gpu = pader(data_gpu)
                
            imgs = data_gpu[:, :6]
            gt = data_gpu[:, 6:9]
            
            with torch.no_grad():
                if args.classifier_intra_v2:
                    pred, infos = model.update_classifier_intra_v2(imgs, gt, training=False, timestep=timestep)
                else:
                    pred, infos = model.update(imgs, gt, training=False, timestep=timestep)   # when the model is IFNet(), the 'timestep' will be ignored internally.
            
            if h == 720:
                pred = pred[:, :, pad : -pad]
                gt = gt[:, :, pad : -pad]
                infos['merged_no_refine'] = infos['merged_no_refine'][:, :, pad : -pad]

            loss_l1_list.append(infos['loss_l1'].cpu().numpy())
            loss_tea_list.append(infos['loss_tea'].cpu().numpy())
            loss_distill_list.append(infos['loss_distill'].cpu().numpy())
            
            for j in range(gt.shape[0]):
                mse = torch.mean((gt[j] - infos['merged_no_refine'][j]) * (gt[j] - infos['merged_no_refine'][j]))
                if mse == 0:
                    print('mse = 0')
                    pass
                else:
                    psnr = -10 * math.log10(torch.mean((gt[j] - pred[j]) * (gt[j] - pred[j])).cpu().data)
                    psnr_list.append(psnr)
                    psnr_no_refine = -10 * math.log10(torch.mean((gt[j] - infos['merged_no_refine'][j]) * (gt[j] - infos['merged_no_refine'][j])).cpu().data)
                    psnr_no_refine_list.append(psnr_no_refine)
            if i == 0:
                merged_img = infos['merged_tea']
                wandb_images = [[] * 8 for j in range(8)]
                gt = (gt.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
                pred = (pred.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
                merged_img = (merged_img.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
                flow0 = infos['flow'].permute(0, 2, 3, 1).cpu().numpy()
                flow1 = infos['flow_tea'].permute(0, 2, 3, 1).cpu().numpy()

                for j in range(8):
                    imgs = np.concatenate((pred[j], gt[j]), 1)[:, :, ::-1]
                    flows_fw = np.concatenate((flow2rgb(flow0[j][:, :, :2]), flow2rgb(flow1[j][:, :, :2])), 1)
                    flows_bw = np.concatenate((flow2rgb(flow0[j][:, :, 2:]), flow2rgb(flow1[j][:, :, 2:])), 1)
                    writer_val.add_image(str(j) + '/img', imgs.copy(), nr_eval, dataformats='HWC')
                    writer_val.add_image(str(j) + '/flow', flow2rgb(flow0[j][:, :, ::-1]), nr_eval, dataformats='HWC')
                    writer_val.add_image(str(j) + '/flow_gt', flow2rgb(flow1[j][:, :, ::-1]), nr_eval, dataformats='HWC')
                    if args.wandb_name is not None and args.wandb_image:
                        timestep_ = timestep[j] if arbitrary else 0.5
                        wandb_images[j].append(wandb.Image(imgs, caption='step_' + str(nr_eval) + '_psnr_' + '{:.2f}'.format(psnr_list[j]) + '<pred_img|gt|timestep({:.2f})>'.format(timestep_)))
                        wandb_images[j].append(wandb.Image(flows_fw, caption='step_' + str(nr_eval) + '<fw_flow|fw_flow_tea|timestep({:.2f})>'.format(timestep_)))
                        wandb_images[j].append(wandb.Image(flows_bw, caption='step_' + str(nr_eval) + '<bw_flow|bw_flow_tea|timestep({:.2f})>'.format(timestep_)))
                        wandb.log({"validation images": wandb_images[j]}, step=nr_eval - (10-j))

    eval_time_interval = time.time() - time_stamp

    writer_val.add_scalar('psnr', np.array(psnr_list).mean(), nr_eval)
    if args.wandb_name is not None:
        wandb.log({"psnr": np.array(psnr_list).mean(),
                   "no refine psnr": np.array(psnr_no_refine_list).mean()}, step=nr_eval)
    print('psnr: {:.3f}, eval_time_interval: {:.3f}sec, iteration: {:03d}'.
          format(np.array(psnr_list).mean(), eval_time_interval, nr_eval))

    return np.array(psnr_list).mean()


if __name__ == "__main__":

    args = add_argparse()
    print(args)
    
    if args.wandb_name is not None:
        wandb.init(project=args.project_name)
        wandb.config.update(args)
        os.environ["WANDB_SILENT"] = "True"
        wandb.run.name = args.wandb_name

    device = torch.device("cuda")

    if not os.path.exists('trained_models'):
        os.makedirs('trained_models')
    args.log_path = 'trained_models/' + args.log_path
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    with open(args.log_path + '/args.txt', 'w') as fw:
        for key, value in vars(args).items():
            fw.write('{}: {}\n'.format(key, value))

    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    model = Model(local_rank=-1, classifier=args.classifier, classifier_intra_v2=args.classifier_intra_v2)
    
    train(model, args, arbitrary=args.arbitrary)
    
    # #inference best_epoch_pkl
    # best_model = glob.glob('{}/best*'.format(args.log_path))
    
    # #inference last_epoch_pkl
    # last_model = glob.glob('{}/last*'.format(args.log_path))

