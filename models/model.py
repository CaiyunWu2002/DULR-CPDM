
from logging import Logger
from torch.cuda.amp import autocast, GradScaler
from models.network_cpdm import DULR
from typing import Any, Dict, List, Union
import torch.nn as nn

from torch.optim import Adam, lr_scheduler

import torch.nn.functional as F

from models.select_network import select_network
from utils import utils_image as util
from utils.utils_image import get_img_from_np, scale

import torch
import numpy as np
import os


class Model:
    def __init__(self, opt: Dict[str, Any]):
        self.opt = opt
        self.opt_train = self.opt['train']
        self.opt_test = self.opt['test']
        self.loss_history = []   
        self.loss_history_s1 = []
        self.loss_history_s2 = []  
        self.weights=opt['train']['weights']

        self.save_dir: str = opt['path']['models']
        self.device = torch.device(
            'cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.is_train = opt['is_train']
        self.type = opt['netG']['type']

        self.test_sparsity_sum = 0
        self.test_sample_num = 0
        self.vis_save_dir = "./test_visualization"
        os.makedirs(self.vis_save_dir, exist_ok=True)
        self.test_sparsity_all_samples=[]

        # self.freeze_modules=opt['netG']['freeze_modules']
        # self.enable_freeze = len(self.freeze_modules) > 0

        self.net = select_network(opt).to(self.device)

        if opt['gpu_ids'] is not None and len(opt['gpu_ids']) > 1:
            self.net = nn.DataParallel(self.net)

        self.schedulers = []
        self.log_dict = {}
        self.metrics = {}

        self.check_interval = 2   
        self.iter_count = 0
        self.frozen_snapshots = None   
        self.scaler = GradScaler()   
        self.accumulation_steps = 2   

    def init(self):

        self.load()
        self.net.train()
        self.define_loss()
        self.define_optimizer()
        self.define_scheduler()


    def load(self):
        load_path = self.opt['path']['pretrained_netG']
        if load_path is not None:
            print('Loading model for G [{:s}] ...'.format(load_path))
            self.load_network(load_path, self.net)

    def load_network(self, load_path: str, network: Union[nn.DataParallel,
                                                          DULR]):
        if isinstance(network, nn.DataParallel):
            network: DULR = network.module

        network.head_p.load_state_dict(torch.load(load_path + 'head_p.pth'),
                                     strict=True)


        state_dict_x = torch.load(load_path + 'pdm_x.pth')
        network.pdm.net_x.load_state_dict(state_dict_x, strict=True)

        state_dict_d = torch.load(load_path + 'pdm_d.pth')
        network.pdm.net_d.load_state_dict(state_dict_d, strict=True)

        network.hypa_p.load_state_dict(torch.load(load_path + 'hypa_p.pth'),
                                       strict=True)


    def load_test_checkpoint(
            self,
            checkpoint_path: str,
            strict: bool = True
    ) -> None:

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

         
        if 'model_state' not in checkpoint:
            raise KeyError("can not find 'model_state'")
        pretrained_state = checkpoint['model_state']
        model_state = self._get_model_state_dict()   

         
        has_pretrained_module = any(k.startswith('module.') for k in pretrained_state)
        has_model_module = any(k.startswith('module.') for k in model_state)

        if has_pretrained_module and not has_model_module:
            pretrained_state = self._remove_module_prefix(pretrained_state)   
        elif not has_pretrained_module and has_model_module:
            pretrained_state = self._add_module_prefix(pretrained_state)   

         
        filtered_state = {
            k: v for k, v in pretrained_state.items()
            if k in model_state and model_state[k].shape == v.shape
        }
         
        model_state.update(filtered_state)
        self.net.load_state_dict(model_state, strict=strict)
        self.net.eval()

        missing_keys = [k for k in model_state if k not in filtered_state]
        if missing_keys and strict:
            print(f"WARNING: Missing parameter keys in strict mode: {missing_keys[:5]}")
        elif missing_keys:
            print(f"INFO: Ignoring missing parameter keys in non-strict mode: {len(missing_keys)}")

        print(f"Model weights loaded for testing, switched to eval mode")


    def save(self, logger: Logger, epoch):
      logger.info(f'Saving the model at epoch {epoch}.')
      net = self.net
      if isinstance(net, nn.DataParallel):
        net = net.module
     
      epoch_folder = os.path.join(self.save_dir, f'epoch_{epoch}')
      if not os.path.exists(epoch_folder):
        os.makedirs(epoch_folder)


      self.save_network(net.pdm.net_x, 'pdm_x', epoch_folder)
      self.save_network(net.hypa_p, 'hypa_p', epoch_folder)
      self.save_network(net.head_p, 'head_p', epoch_folder)
      self.save_network(net.pdm.net_d, 'pdm_d', epoch_folder)



    def save_network(self, network, network_label, epoch_folder):
        filename = '{}.pth'.format(network_label)
        save_path = os.path.join(epoch_folder, filename)
        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path, _use_new_zipfile_serialization=False)

    def save_best_model(self, path, epoch, step, loss):
        state = {
            'epoch': epoch,
            'step': step,
            'best_loss': loss,
            'state_dict': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler_state': self.schedulers[0].state_dict() if self.schedulers else None,
            'loss_history': self.loss_history   
        }
        torch.save(state, path)
        step_model_path = os.path.join(os.path.dirname(path), f'step_{step}.pth')
        torch.save(state, step_model_path)

    def get_best_loss(self, path):
        if os.path.exists(path):
            state = torch.load(path)
            return state['best_loss']
        return float('inf')

     
    def save_checkpoint(self, epoch: int, current_step: int, save_dir: str,saveold=False):
        checkpoint = {
            'epoch': epoch,
            'current_step': current_step,
            'model_state': self.net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            # 'scheduler_state': self.schedulers[0].state_dict() if self.schedulers else None,
            'schedulers_state_dict': [sch.state_dict() for sch in self.schedulers] if self.schedulers else None,
            'loss_history': self.loss_history  
        }
         
        checkpoint_dir = os.path.join(save_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
         
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'latest.pth'))
        if saveold:
          torch.save(checkpoint, os.path.join(checkpoint_dir, f'step_{current_step}.pth'))

    def _get_model_state_dict(self):
        if isinstance(self.net, nn.DataParallel):
            return self.net.module.state_dict()
        else:
            return self.net.state_dict()


    def _remove_module_prefix(self, state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        return new_state_dict

    def _add_module_prefix(self, state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[f'module.{k}'] = v
        return new_state_dict

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model_state = self._get_model_state_dict()
        pretrained_state = checkpoint['model_state']

        if any(k.startswith('module.') for k in pretrained_state) and \
                not any(k.startswith('module.') for k in model_state):
            pretrained_state = self._remove_module_prefix(pretrained_state)
        elif not any(k.startswith('module.') for k in pretrained_state) and \
                any(k.startswith('module.') for k in model_state):
            pretrained_state = self._add_module_prefix(pretrained_state)

        filtered_state = {
            k: v for k, v in pretrained_state.items()
            if k in model_state and model_state[k].shape == v.shape
        }

        hypa_list_keys = [k for k in model_state if k.startswith('hypa_list.')]
        if hypa_list_keys:
            old_modules_count = max(int(k.split('.')[1]) for k in filtered_state if k.startswith('hypa_list.')) + 1
            new_modules_count = max(int(k.split('.')[1]) for k in hypa_list_keys) + 1
             
            for i in range(old_modules_count, new_modules_count):
                for key_template in [k for k in filtered_state if k.startswith(f'hypa_list.0.')]:
                     
                    source_key = key_template.replace('hypa_list.0.', f'hypa_list.{min(i, old_modules_count - 1)}.')
                    target_key = key_template.replace('hypa_list.0.', f'hypa_list.{i}.')

                    if source_key in filtered_state and target_key in model_state:
                        model_state[target_key] = filtered_state[source_key].clone()
                        print(f"copy weights: {source_key} -> {target_key}")

        model_state.update(filtered_state)
        self.net.load_state_dict(model_state)

        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        except ValueError as e:
            current_param_groups = self.optimizer.param_groups
            optimizer_state = checkpoint['optimizer_state']
            new_optimizer_state = {
                'state': {},
                'param_groups': []
            }
            param_to_group_idx = {}
            for i, group in enumerate(current_param_groups):
                for param in group['params']:
                    param_to_group_idx[id(param)] = i

            for param_id, state in optimizer_state['state'].items():
                if param_id in param_to_group_idx:
                    new_optimizer_state['state'][param_id] = state

            for i, group in enumerate(current_param_groups):
                new_group = {k: v for k, v in group.items() if k != 'params'}
                new_group['params'] = [p for p in group['params'] if id(p) in param_to_group_idx]
                new_optimizer_state['param_groups'].append(new_group)

            self.optimizer.load_state_dict(new_optimizer_state)
            print("Successfully loaded optimizer state selectively")
        if self.schedulers and checkpoint.get('schedulers_state_dict'):
            try:
                from collections import Counter
                for i, sch_state in enumerate(checkpoint['schedulers_state_dict']):
                    if i < len(self.schedulers):
                        self.schedulers[i].load_state_dict(sch_state)
                        self.schedulers[i].milestones = Counter(self.opt_train['G_scheduler_milestones'])
                        self.schedulers[i].gamma = self.opt_train['G_scheduler_gamma']

                print(f"Successfully loaded scheduler progress, forcing new configuration (Counter type milestones)")
            except Exception as e:
                print(f"Failed to load scheduler progress: {e}")
         
        self.loss_history = checkpoint.get('loss_history', [])

        start_epoch = checkpoint.get('epoch', 0)
        current_step = checkpoint.get('current_step', 0)
        print(f"Resuming training from epoch {start_epoch}, step {current_step}")
        return start_epoch, current_step, self.loss_history

    def define_loss(self):
        self.lossfn = nn.L1Loss().to(self.device)

    def define_optimizer(self):
        optim_params = list(filter(lambda p: p.requires_grad, self.net.parameters()))

        self.optimizer = Adam(
            optim_params,
            lr=self.opt_train['G_optimizer_lr'],
            weight_decay=0
        )

    def define_scheduler(self):
        self.schedulers.append(
            lr_scheduler.MultiStepLR(self.optimizer,
                                     self.opt_train['G_scheduler_milestones'],
                                     self.opt_train['G_scheduler_gamma']))


    def update_learning_rate(self, n: int):
        for scheduler in self.schedulers:
            scheduler.step(n)

    @property
    def learning_rate(self) -> float:
        return self.schedulers[0].get_last_lr()[0]
        # return self.schedulers[0].get_lr()[0]

    def feed_data(self, data: Dict[str, Any]):
        self.y = data['y'].to(self.device)
        self.y_gt = data['y_gt'].to(self.device)

        self.sigma = data['sigma'].to(self.device)
        self.hpath = data['hpath']

    def polar_loss(self, output, gt):
        l1loss = nn.L1Loss().to(self.device)
        gradientloss = GradientLoss().to(self.device)

        output_0 = output[:, 0:3, :, :].clamp(0, 1)
        output_45 = output[:, 3:6, :, :].clamp(0, 1)
        output_90 = output[:, 6:9, :, :].clamp(0, 1)
        output_135 = output[:, 9:12, :, :].clamp(0, 1)

        gt_0 = gt[:, 0:3, :, :].clamp(0, 1)
        gt_45 = gt[:, 3:6, :, :].clamp(0, 1)
        gt_90 = gt[:, 6:9, :, :].clamp(0, 1)
        gt_135 = gt[:, 9:12, :, :].clamp(0, 1)
         
        output_S0 = (output_0 + output_45 + output_90 + output_135) / 2
        output_S1 = output_0 - output_90
        output_S2 = output_45 - output_135

        gt_S0 = (gt_0 + gt_45 + gt_90 + gt_135) / 2
        gt_S1 = gt_0 - gt_90
        gt_S2 = gt_45 - gt_135

        epsilon = 1e-8

        output_dop = torch.sqrt(output_S1 ** 2 + output_S2 ** 2 + epsilon) / (output_S0 + epsilon)
        output_dop = output_dop.clamp(0, 1)

        gt_dop = torch.sqrt(gt_S1 ** 2 + gt_S2 ** 2 + epsilon) / (gt_S0 + epsilon)
        gt_dop = gt_dop.clamp(0, 1)

         
        epsilon_s = 1e-12
        output_S1 = output_S1 + epsilon_s * torch.sign(output_S1 + 1e-20)   
        output_S2 = output_S2 + epsilon_s * torch.sign(output_S2 + 1e-20)   
        gt_S1 = gt_S1 + epsilon_s * torch.sign(gt_S1 + 1e-20)
        gt_S2 = gt_S2 + epsilon_s * torch.sign(gt_S2 + 1e-20)

         
        clip_max = 1
        output_S1_clamped = torch.clamp(output_S1, -clip_max, clip_max)
        output_S2_clamped = torch.clamp(output_S2, -clip_max, clip_max)
        gt_S1_clamped = torch.clamp(gt_S1, -clip_max, clip_max)
        gt_S2_clamped = torch.clamp(gt_S2, -clip_max, clip_max)

         
        output_aop = scale(torch.atan2(output_S2_clamped, output_S1_clamped) / 2, -torch.pi / 2, torch.pi / 2, 0,
                           torch.pi)
        gt_aop = scale(torch.atan2(gt_S2_clamped, gt_S1_clamped) / 2, -torch.pi / 2, torch.pi / 2, 0, torch.pi)

        # aop_loss=0.8*gradientloss(output_aop, gt_aop) + 1.2*l1loss(output_aop, gt_aop)
        angle_diff = 2 * (output_aop - gt_aop)   
        aoploss = (1 - torch.cos(angle_diff)).mean()

         
        img_loss = gradientloss(output, gt) + l1loss(output, gt)

         
        stokes_loss = l1loss(output_S1, gt_S1) + l1loss(output_S2, gt_S2)+l1loss(output_S0, gt_S0)

         
        # dop_loss = l1loss(output_dop, gt_dop)
         
        dop_l1_loss = l1loss(output_dop, gt_dop)

        high_dop_weight = torch.where(gt_dop > 0.5, 2.0, 1.0).to(self.device)
        weighted_dop_loss = torch.mean(high_dop_weight * torch.abs(output_dop - gt_dop))
        dop_gradient_loss = gradientloss(output_dop, gt_dop)
        dop_loss = 0.4 * dop_l1_loss + 0.4 * weighted_dop_loss + 0.2 * dop_gradient_loss

         
        polar_loss = l1loss(output_0 + output_90, output_45 + output_135)
        lossaLL = img_loss + stokes_loss + 0.05 * aoploss + 0.8 * dop_loss
        # lossaLL = img_loss + s_lameda*stokes_loss+0.05*aop_loss
        return lossaLL

    def cal_multi_loss(self, preds: List[torch.Tensor],
                       gt: torch.Tensor) -> torch.Tensor:
        losses = None
        for i, pred in enumerate(preds):
            # loss = self.lossfn(pred, gt)
            loss = self.polar_loss(pred, gt)
            if i != len(preds) - 1:
                loss *= (1 / (len(preds) - 1))
            if i == 0:
                losses = loss
            else:
                losses += loss
        return losses

    def log_train(self, current_step: int, epoch: int, logger: Logger,aveloss: float):
        message = f'Training epoch:{epoch:3d}, iter:{current_step:8,d}, lr:{self.learning_rate:.3e},aveloss:{aveloss:.3e}'
        for k, v in self.log_dict.items(
        ):  # merge log information into message
            message += f', {k:s}: {v:.3e}'
        logger.info(message)

    def test(self):
        self.net.eval()
        with torch.no_grad():
            y = self.y
            h, w = y.size()[-2:]
            top = slice(0, h // 8 * 8)
            left = slice(0, (w // 8 * 8))
            y = y[..., top, left]
            self.dx, self.d,self.x = self.net(y)
        self.prepare_visuals()
        self.net.train()

    def prepare_visuals(self):
        self.out_dict = {}
        self.out_dict['y'] = util.tensor2single(self.y[0].detach().float().cpu())
        self.out_dict['dx'] = util.tensor2single(
            self.dx[0].detach().float().cpu())
        self.out_dict['d'] = self.d[0].detach().float().cpu()
        self.out_dict['y_gt'] = util.tensor2single(
            self.y_gt[0].detach().float().cpu())
        self.out_dict['hpath'] = self.hpath[0]

    def cal_metrics(self,gt,out):
        gt_img = gt
        processed_img = out
        [gt_I0, gt_I45, gt_I90, gt_I135, gt_s0, gt_s1, gt_s2, gt_aop, gt_dolp] = get_img_from_np(gt_img)
        [out_I0, out_I45, out_I90, out_I135, out_s0, out_s1, out_s2, out_aop, out_dolp] = get_img_from_np(processed_img)

        I0_psnr = util.calculate_psnr(out_I0, gt_I0)
        I90_psnr = util.calculate_psnr(out_I90, gt_I90)
        I45_psnr = util.calculate_psnr(out_I45, gt_I45)
        I135_psnr = util.calculate_psnr(out_I135, gt_I135)
        s0_psnr = util.calculate_psnr(out_s0, gt_s0)
        s1_psnr = util.calculate_psnr(out_s1, gt_s1)
        s2_psnr = util.calculate_psnr(out_s2, gt_s2)
        dolp_psnr = util.calculate_psnr(out_dolp, gt_dolp)
        aop_psnr = util.calculate_psnr(out_aop, gt_aop)

        I0_ssim = util.calculate_ssim(out_I0, gt_I0)
        I90_ssim = util.calculate_ssim(out_I90, gt_I90)
        I45_ssim = util.calculate_ssim(out_I45, gt_I45)
        I135_ssim = util.calculate_ssim(out_I135, gt_I135)
        s0_ssim = util.calculate_ssim(out_s0, gt_s0)
        s1_ssim = util.calculate_ssim(out_s1, gt_s1)
        s2_ssim = util.calculate_ssim(out_s2, gt_s2)
        dolp_ssim = util.calculate_ssim(out_dolp, gt_dolp)
        aop_ssim = util.calculate_ssim(out_aop, gt_aop)
        aop_mae=util.calculate_mae(out_aop, gt_aop)

        img_filename = os.path.basename(self.out_dict['hpath'])
         
        img_filename = os.path.splitext(img_filename)[0]
        psnr_dict = {
            'image_filename': img_filename,
            'I0_psnr': I0_psnr,
            'I45_psnr': I45_psnr,
            'I90_psnr': I90_psnr,
            'I135_psnr': I135_psnr,
            's0_psnr': s0_psnr,
            's1_psnr': s1_psnr,
            's2_psnr': s2_psnr,
            'dolp_psnr': dolp_psnr,
            'aop_psnr': aop_psnr,

            'I0_ssim': I0_ssim,
            'I45_ssim': I45_ssim,
            'I90_ssim': I90_ssim,
            'I135_ssim': I135_ssim,
            's0_ssim': s0_ssim,
            's1_ssim ': s1_ssim,
            's2_ssim': s2_ssim,
            'dolp_ssim': dolp_ssim,
            'aop_ssim': aop_ssim,
            'aop_mae':aop_mae
        }
        return psnr_dict

    def save_visuals(self, epoch: int):

        y_gt = self.out_dict['y_gt']
        y_img = self.out_dict['y']
        d_img = self.out_dict['d']
        dx_img = self.out_dict['dx']
        hpath = self.out_dict['hpath']
        # lpath = self.out_dict['lpath']

        img_name = os.path.splitext(os.path.basename(hpath))[0]
        img_dir = os.path.join(self.opt['path']['images'], str(epoch), img_name)
        os.makedirs(img_dir, exist_ok=True)

         
        [gt_I0, gt_I45, gt_I90, gt_I135, gt_s0, gt_s1, gt_s2, gt_aop, gt_dolp] = get_img_from_np(y_gt)
        [out_I0, out_I45, out_I90, out_I135, out_s0, out_s1, out_s2, out_aop, out_dolp] = get_img_from_np(dx_img)
        [pre_I0, pre_I45, pre_I90, pre_I135, pre_s0, pre_s1, pre_s2, pre_aop, pre_dolp] = get_img_from_np(y_img)

        util.imsave((out_I0 * 255).astype(np.uint8), os.path.join(img_dir, f"0.png"))
        util.imsave((out_I135 * 255).astype(np.uint8), os.path.join(img_dir, f"135.png"))
        util.imsave((out_I45 * 255).astype(np.uint8), os.path.join(img_dir, f"45.png"))
        util.imsave((out_I90 * 255).astype(np.uint8), os.path.join(img_dir, f"90.png"))

        # [out_s0, out_s1, out_s2, out_dolp, out_aop] = calculate_stokes(out_I0, out_I45, out_I90, out_I135)
         
        out_s0 = (out_s0 * 255).astype(np.uint8)
        out_s1 = (out_s1 * 255).astype(np.uint8)
        out_s2 = (out_s2 * 255).astype(np.uint8)
        out_dolp = (out_dolp * 255).astype(np.uint8)
        out_aop = (out_aop * 255).astype(np.uint8)
        util.imsave(out_s0 / 2, os.path.join(img_dir, f"{img_name}_s0/2.png"))

        util.imsave(out_dolp, os.path.join(img_dir, f"{img_name}_dolp.png"))
        util.imsave(out_aop, os.path.join(img_dir, f"{img_name}_aop.png"))

    def _get_module(self, path: str):
        try:
            net = self.net.module if isinstance(self.net, nn.DataParallel) else self.net
            for part in path.split('.'):
                net = getattr(net, part)
            return net
        except:
            print(f"can not find：{path}")
            return None

    def train(self):
        self.optimizer.zero_grad()

        dxs, self.d ,self.x= self.net(self.y)
        loss = self.cal_multi_loss(dxs, self.y_gt)

        self.log_dict['G_loss'] = loss.item()
        self.loss_history.append(loss.item())

        self.dx = dxs[-1]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        self.optimizer.step()


class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
         
        self.sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)

        self.sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)

    def forward(self, pred, target):
         
        in_channels = pred.size(1)

         
        sobel_x = self.sobel_x.expand(in_channels, 1, 3, 3).to(pred.device)
        sobel_y = self.sobel_y.expand(in_channels, 1, 3, 3).to(pred.device)

         
        gx_pred = F.conv2d(pred, sobel_x, padding=1, groups=in_channels)
        gy_pred = F.conv2d(pred, sobel_y, padding=1, groups=in_channels)

         
        gx_target = F.conv2d(target, sobel_x, padding=1, groups=in_channels)
        gy_target = F.conv2d(target, sobel_y, padding=1, groups=in_channels)

         
        loss = F.l1_loss(gx_pred, gx_target) + F.l1_loss(gy_pred, gy_target)
        return loss


