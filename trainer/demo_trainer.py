import time
import torch
import datetime
import torch.nn.functional as F
import numpy as np

from utils.base import *
from utils.logger import *
from networks.wrapper import model_wrapper, loss_wrapper
from torchvision.utils import make_grid



class DemoTrainer(BCIBaseTrainer):

    def __init__(self, configs, exp_dir, resume_ckpt):
        super(DemoTrainer, self).__init__(configs, exp_dir, resume_ckpt)

    def forward(self, train_loader, val_loader):

        best_val_psnr = 0.0
        best_val_clsf = np.inf
        start_time = time.time()

        basic_msg = 'PSNR:{:.4f} SSIM:{:.4f} Epoch:{}'
        for epoch in range(self.start_epoch, self.epochs):
            train_metrics = self._train_epoch(train_loader, epoch)

            # save model with best val psnr
            val_model = self.model
            val_metrics = self._val_epoch(val_model, val_loader, epoch)
            psnr = val_metrics['psnr']
            ssim = val_metrics['ssim']
            info_list = [psnr, ssim, epoch]

            if psnr > best_val_psnr:
                best_val_psnr = psnr
                self._save_model(val_model, 'best_psnr')
                print('>>> Highest PSNR - Save Model <<<')
                psnr_msg = '- Best PSNR: ' + basic_msg.format(*info_list)


            # save checkpoint regularly
            if (epoch % self.ckpt_freq == 0) or (epoch + 1 == self.epochs):
                self._save_checkpoint(epoch)

            # write logs
            self._save_logs(epoch, train_metrics, val_metrics)
            print()

        print(psnr_msg)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('- Training time {}'.format(total_time_str))

        return

    def _train_epoch(self, loader, epoch):
        self.model.train()

        header = 'Train:[{}]'.format(epoch)
        logger = MetricLogger(header, self.print_freq)
        logger.add_meter('lr', SmoothedValue(1, '{value:.6f}'))

        data_iter = logger.log_every(loader)
        for iter_step, data in enumerate(data_iter):
            self.optimizer.zero_grad()

            # lr scheduler on per iteration
            if iter_step % self.accum_iter == 0:
                self._adjust_learning_rate(iter_step / len(loader) + epoch)
            logger.update(lr=self.optimizer.param_groups[0]['lr'])

            # forward
            he, ihc, level = [d.to(self.device) for d in data]
            ihc_phr, first_layer_f, before_koopman_f, koopman_f, last_layer_f = self.model(he)

            loss = self.loss(ihc, ihc_phr)
            loss.backward()
            self.optimizer.step()

            self.step += 1
            if iter_step % 300 == 0:
                grid_image = make_grid(he[0].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('HE', grid_image, iter_step)
                grid_image = make_grid(ihc[0].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('GT_IHC', grid_image, iter_step)
                grid_image = make_grid(ihc_phr[0].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('Predicted_IHC', grid_image, iter_step)

                first_layer_f = first_layer_f.clone()
                first_layer_f = first_layer_f.data.cpu().numpy().squeeze()
                first_layer_f = (first_layer_f - first_layer_f.min()) / (first_layer_f.max() - first_layer_f.min() + 1e-8)
                self.writer.add_image('First_layer', torch.tensor(first_layer_f), iter_step, dataformats='HW')

                before_koopman_f = before_koopman_f.clone()
                before_koopman_f = before_koopman_f.data.cpu().numpy().squeeze()
                before_koopman_f = (before_koopman_f - before_koopman_f.min()) / (before_koopman_f.max() - before_koopman_f.min() + 1e-8)
                self.writer.add_image('before_koopman', torch.tensor(before_koopman_f), iter_step, dataformats='HW')

                koopman_f = koopman_f.clone()
                koopman_f = koopman_f.data.cpu().numpy().squeeze()
                koopman_f = (koopman_f - koopman_f.min()) / (koopman_f.max() - koopman_f.min() + 1e-8)
                self.writer.add_image('after_Koopman', torch.tensor(koopman_f), iter_step, dataformats='HW')

                last_layer_f = last_layer_f.clone()
                last_layer_f = last_layer_f.data.cpu().numpy().squeeze()
                last_layer_f = (last_layer_f - last_layer_f.min()) / (last_layer_f.max() - last_layer_f.min() + 1e-8)
                self.writer.add_image('Last_layer', torch.tensor(last_layer_f), iter_step, dataformats='HW')

        logger_info = {
            key: meter.global_avg
            for key, meter in logger.meters.items()
        }
        return logger_info

    @torch.no_grad()
    def _val_epoch(self, val_model, loader, epoch):

        val_model.eval()
        header = ' Val :[{}]'.format(epoch)
        logger = MetricLogger(header, self.print_freq)

        data_iter = logger.log_every(loader)
        for step, data in enumerate(data_iter):
            he, ihc, level = [d.to(self.device) for d in data]
            ihc_phr, first_layer_f, before_koopman_f, koopman_f, last_layer_f = val_model(he)

            psnr, ssim = self.eval_metrics(ihc_phr, ihc)
            logger.update(psnr=psnr.item(), ssim=ssim.item())

        logger_info = {
            key: meter.global_avg
            for key, meter in logger.meters.items()
        }
        return logger_info