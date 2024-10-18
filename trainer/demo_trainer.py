import time
import torch
import datetime
import torch.nn.functional as F
import numpy as np

from utils.base import *
from utils.logger import *


class BCITrainerBasic(BCIBaseTrainer):

    def __init__(self, configs, exp_dir, resume_ckpt):
        super(BCITrainerBasic, self).__init__(configs, exp_dir, resume_ckpt)

    def forward(self, train_loader, val_loader):

        best_val_psnr = 0.0
        best_val_clsf = np.inf
        start_time = time.time()

        basic_msg = 'PSNR:{:.4f} SSIM:{:.4f} Epoch:{}'
        for epoch in range(self.start_epoch, self.epochs):
            train_metrics = self._train_epoch(train_loader, epoch)

            # save model with best val psnr
            val_model = self.Gema if self.ema else self.G
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
            outputs = self.model(he)
            ihc_phr, ihc_plr, he_plevel = outputs

            # update G
            self._set_requires_grad(self.D, False)
            Ggan, Grec, Gsim = self._G_loss(he, ihc, ihc_phr, ihc_plr)
            Gcls = self.gcl_loss(he_plevel, level)
            logger.update(Gg=Ggan.item(), Gr=Grec.item(),
                          Gs=Gsim.item(), Gc=Gcls.item())
            lossG = self.loss(ihc, ihc_phr)





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
        for _, data in enumerate(data_iter):
            he, ihc, level = [d.to(self.device) for d in data]
            outputs = val_model(he)
            ihc_phr = outputs[0]

            psnr, ssim = self.eval_metrics(ihc_phr, ihc)
            logger.update(psnr=psnr.item(), ssim=ssim.item())

            if self.apply_cmp:
                self.C.eval()
                ihc_plevel, ihc_platent = self.C(ihc_phr)
                clsf = self.ccl_loss(ihc_plevel, level)
                logger.update(clsf=clsf.item())

        logger_info = {
            key: meter.global_avg
            for key, meter in logger.meters.items()
        }
        return logger_info