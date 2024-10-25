from piqa import SSIM, PSNR
import torch.nn as nn


class EvalMetrics(nn.Module):

    def __init__(self):
        super(EvalMetrics, self).__init__()

        self.psnr = PSNR(value_range=255)
        self.ssim = SSIM(window_size=9, sigma=2.375, n_channels=3, value_range=255)

    def forward(self, prediction, target):
        # range in [0, 255]
        target_ = (target + 1.0) / 2.0 * 255.0
        prediction_ = (prediction + 1.0) / 2.0 * 255.0

        psnr = self.psnr(prediction_, target_)
        ssim = self.ssim(prediction_, target_)

        return psnr, ssim