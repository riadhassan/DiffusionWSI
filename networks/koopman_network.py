import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


def gaussian_init_(n_units, std=1):
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]
    return Omega
# Define Koopman Layer
class KoopmanLayer(nn.Module):
    def __init__(self, in_channels, hidden_dim=256):
        super(KoopmanLayer, self).__init__()
        # Learn linear transformations (approximating Koopman operator)
        self.fc1 = nn.Linear(in_channels, in_channels, bias=False)
        # self.fc2 = nn.Linear(hidden_dim, in_channels, bias=False)
        self.fc1.weight.data = gaussian_init_(n_units=in_channels)
        U, _, V = torch.svd(self.fc1.weight.data)
        self.fc1.weight.data = torch.mm(U, V.t())


    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        x = x.view(batch_size, num_channels, -1)  # Flatten spatial dimensions
        # x = F.relu(self.fc1(x))  # Nonlinear transformation
        x = self.fc1(x)  # Linear Koopman operator approximation
        x = x.view(batch_size, num_channels, height, width)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // ratio, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // ratio, in_channels, bias=False)

    def forward(self, x):
        batch, channel, height, width = x.size()

        # Average Pooling
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(batch, channel)
        avg_out = self.relu(self.fc1(avg_pool))
        avg_out = self.fc2(avg_out).view(batch, channel, 1, 1)

        # Max Pooling
        max_pool = F.adaptive_max_pool2d(x, 1).view(batch, channel)
        max_out = self.fc2(self.relu(self.fc1(max_pool))).view(batch, channel, 1, 1)

        # Combine and apply sigmoid
        scale = torch.sigmoid(avg_out + max_out)
        return x * scale.expand_as(x)


# channel attention
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_pool, max_pool], dim=1)
        scale = torch.sigmoid(self.conv(concat))
        return x * scale.expand_as(x)


class CBAM(nn.Module):
    def __init__(self, in_channels, ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, ratio=ratio)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


#Global context block
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=0, padding_type='zero',
                 use_bias=True, spectral_norm=False):
        super(ConvLayer, self).__init__()

        self.padding_type = padding_type
        self.spectral_norm = spectral_norm

        if padding > 0:
            self.padding = padding
        else:
            self.padding = kernel_size // 2 if stride == 1 else 0

        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                         padding=self.padding, bias=use_bias)

        if spectral_norm:
            self.conv = nn.utils.spectral_norm(conv)
        else:
            self.conv = conv

    def forward(self, x):
        if self.padding_type == 'reflect':
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='reflect')
        elif self.padding_type == 'zero' and self.padding > 0:
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='constant')

        x = self.conv(x)
        return x

class GlobalContextBlock(nn.Module):
    def __init__(self, in_channels, use_bias=True, spectral_norm=False):
        super(GlobalContextBlock, self).__init__()

        self.context_conv = ConvLayer(in_channels, 1, kernel_size=1, stride=1, use_bias=use_bias,
                                      spectral_norm=spectral_norm)
        self.transform_0_conv1 = ConvLayer(in_channels, in_channels, kernel_size=1, stride=1, use_bias=use_bias,
                                           spectral_norm=spectral_norm)
        self.transform_0_conv2 = ConvLayer(in_channels, in_channels, kernel_size=1, stride=1, use_bias=use_bias,
                                           spectral_norm=spectral_norm)
        self.transform_1_conv1 = ConvLayer(in_channels, in_channels, kernel_size=1, stride=1, use_bias=use_bias,
                                           spectral_norm=spectral_norm)
        self.transform_1_conv2 = ConvLayer(in_channels, in_channels, kernel_size=1, stride=1, use_bias=use_bias,
                                           spectral_norm=spectral_norm)

    def forward(self, x):
        bs, c, h, w = x.size()

        # Context Modeling
        context_mask = self.context_conv(x).view(bs, 1, h * w)
        context_mask = F.softmax(context_mask, dim=-1)

        input_x = x.view(bs, c, h * w)
        context = torch.bmm(input_x, context_mask.permute(0, 2, 1)).view(bs, c, 1, 1)

        # Transform 0
        context_transform = self.transform_0_conv1(context)
        context_transform = F.layer_norm(context_transform, [c, 1, 1])
        context_transform = F.relu(context_transform)
        context_transform = self.transform_0_conv2(context_transform)
        context_transform = torch.sigmoid(context_transform)

        x = x * context_transform

        # Transform 1
        context_transform = self.transform_1_conv1(context)
        context_transform = F.layer_norm(context_transform, [c, 1, 1])
        context_transform = F.relu(context_transform)
        context_transform = self.transform_1_conv2(context_transform)

        x = x + context_transform
        return x

class Encoder(nn.Module):
  def __init__(self, in_channels, init_features):
    super(Encoder,self).__init__()
    self.features = init_features
    self.in_channels = in_channels
    self.encoder1 = Unet_block._block(self.in_channels, self.features, name="enc1")
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.encoder2 = Unet_block._block(self.features, self.features * 2, name="enc2")
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.encoder3 = Unet_block._block(self.features * 2, self.features * 4, name="enc3")
    self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.encoder4 = Unet_block._block(self.features * 4, self.features * 8, name="enc4")


  def forward(self, x):
    enc1 = self.encoder1(x)
    enc2 = self.encoder2(self.pool1(enc1))
    enc3 = self.encoder3(self.pool2(enc2))
    enc4 = self.encoder4(self.pool3(enc3))
    skips = [enc1, enc2, enc3, enc4]

    return enc4, skips

class Bottleneck(nn.Module):
  def __init__(self, init_features):
    super(Bottleneck, self).__init__()
    self.features = init_features
    self.bottleneck = Unet_block._block(self.features * 8, self.features * 16, name="bottleneck")
    self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

  def forward(self, x):
    return self.bottleneck(self.pool4(x))


class Decoder(nn.Module):
  def __init__(self, init_features):
    super(Decoder, self).__init__()
    self.features = init_features
    self.upconv4 = nn.ConvTranspose2d(self.features * 16, self.features * 8, kernel_size=2, stride=2)
    self.channel_attention4 = ChannelAttention(self.features * 16)
    self.decoder4 = Unet_block._block((self.features * 8) * 2, self.features * 8, name="dec4")


    self.upconv3 = nn.ConvTranspose2d(self.features * 8, self.features * 4, kernel_size=2, stride=2)
    self.channel_attention3 = ChannelAttention(self.features * 8)
    self.decoder3 = Unet_block._block((self.features * 4) * 2, self.features * 4, name="dec3")

    self.upconv2 = nn.ConvTranspose2d(self.features * 4, self.features * 2, kernel_size=2, stride=2)
    self.channel_attention2 = ChannelAttention(self.features * 4)
    self.decoder2 = Unet_block._block((self.features * 2) * 2, self.features * 2, name="dec2")

    self.upconv1 = nn.ConvTranspose2d(self.features * 2, self.features, kernel_size=2, stride=2)
    self.channel_attention1 = ChannelAttention(self.features * 2)
    self.decoder1 = Unet_block._block(self.features * 2, self.features, name="dec1")



  def forward(self, x, skips):
    dec4 = self.upconv4(x)
    dec4 = torch.cat((dec4, skips[-1]), dim=1)
    dec4 = self.channel_attention4(dec4)
    dec4 = self.decoder4(dec4)
    dec3 = self.upconv3(dec4)
    dec3 = torch.cat((dec3, skips[-2]), dim=1)
    dec3 = self.channel_attention3(dec3)
    dec3 = self.decoder3(dec3)
    dec2 = self.upconv2(dec3)
    dec2 = torch.cat((dec2, skips[-3]), dim=1)
    dec2 = self.channel_attention2(dec2)
    dec2 = self.decoder2(dec2)
    dec1 = self.upconv1(dec2)
    dec1 = torch.cat((dec1, skips[-4]), dim=1)
    dec1 = self.channel_attention1(dec1)
    dec1 = self.decoder1(dec1)

    return dec1

class UNet_koopman(nn.Module):
  def __init__(self, in_channels=1, out_channels=5, init_features=64):
    super(UNet_koopman, self).__init__()
    self.features = init_features
    self.in_channel = in_channels
    self.out_channel = out_channels

    self.encoder = Encoder(self.in_channel, self.features)
    self.bottleneck_layer = Bottleneck(self.features)
    self.decoder = Decoder(self.features)

    self.koopman_layer = KoopmanLayer(256)
    self.global_contex_attention = GlobalContextBlock(1024)


    self.conv = nn.Conv2d(in_channels=self.features, out_channels=self.out_channel, kernel_size=1)


  def forward(self, x):
    encoded, skips = self.encoder(x)

    bn = self.bottleneck_layer(encoded)
    bn = self.global_contex_attention(bn)
    km = self.koopman_layer(bn)

    decoded1 = self.decoder(km, skips)

    output1 = self.conv(decoded1)

    return output1, skips[0], bn, km, decoded1

class Unet_block(nn.Module):
    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                ]
            )
        )


# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet_koopman(in_channels=3, out_channels=3).to(device)

    # Example input tensor (batch size, channels, height, width)
    input_tensor = torch.randn(3, 3, 256, 256).to(device)
    output = model(input_tensor)

    print("Output shape:", output.shape)
