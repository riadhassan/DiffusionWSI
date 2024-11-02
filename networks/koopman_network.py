import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_init_(n_units, std=1):
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]
    return Omega
# Define Koopman Layer
class KoopmanLayer(nn.Module):
    def __init__(self, in_channels, hidden_dim):
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


# Define U-Net Building Blocks
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# Define Koopman-Enhanced U-Net
class KoopmanUNet(nn.Module):
    def __init__(self, n_channels, n_classes, hidden_dim=128):
        super(KoopmanUNet, self).__init__()
        self.in_conv = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # Integrate Koopman Layer here for transformation
        self.koopman_layer = KoopmanLayer(1024, hidden_dim)

        # self.down4 = Down(512, 512)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.out_conv = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Apply Koopman Layer transformation on the bottleneck features
        k = self.koopman_layer(x5)

        x = self.up1(k, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        first_layer_f = torch.mean(x1, dim=1, keepdim=True)
        before_koopman_f = torch.mean(x5, dim=1, keepdim=True)
        koopman_f = torch.mean(k, dim=1, keepdim=True)
        last_layer_f = torch.mean(x, dim=1, keepdim=True)
        return logits, first_layer_f, before_koopman_f, koopman_f, last_layer_f


# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KoopmanUNet(n_channels=3, n_classes=3).to(device)

    # Example input tensor (batch size, channels, height, width)
    input_tensor = torch.randn(1, 3, 512, 512).to(device)
    output = model(input_tensor)

    print("Output shape:", output.shape)
