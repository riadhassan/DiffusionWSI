import torch
import torch.nn as nn

# Define a simple encoder-decoder CNN for image translation
class ImageTranslationCNN(nn.Module):
    def __init__(self):
        super(ImageTranslationCNN, self).__init__()

        # Encoder: down-sampling
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # Output: 64 x 16 x 16
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: 128 x 8 x 8
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Output: 256 x 4 x 4
            nn.ReLU(inplace=True)
        )

        # Decoder: up-sampling
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # Output: 128 x 8 x 8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Output: 64 x 16 x 16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # Output: 3 x 32 x 32
            nn.Tanh()  # Output in the range [-1, 1]
        )

    def forward(self, x):
        # Pass through the encoder
        encoded = self.encoder(x)

        # Pass through the decoder
        decoded = self.decoder(encoded)

        return decoded


if __name__ == "__main__":
    model = ImageTranslationCNN()

    # Example input: a batch of 32x32 RGB images (batch size of 8)
    dummy_input = torch.randn(8, 3, 32, 32)  # Random noise as input
    output = model(dummy_input)

    # Print the output shape to verify the network works
    print(f"Output shape: {output.shape}")