import torch
from hanzi_font_deconstructor.common.train_net import train_net
from hanzi_font_deconstructor.model.UNet import UNet

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    net = UNet(
        n_channels=1,
        n_classes=5,
        bilinear=False,
    )
    net.to(device=device)
    train_net(
        net,
        device,
        total_samples=10000,
        max_strokes=5,
        size_px=512,
        epochs=5,
        batch_size=1,
        lr=0.001,
        val_portion=0.1,
    )
