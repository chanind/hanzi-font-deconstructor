import torch
from .utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
from os import path
from .RandomStrokesDataset import RandomStrokesDataset
from .Generator import Generator
from .Discriminator import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image

torch.backends.cudnn.benchmark = True

L1_LAMBDA = 100


def inner_train_loop(
    disc,
    gen,
    device,
    loader,
    opt_disc,
    opt_gen,
    l1_loss,
    bce,
    g_scaler,
    d_scaler,
):
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):
        x = x.to(device)
        y = y.to(device)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * L1_LAMBDA
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )


def train_net(
    gen: Generator,
    disc: Discriminator,
    device,
    lr=2e-4,
    save_checkpoint_dir=None,
    batch_size=16,
    num_workers=2,
    num_epochs=100,
    samples_per_epoch=2000,
    val_portion=0.01,
    size_px=256,
):
    # disc = Discriminator(in_channels=3).to(config.DEVICE)
    # gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    opt_disc = optim.Adam(
        disc.parameters(),
        lr=lr,
        betas=(0.5, 0.999),
    )
    opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    # if config.LOAD_MODEL:
    #     load_checkpoint(
    #         config.CHECKPOINT_GEN,
    #         gen,
    #         opt_gen,
    #         config.LEARNING_RATE,
    #     )
    #     load_checkpoint(
    #         config.CHECKPOINT_DISC,
    #         disc,
    #         opt_disc,
    #         config.LEARNING_RATE,
    #     )

    train_dataset = RandomStrokesDataset(samples_per_epoch, size_px=size_px)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    val_dataset = RandomStrokesDataset(
        int(samples_per_epoch * val_portion), size_px=size_px
    )
    val_loader = DataLoader(val_dataset, batch_size=1)

    for epoch in range(num_epochs):
        inner_train_loop(
            disc,
            gen,
            device,
            train_loader,
            opt_disc,
            opt_gen,
            L1_LOSS,
            BCE,
            g_scaler,
            d_scaler,
        )

        if save_checkpoint_dir and epoch % 5 == 0:
            save_checkpoint(
                gen,
                opt_gen,
                filename=path.join(save_checkpoint_dir, "gen_cp.pth"),
            )
            save_checkpoint(
                disc,
                opt_disc,
                filename=path.join(save_checkpoint_dir, "disc_cp.pth"),
            )

        save_some_examples(gen, val_loader, epoch, device=device, folder="evaluation")
