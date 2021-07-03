import logging
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
from .generate_training_data import RandomStrokesDataset
from .eval_net import eval_net

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


def train_net(
    net,
    device,
    total_samples=10000,
    max_strokes=5,
    size_px=512,
    epochs=5,
    batch_size=1,
    lr=0.001,
    val_portion=0.1,
    save_cp_dir=None,
    img_scale=1,
):

    n_val = int(total_samples * val_portion)
    n_train = total_samples - n_val
    train_loader = DataLoader(
        RandomStrokesDataset(n_train, max_strokes=max_strokes, size_px=size_px),
        batch_size=batch_size,
        num_workers=2,
    )
    val_loader = DataLoader(
        RandomStrokesDataset(n_val, max_strokes=max_strokes, size_px=size_px),
        batch_size=batch_size,
        num_workers=2,
    )

    writer = SummaryWriter(comment=f"LR_{lr}_BS_{batch_size}_SCALE_{img_scale}")
    global_step = 0

    logging.info(
        f"""Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Device:          {device.type}
        Images scaling:  {img_scale}
    """
    )

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min" if net.n_classes > 1 else "max", patience=2
    )
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(
            total=n_train, desc=f"Epoch {epoch + 1}/{epochs}", unit="img"
        ) as pbar:
            for batch in train_loader:
                imgs = batch["image"]
                true_masks = batch["mask"]
                assert imgs.shape[1] == net.n_channels, (
                    f"Network has been defined with {net.n_channels} input channels, "
                    f"but loaded images have {imgs.shape[1]} channels. Please check that "
                    "the images are loaded correctly."
                )

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar("Loss/train", loss.item(), global_step)

                pbar.set_postfix(**{"loss (batch)": loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (10 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace(".", "/")
                        writer.add_histogram(
                            "weights/" + tag, value.data.cpu().numpy(), global_step
                        )
                        writer.add_histogram(
                            "grads/" + tag, value.grad.data.cpu().numpy(), global_step
                        )
                    val_score = eval_net(net, val_loader, device)
                    scheduler.step(val_score)
                    writer.add_scalar(
                        "learning_rate", optimizer.param_groups[0]["lr"], global_step
                    )

                    if net.n_classes > 1:
                        logging.info("Validation cross entropy: {}".format(val_score))
                        writer.add_scalar("Loss/test", val_score, global_step)
                    else:
                        logging.info("Validation Dice Coeff: {}".format(val_score))
                        writer.add_scalar("Dice/test", val_score, global_step)

                    writer.add_images("images", imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images("masks/true", true_masks, global_step)
                        writer.add_images(
                            "masks/pred", torch.sigmoid(masks_pred) > 0.5, global_step
                        )

        if save_cp_dir:
            try:
                os.mkdir(save_cp_dir)
                logging.info("Created checkpoint directory")
            except OSError:
                pass
            torch.save(
                net.state_dict(), os.path.join(save_cp_dir, f"CP_epoch{epoch + 1}.pth")
            )
            logging.info(f"Checkpoint {epoch + 1} saved !")

    writer.close()
