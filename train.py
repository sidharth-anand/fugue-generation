import os

import tqdm
import argparse


import torch
import torch.utils.tensorboard


import numpy as np

from data.encode import get_encoding

from model.dataset import MIDIDataset
from model.model import TransformerModel


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path", type=str, default="datasets/baroque_small_trainset_multipart"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="random",
        choices=["random", "voice"],
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--embedding_size", type=int, default=16)
    parser.add_argument("--head_size", type=int, default=8)
    parser.add_argument("--hidden_size", type=int, default=16)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)

    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--log-name", type=str, default="transformer")

    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--save-name", type=str, default="transformer")

    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--eval-interval", type=int, default=1)

    return parser.parse_args()


def masking_loss(criterion, input, output, mask, vocab_size):
    start = 0

    loss = torch.zeros(1)

    for i, size in enumerate(vocab_size):
        masked_indices = torch.argwhere(mask[:, :, i].flatten()).flatten()

        prediction = output[:, :, start:start+size].view(-1, size)[masked_indices, :]
        target = input[:, :, i].flatten()[masked_indices]

        if masked_indices.nelement() == 0:
            continue
    
        loss += criterion(
            input=prediction,
            target=target,
        )

        start += size


    return loss


def split(length):
    return [
        int(length * 0.8),
        int(length * 0.1),
        length - int(length * 0.8) - int(length * 0.1),
    ]


def collate_fn(batch):
    return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)


def step(transformer, data, task, criterion, encoding, optimizer, no_grad=False):
    input = data

    if not no_grad:
        optimizer.zero_grad()

    transformer.train(not no_grad)

    outputs, mask = transformer(
        input,
        mask=task,
    )

    loss = masking_loss(
        criterion,
        input,
        outputs,
        mask,
        np.array(encoding["n_tokens"]) + 1,
    )

    if not no_grad:
        loss.backward()
        optimizer.step()

    return loss.item()


def main():
    args = parse_args()

    encoding = get_encoding()

    dataset = MIDIDataset(args.path)

    train_loader, val_loader, test_loader = [
        torch.utils.data.DataLoader(
            set,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        for set in torch.utils.data.random_split(
            dataset,
            split(len(dataset)),
        )
    ]

    transformer = TransformerModel(
        embedding_size=args.embedding_size,
        head_size=args.head_size,
        hidden_size=args.hidden_size,
        layers=args.layers,
        dropout=args.dropout,
    )

    if args.resume is not None:
        transformer.load_state_dict(torch.load(args.resume), strict=False)

    writer = torch.utils.tensorboard.SummaryWriter(
        log_dir=f"{args.log_dir}/{args.log_name}_{args.task}"
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(transformer.parameters(), lr=args.lr)

    current_best = float("inf")

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    for epoch in range(args.epochs):
        transformer.train()

        for i, data in enumerate(train_loader):
            loss = step(transformer, data, args.task, criterion, encoding, optimizer)

            writer.add_scalar("Training Loss", loss, epoch * len(train_loader) + i)

            print(f"Epoch {epoch}, Batch {i}, Loss: {loss}")

        if epoch % args.eval_interval == 0:
            running_loss = 0

            with torch.no_grad():
                transformer.eval()

                for i, data in enumerate(val_loader):
                    running_loss += step(
                        transformer,
                        data,
                        args.task,
                        criterion,
                        encoding,
                        optimizer,
                        no_grad=True,
                    )

            writer.add_scalar(
                "Validation Loss",
                running_loss / len(val_loader),
                epoch * len(train_loader),
            )

            if running_loss < current_best:
                current_best = running_loss

                torch.save(
                    transformer.state_dict(),
                    f"{args.save_dir}/{args.save_name}_{args.task}.model",
                )

    transformer.eval()

    running_loss = 0

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            running_loss += step(
                transformer,
                data,
                args.task,
                criterion,
                encoding,
                optimizer,
                no_grad=True,
            )

    writer.add_scalar(
        "Test Loss",
        running_loss / len(test_loader),
        epoch * len(train_loader),
    )

    print(f"Test Loss: {running_loss / len(test_loader)}")

    writer.close()


if __name__ == "__main__":
    main()
