import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from dataset import RetinaDataset
from model import get_model


# Checks validation accuracy after each epoch.
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = RetinaDataset(args.csv_file, args.image_dir)
    if args.val_csv and args.val_image_dir:
        val_set = RetinaDataset(args.val_csv, args.val_image_dir)
    else:
        val_size = int(0.2 * len(train_set))
        train_size = len(train_set) - val_size
        train_set, val_set = random_split(train_set, [train_size, val_size],
                                          generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    model = get_model(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_acc = 0.0

    # Main training loop for 20 epochs by default.
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch + 1}/{args.epochs} | "
              f"Loss: {running_loss / len(train_loader):.4f} | "
              f"Val Acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            print(f"Saved best model to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", default="train.csv")
    parser.add_argument("--image_dir", default="train_images")
    parser.add_argument("--val_csv", default=None)
    parser.add_argument("--val_image_dir", default=None)
    parser.add_argument("--save_path", default="best_model.pth")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    main(parser.parse_args())
