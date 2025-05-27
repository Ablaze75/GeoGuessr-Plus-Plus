import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models
from src.preprocess import get_image_transform, split_dataset, discover_images

def main(batch_size=32, epochs=5, lr=1e-4, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) Discover & split
    df = discover_images()
    train_df, val_df = split_dataset(df, test_size=0.2)

    # 2) Transforms
    train_tf, val_tf = get_image_transform()

    # 3) Build ImageFolder & filter by split
    full_ds = datasets.ImageFolder(
        root=os.path.join(os.path.dirname(__file__), '..', 'compressed_dataset'),
        transform=None
    )
    path_to_idx = {p: idx for idx, (p,_) in enumerate(full_ds.samples)}
    train_idx = [path_to_idx[p] for p in train_df['image_path']]
    val_idx   = [path_to_idx[p] for p in val_df['image_path']]

    full_ds.transform = lambda img: img
    train_ds = Subset(full_ds, train_idx)
    val_ds   = Subset(full_ds, val_idx)
    train_ds.dataset.transform = train_tf
    val_ds.dataset.transform   = val_tf

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)

    # 4) Model & head
    num_classes = len(full_ds.classes)
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # 5) Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 6) Training loop
    for epoch in range(1, epochs+1):
        model.train()
        total_loss, correct = 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct    += (out.argmax(1) == labels).sum().item()

        train_loss = total_loss / len(train_ds)
        train_acc  = correct / len(train_ds)

        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                val_loss += criterion(out, labels).item() * imgs.size(0)
                val_correct += (out.argmax(1) == labels).sum().item()
        val_loss /= len(val_ds)
        val_acc  = val_correct / len(val_ds)

        print(f"Epoch {epoch}/{epochs} | "
              f"Train loss={train_loss:.3f}, acc={train_acc:.3f} | "
              f"Val loss={val_loss:.3f}, acc={val_acc:.3f}")

    # 7) Save weights
    torch.save(model.state_dict(), "geoguessr_resnet18.pth")
    print("Saved model to geoguessr_resnet18.pth")

if __name__=="__main__":
    main()

