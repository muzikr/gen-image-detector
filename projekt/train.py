import torch
from tqdm import tqdm
from dataset import FrameDataset
from models.baseline import EfficientNetBasedModel
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import create_face_data

def validate_model(model, dataloader, criterion, device):
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, acc, f1

def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    device,
    log_every=100
):
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        seen_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
          #  print(outputs.shape)
          #  print(labels.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            seen_samples += batch_size

            if seen_samples % log_every == 0:
                pbar.set_postfix(
                    loss=f"{running_loss / seen_samples:.4f}"
                )

        train_loss = running_loss / seen_samples

        # ----- Validation -----
        val_loss, val_acc, val_f1 = validate_model(
            model, val_loader, criterion, device
        )

        print(
            f"\nEpoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

def get_split() ->tuple[FrameDataset, FrameDataset]:
    dataset = FrameDataset()
    test_videos = pd.read_csv(create_face_data.PATH_TO_VIDEOS + "\\List_of_testing_videos.txt", sep = " ")
    test_videos['video_name'] = test_videos['video_path'].apply(
        lambda x:
            create_face_data.PATH_TO_VIDEOS + \
            "\\"+\
            x.replace('/','\\')
    )
    test_videos["label"] = test_videos["label"].map({1:0,0:1}) 

    test_df = dataset.df[dataset.df['video_path'].isin(test_videos['video_name'])]
    train_df = dataset.df[~dataset.df['video_path'].isin(test_videos['video_name'])]   

    train_dataset = FrameDataset()
    train_dataset.df = train_df.reset_index(drop=True)
    test_dataset = FrameDataset()
    test_dataset.df = test_df.reset_index(drop=True)

    return train_dataset, test_dataset

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset, val_dataset = get_split()

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=16, shuffle=False
    )

    model = EfficientNetBasedModel()
    model.backbone.requires_grad_(False)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.classifier.parameters(), lr=1e-3
    )

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=10,
        device=device,
        log_every=10
    )

if __name__ == "__main__":
    get_split()
  #  exit()
    main()