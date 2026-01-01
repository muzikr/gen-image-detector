import torch
import pandas as pd
import cv2

class FrameDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file="faces_data/face_data.csv", transform=None):
        self.transform = transform
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = cv2.imread(row["image_path"])
        if image is None:
            raise ValueError(f"Image not found at path: {row['image_path']}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(row["label"], dtype=torch.long)
        return image, label
