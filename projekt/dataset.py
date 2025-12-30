import torch
import pandas as pd
import cv2

class FrameDataset(torch.utils.data.Dataset):
    def __init__(self,  transform=None, csv_file: str ="faces_data/face_data.csv"):
        self.transform = transform
        self.df= pd.read_csv(csv_file)
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
       # print(self.dataset.head())
        label = self.df.iloc[idx]["label"]
        image = cv2.imread(self.df.iloc[idx]["image_path"])
        if image is None:
            raise ValueError(f"Image not found at path: {self.df.iloc[idx]['image_path']}")
        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0  # Normalize to [0, 1]
        if self.transform:
            image = self.transform(image)
        return image, label
    