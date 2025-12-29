import torch
import os
from tqdm import tqdm

def get_dataframe() -> list[tuple[str, int]]:
    data = []
    for folder in [
        ("data\\archive (3)\\Celeb-real",0),
        ("data\\archive (3)\\Celeb-synthesis",1),
        ("data\\archive (3)\\Youtube-real",0),]:
        for file in tqdm(os.listdir(folder[0])):
            if file.endswith('.mp4'):
                video_path = os.path.join(folder[0], file)
                data.append((video_path, folder[1]))
    print(data[0])
    return data

class FrameDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.transform = transform
        self.dataset = data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    
if __name__ == "__main__":
    get_dataframe()