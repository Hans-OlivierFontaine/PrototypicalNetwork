from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class PrototypicalDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.root_dir / "images" / self.data.iloc[idx, 0]
        image = Image.open(img_name.__str__())

        if self.transform:
            image = self.transform(image)

        label = self.data.iloc[idx, 1]
        return image, label
