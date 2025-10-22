from datasets import load_dataset
from torch.utils.data import random_split
import torch

class PlantVillageDataLoader:
    def __init__(self):
        self.dataset_name = "DScomp380/plant_village"

    def load_data(self):
        dataset = load_dataset(self.dataset_name)
        return dataset
    
    def get_subset_split(self, fraction=0.3, seed=42):
        # Load full dataset
        dataset = self.load_data()
        full_train = dataset["train"]

        # Take 30% subset
        total = len(full_train)
        subset_size = int(fraction * total)
        subset = full_train.shuffle(seed=seed).select(range(subset_size))

        # 70/15/15 split
        train_size = int(0.7 * len(subset))
        val_size = int(0.15 * len(subset))
        test_size = len(subset) - train_size - val_size

        train_ds, val_ds, test_ds = random_split(
            subset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(seed)
        )

        return train_ds, val_ds, test_ds
        

        

if __name__ == "__main__":
    loader = PlantVillageDataLoader()
    ds = loader.load_data()
    print(f"Full train: {len(ds['train'])}")

    train, val, test = loader.get_subset_split()
    print(f"Subset split: {len(train)} / {len(val)} / {len(test)}")
