from datasets import load_dataset
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch


class PlantVillageDataLoader:
    def __init__(self):
        self.dataset_name = "DScomp380/plant_village"
        self.class_names = None

    def load_data(self):
        dataset = load_dataset(self.dataset_name)
        return dataset
    
    def get_class_names(self):
        if self.class_names is None:
            dataset = self.load_data()
            label_feature = dataset["train"].features["label"]
            self.class_names = label_feature.names
        return self.class_names
    
    def get_subset_split(self, fraction=0.3, seed=42):
        dataset = self.load_data()
        full_train = dataset["train"]

        # Take 30% subset
        total = len(full_train)
        subset_size = int(fraction * total)
        subset = full_train.shuffle(seed=seed).select(range(subset_size))

        # Split: 70% train, 15% val, 15% test
        train_size = int(0.7 * len(subset))
        val_size = int(0.15 * len(subset))
        test_size = len(subset) - train_size - val_size
        
        train_ds, val_ds, test_ds = random_split(
            subset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(seed)
        )

        return train_ds, val_ds, test_ds
    
    def get_base_transform(self):
        return transforms.Compose([
            transforms.Resize((256, 256)), 
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def get_train_transform(self):
        #filler - can be extended for data augmentation
        return self.get_base_transform()
    
    def prepare_batch(self, batch):
        transform = self.get_base_transform()
        images = torch.stack([transform(item["image"]) for item in batch])
        labels = torch.tensor([item["label"] for item in batch])
        return {"image": images, "label": labels}
    
    def get_dataloaders(self, batch_size=64, num_workers=4):
        train_ds, val_ds, test_ds = self.get_subset_split()

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                                num_workers=num_workers, collate_fn=self.prepare_batch)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, collate_fn=self.prepare_batch)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, 
                                num_workers=num_workers, collate_fn=self.prepare_batch)

        return {"train": train_loader, "val": val_loader, "test": test_loader}

if __name__ == "__main__":
    loader = PlantVillageDataLoader()
    
    # Get class names first
    class_names = loader.get_class_names()
    print(f"Number of classes: {len(class_names)}")
    print("Class names:", class_names)
    
    # Then get dataloaders
    loaders = loader.get_dataloaders(batch_size=64)
    
    print("\nTesting dataloaders...")
    batch = next(iter(loaders["train"]))
    print(f"Batch image shape: {batch['image'].shape}")
    print(f"Batch labels shape: {batch['label'].shape}")
    print(f"Train batches: {len(loaders['train'])}")
    print(f"Val batches: {len(loaders['val'])}")
    print(f"Test batches: {len(loaders['test'])}")