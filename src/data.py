from datasets import load_dataset

class PlantVillageDataLoader:
    def __init__(self):
        self.dataset_name = "DScomp380/plant_village"

    def load_data(self):
        dataset = load_dataset(self.dataset_name)
        return dataset

if __name__ == "__main__":
    dl = PlantVillageDataLoader()
    ds = dl.load_data()
    print(ds)
    print("Keys:", ds.keys())
    
