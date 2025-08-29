from torch.utils.data import Dataset


class TVDataset(Dataset):
    def __init__(self, dataset: Dataset, key: str | None = None):

        self._dataset = dataset
        self._key = key

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        item = self._dataset[idx]
        if self._key is not None:
            return item[self._key]
        try:
            return item[0]
        except Exception as e:
            return item
    
    @property
    def dataset(self):
        return self._dataset

    @property
    def key(self):
        return self._key