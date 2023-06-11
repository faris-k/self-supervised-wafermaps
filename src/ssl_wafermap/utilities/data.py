# Deprecated: new import lives in ssl_wafermap.data

import torch
from torch.utils.data import Dataset


class WaferMapDataset(Dataset):
    """Dataset for wafermaps.

    Parameters
    ----------
    X : pd.Series
        Series of wafermaps
    y : pd.Series
        Series of labels
    transform : torchvision.transforms, optional
        Transformations to apply to each wafermap, by default None
    """

    def __init__(self, X, y, transform=None):
        # self.data = pd.concat([X, y], axis="columns")
        # All resizing is done in augmentations, so we have tensors/arrays of different sizes
        # Because of this, just create a list of tensors
        self.X_list = [torch.tensor(ndarray) for ndarray in X]
        self.y_list = [torch.tensor(ndarray) for ndarray in y]
        self.transform = transform

    def __getitem__(self, index):
        x = self.X_list[index]
        y = self.y_list[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.X_list)


class TensorDataset(Dataset):
    """Simple dataset for tensors.

    Parameters
    ----------
    X : np.ndarray
        Array of features
    y : np.ndarray
        Array of labels
    """

    def __init__(self, X, y):
        self.X = torch.tensor(X).type(torch.float32)
        self.y = torch.tensor(y).type(torch.LongTensor)
        self.data = (X, y)

    def __getitem__(self, index):
        x = self.X[index]
        y = self.y[index]
        return x, y

    def __len__(self):
        return len(self.X)
