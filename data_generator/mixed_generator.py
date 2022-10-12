from torch.utils.data import Dataset
import np
import bisect

from data_generator.Objects365 import get_365_data_generator, build_365_dataset
from data_generator.AVD import get_avd_data_generator, build_AVD_dataset
from data_generator.GMU_kitchens import get_gmu_data_generator, build_GMU_dataset


class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets
        self.data_lengths = [len(d) for d in self.datasets]
        self.cumulative_lengths = np.cumsum(self.data_lengths)

    def __len__(self):
        return sum(len(d) for d in self.datasets)

    def __getitem__(self, idx):
        indices = self.cumulative_lengths - idx
        # Get first non-negative index
        min_idx = bisect.bisect_left(indices, 0)
        idx = idx - self.cumulative_lengths[min_idx-1]
        return self.datasets[min_idx][idx]

    