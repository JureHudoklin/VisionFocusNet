from torch.utils.data import Dataset, RandomSampler, SequentialSampler, BatchSampler, DataLoader
import numpy as np
import bisect

from util.misc import nested_tensor_from_tensor_list
from data_generator.Objects365 import get_365_data_generator, build_365_dataset
from data_generator.AVD import get_avd_data_generator, build_AVD_dataset
from data_generator.GMU_kitchens import get_gmu_data_generator, build_GMU_dataset


class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        super().__init__()
        self.datasets = datasets
        self.data_lengths = [len(d) for d in self.datasets]
        self.cumulative_lengths = np.cumsum(self.data_lengths)

    def __len__(self):
        return sum(len(d) for d in self.datasets)

    def __getitem__(self, idx):
        indices = self.cumulative_lengths - idx
        # Get first non-negative index
        min_idx = bisect.bisect_left(indices, 0)
        if min_idx > 0:
            idx = idx - self.cumulative_lengths[min_idx - 1]
        ds = self.datasets[min_idx]
        img, tgt_img, target = ds[idx]

        return img, tgt_img, target

def collate_fn( batch):
        batch = list(zip(*batch))
        batch[0] = nested_tensor_from_tensor_list(batch[0])
        batch[1] = nested_tensor_from_tensor_list(batch[1])
        return tuple(batch)    

def get_concat_dataset(args):
    dataset_avd_train = build_AVD_dataset(image_set='train', args=args)
    dataset_gmu_train = build_GMU_dataset(image_set='train', args=args)
    dataset_365_train = build_365_dataset(image_set='train', args=args)
    
    dataset_avd_val = build_AVD_dataset(image_set='val', args=args)
    dataset_gmu_val = build_GMU_dataset(image_set='val', args=args)
    dataset_365_val = build_365_dataset(image_set='val', args=args)

    dataset_train = ConcatDataset(dataset_avd_train, dataset_gmu_train, dataset_365_train)
    dataset_val = ConcatDataset(dataset_avd_val, dataset_gmu_val, dataset_365_val)
    
    sampler_train = RandomSampler(dataset_train)
    sampler_val = SequentialSampler(dataset_val)

    # batch_sampler_train = BatchSampler(
    #     sampler_train, args.BATCH_SIZE, drop_last=True)
    batch_sampler_val = BatchSampler(
        sampler_val, args.BATCH_SIZE, drop_last=True)

    data_loader_train = DataLoader(dataset_train, args.BATCH_SIZE, sampler=sampler_train,
                                   collate_fn=collate_fn, num_workers=args.NUM_WORKERS)
    data_loader_val = DataLoader(dataset_val, args.BATCH_SIZE, sampler=batch_sampler_val,
                                 drop_last=False, collate_fn=collate_fn, num_workers=args.NUM_WORKERS)

    
    return data_loader_train, data_loader_val