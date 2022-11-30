from torch.utils.data import Dataset, RandomSampler, SequentialSampler, BatchSampler, DataLoader
import numpy as np
import bisect

from util.misc import nested_tensor_from_tensor_list
from data_generator.coco import build_dataset
from data_generator.mix_data_generator import build_MIX_dataset
from util.data_utils import collate_wrapper, set_worker_sharing_strategy


class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        super().__init__()
        self.datasets = datasets
        self.data_lengths = [len(d) for d in self.datasets]
        self.cumulative_lengths = np.cumsum(self.data_lengths)

    def __len__(self):
        return sum(len(d) for d in self.datasets)

    def __getitem__(self, idx):
        ds_lens = self.data_lengths
        ds_idx = np.argmax(np.cumsum(ds_lens) > idx)
        idx = idx - int(np.sum(ds_lens[:ds_idx]))
        
        ds = self.datasets[ds_idx]
        img, tgt_img, target = ds[idx]

        return img, tgt_img, target

def get_concat_dataset(args):
    coco_train = build_dataset("train", args)
    mix_train = build_MIX_dataset("train", args.TRAIN_DATASETS, args)
    
    dataset_train = ConcatDataset(coco_train, mix_train)
    sampler_train = RandomSampler(dataset_train)

    pin_memory = args.PIN_MEMORY
    data_loader_train = DataLoader(dataset_train, args.BATCH_SIZE, sampler=sampler_train, pin_memory=pin_memory,
                                   collate_fn=collate_wrapper, num_workers=args.NUM_WORKERS, worker_init_fn=set_worker_sharing_strategy)
    
    dl_val_list = []
    for val_pth in args.TEST_DATASETS:
        ds_val = build_MIX_dataset("val", [val_pth], args)
        sampler_val = RandomSampler(ds_val)
        
        data_loader_val = DataLoader(ds_val, args.BATCH_SIZE, sampler=sampler_val,
                                    drop_last=False, collate_fn=collate_wrapper, num_workers=args.NUM_WORKERS, pin_memory=pin_memory,
                                    worker_init_fn=set_worker_sharing_strategy)
        
        dl_val_list.append(data_loader_val)
    
    return data_loader_train, dl_val_list