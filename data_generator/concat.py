from torch.utils.data import Dataset, RandomSampler, SequentialSampler, BatchSampler, DataLoader
import numpy as np

from util.misc import nested_tensor_from_tensor_list
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
    
def concat_datasets(train_datasets, val_datasets, args):
    # --- Train Data Loader ---
    if len(train_datasets) == 0:
        data_loader_train = None 
    else:
        concat_train_ds = ConcatDataset(*train_datasets)
        sampler_train = RandomSampler(concat_train_ds)

        pin_memory = args.PIN_MEMORY
        data_loader_train = DataLoader(concat_train_ds, args.BATCH_SIZE, sampler=sampler_train, pin_memory=pin_memory,
                                    collate_fn=collate_wrapper, num_workers=args.NUM_WORKERS, worker_init_fn=set_worker_sharing_strategy)
    
    # --- Test Data Loaders ---
    dl_val_list = []
    for val_ds in val_datasets:
        sampler_val = RandomSampler(val_ds)
        data_loader_val = DataLoader(val_ds, args.BATCH_SIZE, sampler=sampler_val,
                                    drop_last=False, collate_fn=collate_wrapper, num_workers=args.NUM_WORKERS, pin_memory=pin_memory,
                                    worker_init_fn=set_worker_sharing_strategy)
        dl_val_list.append(data_loader_val)
    
    return data_loader_train, dl_val_list
