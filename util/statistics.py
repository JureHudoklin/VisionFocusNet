import torch
from collections import deque


class ValueStats():
    def __init__(self, window_size = 20, fmt = None) -> None:
        self.fmt = fmt
        if self.fmt is None:
            self.fmt = "{median:.4f} ({global_avg:.4f})"
        self.window_size = window_size
        self.deque = deque(maxlen = window_size)
        self.total = 0
        self.count = 0
    
    def update(self, value, n = 1):
        self.deque.append(value)
        self.count += n
        self.total += value*n
    
    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        if d.shape[0] == 0:
            return 0
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        avg =self.total / self.count
        if isinstance(avg, torch.Tensor):
            return avg.item()
        return avg

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        val = self.deque[-1]
        if isinstance(val, torch.Tensor):
            return val.item()
        return val
    
    def __str__(self):
        return self.fmt.format(
            median=self.median,
            global_avg=self.global_avg)
    

class StatsTracker():
    def __init__(self) -> None:
        self.tracked_losses = {}
        self.tracked_stats = {}
  
    def _update_stats(self, new_values, tracked_values):
        for k, v in new_values.items():
            if isinstance(v, torch.Tensor):
                # Check that the tensor is 1d
                if v.dim() <= 1:
                    v = v.item()
                else:
                    continue
            if k not in tracked_values:
                tracked_values[k] = ValueStats()

            tracked_values[k].update(v)
            
        return tracked_values
    
    def _fmt(self):
        out = [f"{k}:{str(v)}" for k, v in self.tracked_stats.items()]
        out = "\n ".join(out)
        return out
        
    @torch.no_grad()
    def update(self, losses_dict, stats_dict):
        self._update_stats(losses_dict, self.tracked_losses)
        self._update_stats(stats_dict, self.tracked_stats)
        
    def get_stats_avg(self):
        loss_dict = {k: v.global_avg for k, v in self.tracked_losses.items()}
        stats_dict = {k: v.global_avg for k, v in self.tracked_stats.items()}
        return loss_dict, stats_dict
    
    def get_stats_current(self):
        loss_dict = {k: v.avg for k, v in self.tracked_losses.items()}
        stats_dict = {k: v.avg for k, v in self.tracked_stats.items()}
        return loss_dict, stats_dict
    
    def save_info(self, path, epoch, batch):
        l_d, s_d = self.get_stats()
        with open(path, "a") as f:
            f.write(f"E:{epoch}, Batch:{batch}, Losses: {l_d} ")
            f.write(f"Stats: {s_d} \n")
        return
    
    def __str__(self):
        return self._fmt()
    
    

@torch.no_grad()
def prec_acc_rec(output, target, topk=(1,)):
    """
    Computes the accuracy@k for the specified values of k
    
    Parameters
    ----------
    output : torch.Tensor [bs* q, 2]
    target : torch.Tensor [bs* q]
    """
    assert isinstance(output, torch.Tensor)
    if target.numel() == 0: # Check how many elements in the tensor
        return [torch.zeros([], device=output.device)]
    
    maxk = max(topk)
    batch_size = target.size(0)
    output = output.view(-1, 2)
    target = target.view(-1)
    
    output = output[:, 1]
    output_pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
    
    true_pos = torch.sum(output_pred * target)
    false_pos = torch.sum(output_pred * (1 - target))
    true_neg = torch.sum((1 - output_pred) * (1 - target))
    false_neg = torch.sum((1 - output_pred) * target)
    prec = true_pos / (true_pos + false_pos)
    prec = prec if not torch.isnan(prec) else torch.zeros_like(prec)
    acc = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    rec = true_pos / (true_pos + false_neg)
    #print(f"True Pos: {true_pos}, False Pos: {false_pos}, True Neg: {true_neg}, False Neg: {false_neg}", "Accuracy: ", acc, "Precision: ", prec)

    # _, pred = output.topk(maxk, dim = -1, largest=True, sorted=True) # [bs* q, 1]
    # pred = pred.t() # [1, bs* q]
    # correct = pred.eq(target.view(1, -1).expand_as(pred))

    # res = []
    # for k in topk:
    #     correct_k = correct[:k].view(-1).float().sum(0)
    #     res.append(correct_k.mul_(100.0 / batch_size))
    return prec*100, acc*100, rec*100

