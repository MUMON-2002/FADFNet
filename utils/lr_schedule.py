import math
from collections import Counter
from torch.optim.lr_scheduler import _LRScheduler
import torch


class MultiStepRestartLR(_LRScheduler):

    def __init__(self,
                 optimizer,
                 milestones,
                 gamma=0.1,
                 restarts=(0, ),
                 restart_weights=(1, ),
                 last_epoch=-1):
        
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.restarts = restarts
        self.restart_weights = restart_weights

        assert len(self.restarts) == len(self.restart_weights), 'restarts and their weights do not match.'
        super(MultiStepRestartLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):

        if self.last_epoch in self.restarts:
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [
                group['initial_lr'] * weight
                for group in self.optimizer.param_groups
            ]
        
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        
        return [
            group['lr'] * self.gamma**self.milestones[self.last_epoch]
            for group in self.optimizer.param_groups
        ]

class LinearLR(_LRScheduler):

    def __init__(self,
                 optimizer,
                 total_iter,
                 last_epoch=-1):
        self.total_iter = total_iter
        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):

        process = self.last_epoch / self.total_iter
        weight = (1 - process)

        return [weight * group['initial_lr'] for group in self.optimizer.param_groups]

class VibrateLR(_LRScheduler):

    def __init__(self,
                 optimizer,
                 total_iter,
                 last_epoch=-1):
        self.total_iter = total_iter
        super(VibrateLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):

        process = self.last_epoch / self.total_iter

        f = 0.1
        if process < 3 / 8:
            f = 1 - process * 8 / 3
        elif process < 5 / 8:
            f = 0.2

        T = self.total_iter // 80
        Th = T // 2
        t = self.last_epoch % T

        f2 = t / Th
        if t >= Th:
            f2 = 2 - f2
        weight = f * f2

        if self.last_epoch < Th:
            weight = max(0.1, weight)

        return [weight * group['initial_lr'] for group in self.optimizer.param_groups]

def get_position_from_periods(iteration, cumulative_period):

    for i, period in enumerate(cumulative_period):
        if iteration <= period:
            return i


class CosineAnnealingRestartLR(_LRScheduler):

    def __init__(self,
                 optimizer,
                 periods,
                 restart_weights=(1, ),
                 eta_min=0,
                 last_epoch=-1):
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_min = eta_min
        assert (len(self.periods) == len(self.restart_weights)), 'periods and restart_weights should have the same length.'
        self.cumulative_period = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]
        super(CosineAnnealingRestartLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):

        idx = get_position_from_periods(self.last_epoch,
                                        self.cumulative_period)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period = self.periods[idx]

        return [
            self.eta_min + current_weight * 0.5 * (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * (
                (self.last_epoch - nearest_restart) / current_period)))
            for base_lr in self.base_lrs
        ]

class CosineAnnealingRestartCyclicLR(_LRScheduler):

    def __init__(self,
                 optimizer,
                 periods,
                 restart_weights=(1, ),
                 eta_mins=(0, ),
                 last_epoch=-1):
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_mins = eta_mins
        assert (len(self.periods) == len(self.restart_weights)), 'periods and restart_weights should have the same length.'
        self.cumulative_period = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]
        super(CosineAnnealingRestartCyclicLR, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        idx = get_position_from_periods(self.last_epoch, self.cumulative_period)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period = self.periods[idx]
        eta_min = self.eta_mins[idx]

        return [
            eta_min + current_weight * 0.5 * (base_lr - eta_min) *
            (1 + math.cos(math.pi * (
                (self.last_epoch - nearest_restart) / current_period)))
            for base_lr in self.base_lrs
        ]
    

class CosineAnnealingLR(_LRScheduler):
    
    def __init__(self,
                 optimizer,
                 T_max,
                 eta_min=0,
                 last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineAnnealingLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [
            self.eta_min + 0.5 * (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * (self.last_epoch / self.T_max)))
            for base_lr in self.base_lrs
        ]


class ConstantLR(_LRScheduler):
    def get_lr(self):
        return list(self.base_lrs)


class HybridLRScheduler(_LRScheduler):
    def __init__(self, 
                 optimizer, 
                 total_iters,
                 warmup_iters=0, 
                 eta_min=1e-6,   
                 last_epoch=-1):
        
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        self.eta_min = eta_min
        
        assert self.total_iters >= self.warmup_iters
        
        self.cosine_iters = total_iters - warmup_iters
            
        super(HybridLRScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            return list(self.base_lrs)
        else:
            cosine_step = self.last_epoch - self.warmup_iters
            
            if self.cosine_iters <= 0:
                return [self.eta_min for _ in self.base_lrs]
            
            cosine_step = min(cosine_step, self.cosine_iters)
            return [
                self.eta_min + 0.5 * (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * cosine_step / self.cosine_iters))
                for base_lr in self.base_lrs
            ]


