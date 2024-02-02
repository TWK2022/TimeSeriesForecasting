import math
import copy
import torch


class model_ema:
    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        self.ema = copy.deepcopy(self._get_model(model)).eval()  # FP32 EMA
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)
            state_dict = self._get_model(model).state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * state_dict[k].detach()

    def _get_model(self, model):
        if type(model) in (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel):
            return model.module
        else:
            return model
