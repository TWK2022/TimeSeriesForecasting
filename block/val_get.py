import cv2
import tqdm
import torch
import albumentations
from block.metric_get import metric


def val_get(args, data_dict, model, loss):
    with torch.no_grad():
        model.eval().to(args.device, non_blocking=args.latch)
        val_dataloader = torch.utils.data.DataLoader(torch_dataset(args, data_dict), batch_size=args.batch,
                                                     shuffle=False, drop_last=False, pin_memory=args.latch)
        val_pred = []
        val_true = []
        for item, (val_batch, true_batch) in enumerate(tqdm.tqdm(val_dataloader)):
            val_batch = val_batch.to(args.device, non_blocking=args.latch)
            val_pred.extend(model(val_batch).detach().cpu())
            val_true.extend(true_batch)
        val_pred = torch.stack(val_pred, dim=0)
        val_true = torch.stack(val_true, dim=0)
        val_loss = loss(val_pred, val_true)
        mae, mse = metric(val_pred, val_true)
        print('\n| 验证集:{} | val_loss:{:.4f} | val_mae:{:.4f} | val_mse:{:.4f} |\n'
              .format(len(data_dict['val_input']), val_loss, mae, mse))
    return val_loss, mae, mse


class torch_dataset(torch.utils.data.Dataset):
    def __init__(self, args, data_dict):
        self.args = args
        self.data_dict = data_dict

    def __len__(self):
        return len(self.data_dict['val_input']) - self.args.input_size - self.args.output_size + 1

    def __getitem__(self, index):
        boundary = index + self.args.input_size
        series = self.data_dict['val_input'][index:boundary]  # 输入数据
        series = torch.tensor(series, dtype=torch.float32).permute(1, 0)  # 转换为tensor
        label = self.data_dict['val_output'][boundary:boundary + self.args.output_size]  # 输出标签
        label = torch.tensor(label, dtype=torch.float32).permute(1, 0)  # 转换为tensor
        return series, label
