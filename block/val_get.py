import tqdm
import torch
from block.metric_get import metric


def val_get(args, data_dict, model, loss):
    with torch.no_grad():
        model.eval().to(args.device, non_blocking=args.latch)
        dataloader = torch.utils.data.DataLoader(torch_dataset(args, data_dict), batch_size=args.batch,
                                                 shuffle=False, drop_last=False, pin_memory=args.latch,
                                                 num_workers=args.num_worker)
        pred = []
        true = []
        for item, (series_batch, true_batch) in enumerate(tqdm.tqdm(dataloader)):
            series_batch = series_batch.to(args.device, non_blocking=args.latch)
            pred.extend(model(series_batch).detach().cpu())
            true.extend(true_batch)
        # 计算指标
        pred = torch.stack(pred, dim=0)
        true = torch.stack(true, dim=0)
        val_loss = loss(pred, true)
        mae, mse = metric(pred, true)
        print('\n| 验证集:{} | val_loss:{:.4f} | val_mae:{:.4f} | val_mse:{:.4f} |'
              .format(len(data_dict['val_input']), val_loss, mae, mse))
        # 分别计算每个输出的指标
        for i in range(pred.shape[1]):
            column = args.output_column[i]
            _mae, _mse = metric(pred[:, i], true[:, i])
            print('| {} | val_mae:{:.4f} | val_mse:{:.4f} |'
                  .format(column, _mae, _mse))
    return val_loss, mae, mse


class torch_dataset(torch.utils.data.Dataset):
    def __init__(self, args, data_dict):
        self.data_dict = data_dict
        self.input_size = args.input_size
        self.output_size = args.output_size

    def __len__(self):
        return len(self.data_dict['val_input']) - self.input_size - self.output_size + 1

    def __getitem__(self, index):
        boundary = index + self.input_size
        series = self.data_dict['val_input'][index:boundary]  # 输入数据
        series = torch.tensor(series, dtype=torch.float32).permute(1, 0)  # 转换为tensor
        label = self.data_dict['val_output'][boundary:boundary + self.output_size]  # 输出标签
        label = torch.tensor(label, dtype=torch.float32).permute(1, 0)  # 转换为tensor
        return series, label
