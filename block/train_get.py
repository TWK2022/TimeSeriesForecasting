import tqdm
import wandb
import torch
from block.val_get import val_get


def train_get(args, data_dict, model_dict, loss):
    model = model_dict['model'].to(args.device, non_blocking=args.latch)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_dataloader = torch.utils.data.DataLoader(torch_dataset(args, data_dict['train_input'],
                                                                 data_dict['train_output']),
                                                   batch_size=args.batch, shuffle=True, drop_last=True,
                                                   pin_memory=args.latch, num_workers=args.num_worker)
    val_dataloader = torch.utils.data.DataLoader(torch_dataset(args, data_dict['val_input'], data_dict['val_output']),
                                                 batch_size=args.batch, shuffle=False,
                                                 drop_last=False, pin_memory=args.latch, num_workers=args.num_worker)
    for epoch in range(args.epoch):
        # 训练
        print(f'\n-----------------------第{epoch + 1}轮-----------------------')
        model.train()
        train_loss = 0  # 记录训练损失
        for item, (series_batch, true_batch) in enumerate(tqdm.tqdm(train_dataloader)):
            series_batch = series_batch.to(args.device, non_blocking=args.latch)
            true_batch = true_batch.to(args.device, non_blocking=args.latch)
            pred_batch = model(series_batch)
            loss_batch = loss(pred_batch, true_batch)
            train_loss += loss_batch.item()
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()
        train_loss = train_loss / (item + 1)
        print('\n| train_loss:{:.4f} |\n'.format(train_loss))
        # 清理显存空间
        del series_batch, true_batch, pred_batch, loss_batch
        torch.cuda.empty_cache()
        # 验证
        val_loss, mae, mse = val_get(args, val_dataloader, model, loss)
        # 保存
        if mae < 3 and mse < model_dict['val_mse']:
            model_dict['model'] = model
            model_dict['epoch'] = epoch
            model_dict['train_loss'] = train_loss
            model_dict['val_loss'] = val_loss
            model_dict['val_mae'] = mae
            model_dict['val_mse'] = mse
            torch.save(model_dict, args.save_name)
            print('\n| 保存模型:{} | val_loss:{:.4f} | val_mae:{:.4f} | val_mse:{:.4f} |\n'
                  .format(args.save_name, val_loss, mae, mse))
        # wandb
        if args.wandb:
            args.wandb_run.log({'metric/train_loss': train_loss, 'metric/val_loss': val_loss, 'metric/val_mae': mae,
                                'metric/val_mse': mse})
    return model_dict


class torch_dataset(torch.utils.data.Dataset):
    def __init__(self, args, data_input, data_output):
        self.data_input = data_input
        self.data_output = data_output
        self.input_size = args.input_size
        self.output_size = args.output_size

    def __len__(self):
        return len(self.data_input) - self.input_size - self.output_size + 1

    def __getitem__(self, index):
        boundary = index + self.input_size
        series = self.data_input[index:boundary]  # 输入数据
        series = torch.tensor(series, dtype=torch.float32).permute(1, 0)  # 转换为tensor
        label = self.data_output[boundary:boundary + self.output_size]  # 输出标签
        label = torch.tensor(label, dtype=torch.float32).permute(1, 0)  # 转换为tensor
        return series, label
