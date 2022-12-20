import tqdm
import torch
from block.val_get import val_get


def train_get(args, data_dict, model_dict, loss):
    model = model_dict['model']
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epoch):
        # 训练
        print('\n-----------------------------------------------')
        print('| 第{}轮 | 训练集:{} | 批量:{} | 学习率:{} |\n'
              .format(epoch + 1, len(data_dict['train_input']), args.batch, optimizer.defaults['lr']))
        model.train().to(args.device, non_blocking=args.latch)
        train_loss = 0  # 记录训练损失
        train_dataloader = torch.utils.data.DataLoader(torch_dataset(args, data_dict), batch_size=args.batch,
                                                       shuffle=True, drop_last=True, pin_memory=args.latch)
        for item, (train_batch, true_batch) in enumerate(tqdm.tqdm(train_dataloader)):
            train_batch = train_batch.to(args.device, non_blocking=args.latch)
            true_batch = true_batch.to(args.device, non_blocking=args.latch)
            pred_batch = model(train_batch)
            loss_batch = loss(pred_batch, true_batch)
            train_loss += loss_batch.item()
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()
        train_loss = train_loss / (item + 1)
        print('\n| 训练集:{} | train_loss:{:.4f} |\n'.format(len(data_dict['train_input']), train_loss))
        # 清理显存空间
        del train_batch, true_batch, pred_batch, loss_batch
        torch.cuda.empty_cache()
        # 验证
        val_loss, mae, mse = val_get(args, data_dict, model, loss)
        # 保存
        if mse < 0.5 and mse < model_dict['val_mse']:
            model_dict['model'] = model
            model_dict['epoch'] = epoch
            model_dict['train_loss'] = train_loss
            model_dict['val_loss'] = val_loss
            model_dict['val_mae'] = mae
            model_dict['val_mse'] = mse
            model_dict['input_column'] = data_dict['input_column']
            model_dict['output_column'] = data_dict['output_column']
            model_dict['input_mean'] = data_dict['input_mean']
            model_dict['input_std'] = data_dict['input_std']
            model_dict['output_mean'] = data_dict['output_mean']
            model_dict['output_std'] = data_dict['output_std']
            torch.save(model_dict, args.save_name)
            print('\n| 保存模型:{} | val_loss:{:.4f} | val_mae:{:.4f} | val_mse:{:.4f} |\n'
                  .format(args.save_name, val_loss, mae, mse))
        # wandb
        if args.wandb:
            args.wandb_run.log({'metric/train_loss': train_loss, 'metric/val_loss': val_loss, 'metric/val_mae': mae,
                                'metric/val_mse': mse})
    return model_dict


class torch_dataset(torch.utils.data.Dataset):
    def __init__(self, args, data_dict):
        self.args = args
        self.data_dict = data_dict

    def __len__(self):
        return len(self.data_dict['train_input']) - self.args.input_size - self.args.output_size + 1

    def __getitem__(self, index):
        boundary = index + self.args.input_size
        series = self.data_dict['train_input'][index:boundary]  # 输入数据
        series = torch.tensor(series, dtype=torch.float32).permute(1, 0)  # 转换为tensor
        label = self.data_dict['train_output'][boundary:boundary + self.args.output_size]  # 输出标签
        label = torch.tensor(label, dtype=torch.float32).permute(1, 0)  # 转换为tensor
        return series, label
