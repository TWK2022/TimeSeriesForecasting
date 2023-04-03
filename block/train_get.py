import tqdm
import torch
from block.val_get import val_get
from block.ModelEMA import ModelEMA


def train_get(args, data_dict, model_dict, loss):
    # 加载模型
    model = model_dict['model'].to(args.device, non_blocking=args.latch)
    # 分布式初始化
    torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                              output_device=args.local_rank) if args.distributed else None
    # 使用平均指数移动(EMA)调整参数(不能将ema放到args中，否则会导致模型保存出错)
    ema = ModelEMA(model) if args.ema else None
    if args.ema:
        ema.updates = model_dict['ema_updates']
    # 学习率
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer.load_state_dict(model_dict['optimizer_state_dict']) if model_dict['optimizer_state_dict'] else None
    # 数据集
    train_dataset = torch_dataset(args, data_dict['train_input'], data_dict['train_output'])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, drop_last=True,
                                                   pin_memory=args.latch, num_workers=args.num_worker,
                                                   sampler=train_sampler)
    val_dataset = torch_dataset(args, data_dict['val_input'], data_dict['val_output'])
    val_sampler = None  # 分布式时数据合在主GPU上进行验证
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch, shuffle=False, drop_last=False,
                                                 pin_memory=args.latch, num_workers=args.num_worker,
                                                 sampler=val_sampler)
    for epoch in range(args.epoch):
        # 训练
        print(f'\n-----------------------第{epoch + 1}轮-----------------------')
        model.train()
        train_loss = 0  # 记录训练损失
        for item, (series_batch, true_batch) in enumerate(tqdm.tqdm(train_dataloader)):
            series_batch = series_batch.to(args.device, non_blocking=args.latch)
            true_batch = true_batch.to(args.device, non_blocking=args.latch)
            if args.scaler:
                with torch.cuda.amp.autocast():
                    pred_batch = model(series_batch)
                    loss_batch = loss(pred_batch, true_batch)
                optimizer.zero_grad()
                args.scaler.scale(loss_batch).backward()
                args.scaler.step(optimizer)
                args.scaler.update()
            else:
                pred_batch = model(series_batch)
                loss_batch = loss(pred_batch, true_batch)
                optimizer.zero_grad()
                loss_batch.backward()
                optimizer.step()
            # 调整参数，ema.updates会自动+1
            ema.update(model) if args.ema else None
            # 记录损失
            train_loss += loss_batch.item()
        train_loss = train_loss / (item + 1)
        print('\n| train_loss:{:.4f} |\n'.format(train_loss))
        # 清理显存空间
        del series_batch, true_batch, pred_batch, loss_batch
        torch.cuda.empty_cache()
        # 验证
        if args.local_rank == 0:  # 分布式时只验证一次
            val_loss, mae, mse = val_get(args, val_dataloader, model, loss, data_dict, ema)
        # 保存
        if args.local_rank == 0:  # 分布式时只保存一次
            model_dict['model'] = model.eval()
            model_dict['epoch'] += 1
            model_dict['optimizer_state_dict'] = optimizer.state_dict()
            model_dict['ema_updates'] = ema.updates if args.ema else model_dict['ema_updates']
            model_dict['input_mean'] = data_dict['input_mean']
            model_dict['input_std'] = data_dict['input_std']
            model_dict['output_mean'] = data_dict['output_mean']
            model_dict['output_std'] = data_dict['output_std']
            model_dict['train_loss'] = train_loss
            model_dict['val_loss'] = val_loss
            model_dict['val_mae'] = mae
            model_dict['val_mse'] = mse
            torch.save(model_dict, 'last.pt')  # 保存最后一次训练的模型
            if mse < 1 and mse < model_dict['standard']:
                model_dict['standard'] = mse
                torch.save(model_dict, args.save_name)  # 保存最佳模型
                print('\n| 保存最佳模型:{} | val_loss:{:.4f} | val_mae:{:.4f} | val_mse:{:.4f} |\n'
                      .format(args.save_name, val_loss, mae, mse))
            # wandb
            if args.wandb:
                args.wandb_run.log({'metric/train_loss': train_loss,
                                    'metric/val_loss': val_loss,
                                    'metric/val_mae': mae,
                                    'metric/val_mse': mse})
        torch.distributed.barrier() if args.distributed else None  # 分布式时每轮训练后让所有GPU进行同步，快的GPU会在此等待


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
