import tqdm
import torch
from block.val_get import val_get
from block.model_ema import model_ema
from block.lr_get import adam, lr_adjust


def train_get(args, data_dict, model_dict, loss):
    # 加载模型
    model = model_dict['model'].to(args.device, non_blocking=args.latch)
    # 学习率
    optimizer = adam(args.regularization, args.r_value, model.parameters(), lr=args.lr_start, betas=(0.937, 0.999))
    optimizer.load_state_dict(model_dict['optimizer_state_dict']) if model_dict['optimizer_state_dict'] else None
    step_epoch = ((data_dict['train_input'].shape[1] - args.input_size - args.output_size + 1)
                  // args.batch // args.device_number * args.device_number)  # 每轮的步数
    optimizer_adjust = lr_adjust(args, step_epoch, model_dict['epoch_finished'])  # 学习率调整函数
    optimizer = optimizer_adjust(optimizer)  # 学习率初始化
    # 使用平均指数移动(EMA)调整参数(不能将ema放到args中，否则会导致模型保存出错)
    ema = model_ema(model) if args.ema else None
    if args.ema:
        ema.updates = model_dict['ema_updates']
    # 数据集
    train_dataset = torch_dataset(args, data_dict['train_input'], data_dict['train_output'])
    train_shuffle = False if args.distributed else True  # 分布式设置sampler后shuffle要为False
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=train_shuffle,
                                                   drop_last=True, pin_memory=args.latch, num_workers=args.num_worker,
                                                   sampler=train_sampler)
    val_dataset = torch_dataset(args, data_dict['val_input'], data_dict['val_output'])
    val_sampler = None  # 分布式时数据合在主GPU上进行验证
    val_batch = args.batch // args.device_number  # 分布式验证时batch要减少为一个GPU的量
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch, shuffle=False,
                                                 drop_last=False, pin_memory=args.latch, num_workers=args.num_worker,
                                                 sampler=val_sampler)
    # 分布式初始化
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                      output_device=args.local_rank) if args.distributed else model
    epoch_base = model_dict['epoch_finished'] + 1  # 新的一轮要+1
    for epoch in range(epoch_base, args.epoch + 1):  # 训练
        print(f'\n-----------------------第{epoch}轮-----------------------') if args.local_rank == 0 else None
        model.train()
        train_loss = 0  # 记录损失
        if args.local_rank == 0:  # tqdm
            tqdm_show = tqdm.tqdm(total=step_epoch)
        for index, (series_batch, true_batch) in enumerate(train_dataloader):
            series_batch = series_batch.to(args.device, non_blocking=args.latch)
            true_batch = true_batch.to(args.device, non_blocking=args.latch)
            if args.amp:
                with torch.cuda.amp.autocast():
                    pred_batch = model(series_batch)
                    loss_batch = loss(pred_batch, true_batch)
                args.amp.scale(loss_batch).backward()
                args.amp.step(optimizer)
                args.amp.update()
                optimizer.zero_grad()
            else:
                pred_batch = model(series_batch)
                loss_batch = loss(pred_batch, true_batch)
                loss_batch.backward()
                optimizer.step()
                optimizer.zero_grad()
            # 调整参数，ema.updates会自动+1
            ema.update(model) if args.ema else None
            # 记录损失
            train_loss += loss_batch.item()
            # 调整学习率
            optimizer = optimizer_adjust(optimizer)
            # tqdm
            if args.local_rank == 0:
                tqdm_show.set_postfix({'train_loss': loss_batch.item(),
                                       'lr': optimizer.param_groups[0]['lr']})  # 添加显示
                tqdm_show.update(args.device_number)  # 更新进度条
        # tqdm
        if args.local_rank == 0:
            tqdm_show.close()
        # 计算平均损失
        train_loss /= index + 1
        if args.local_rank == 0:
            print(f'\n| 训练 | train_loss:{train_loss:.4f} | lr:{optimizer.param_groups[0]["lr"]:.6f} |\n')
        # 清理显存空间
        del series_batch, true_batch, pred_batch, loss_batch
        torch.cuda.empty_cache()
        # 验证
        if args.local_rank == 0:  # 分布式时只验证一次
            val_loss, mae, mse = val_get(args, val_dataloader, model, loss, data_dict, ema,
                                         data_dict['val_input'].shape[1])
        # 保存
        if args.local_rank == 0:  # 分布式时只保存一次
            model_dict['model'] = model.module if args.distributed else model
            model_dict['epoch_finished'] = epoch
            model_dict['optimizer_state_dict'] = optimizer.state_dict()
            model_dict['ema_updates'] = ema.updates if args.ema else model_dict['ema_updates']
            model_dict['mean_input'] = data_dict['mean_input']
            model_dict['mean_output'] = data_dict['mean_output']
            model_dict['std_input'] = data_dict['std_input']
            model_dict['std_output'] = data_dict['std_output']
            model_dict['train_loss'] = train_loss
            model_dict['val_loss'] = val_loss
            model_dict['val_mae'] = mae
            model_dict['val_mse'] = mse
            torch.save(model_dict, 'last.pt')  # 保存最后一次训练的模型
            if mse < 1 and mse < model_dict['standard']:
                model_dict['standard'] = mse
                torch.save(model_dict, args.save_path)  # 保存最佳模型
                print(f'\n| 保存最佳模型:{args.save_path} | val_loss:{val_loss:.4f} | val_mae:{mae:.4f} |'
                      f' val_mse:{mse:.4f} |\n')
            # wandb
            if args.wandb:
                args.wandb_run.log({'metric/train_loss': train_loss,
                                    'metric/val_loss': val_loss,
                                    'metric/val_mae': mae,
                                    'metric/val_mse': mse})
        torch.distributed.barrier() if args.distributed else None  # 分布式时每轮训练后让所有GPU进行同步，快的GPU会在此等待


class torch_dataset(torch.utils.data.Dataset):
    def __init__(self, args, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.input_size = args.input_size
        self.output_size = args.output_size

    def __len__(self):
        return self.input_data.shape[1] - self.input_size - self.output_size + 1

    def __getitem__(self, index):
        boundary = index + self.input_size
        series = self.input_data[:, index:boundary]  # 输入数据
        series = torch.tensor(series, dtype=torch.float32)  # 转换为tensor
        label = self.output_data[:, boundary:boundary + self.output_size]  # 输出标签
        label = torch.tensor(label, dtype=torch.float32)  # 转换为tensor
        return series, label
