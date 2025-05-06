import os
import math
import copy
import torch
import logging
import numpy as np
import pandas as pd


class train_class:
    '''
        model_load: 加载模型
        data_load: 加载数据
        dataloader_load: 加载数据处理器
        optimizer_load: 加载学习率
        loss_load: 训练损失
        train: 训练模型
        validation: 训练时的模型验证
    '''

    def __init__(self, args):
        self.args = args
        self.model_dict = self.model_load()  # 模型
        self.model_dict['model'] = self.model_dict['model'].to(args.device, non_blocking=args.latch)  # 设备
        self.data_dict = self.data_load()  # 数据
        self.train_dataloader, self.val_dataloader = self.dataloader_load()  # 数据处理器
        self.optimizer, self.optimizer_adjust = self.optimizer_load()  # 学习率、学习率调整
        self.loss = self.loss_load()  # 损失函数
        if args.local_rank == 0 and args.ema:  # 平均指数移动(EMA)，创建ema模型
            self.ema = model_ema(self.model_dict['model'])
            self.ema.update_total = self.model_dict['ema_update']
        if args.distributed:  # 分布式初始化
            self.model_dict['model'] = torch.nn.parallel.DistributedDataParallel(self.model_dict['model'],
                                                                                 device_ids=[args.local_rank],
                                                                                 output_device=args.local_rank)
        if args.local_rank == 0 and args.log:  # 日志
            log_path = os.path.dirname(__file__) + '/log.log'
            logging.basicConfig(filename=log_path, level=logging.INFO,
                                format='%(asctime)s | %(levelname)s | %(message)s')
            logging.info('-------------------- log --------------------')

    @staticmethod
    def read_column(column_file):  # column处理
        if os.path.exists(column_file):
            with open(column_file, encoding='utf-8') as f:
                column = [_.strip() for _ in f.readlines()]
        else:
            column = column_file.split(',')
        return column

    @staticmethod
    def metric(pred, label):
        mae = torch.mean(abs(pred - label))
        rmse = torch.sqrt(torch.mean(torch.square(pred - label)))
        return mae, rmse

    def model_load(self):
        args = self.args
        if os.path.exists(args.weight_path):
            model_dict = torch.load(args.weight_path, map_location='cpu', weights_only=False)
            for param in model_dict['model'].parameters():
                param.requires_grad_(True)  # 打开梯度(保存的ema模型为关闭)
            if args.weight_again:
                model_dict['epoch_finished'] = 0  # 已训练的轮数
                model_dict['optimizer_state_dict'] = None  # 学习率参数
                model_dict['ema_update'] = 0  # ema参数
                model_dict['standard'] = 100  # 评价指标
        else:
            exec(f'from model.{args.model} import {args.model}')
            model = eval(f'{args.model}(self.args)')
            model_dict = {
                'model': model,
                'epoch_finished': 0,  # 已训练的轮数
                'optimizer_state_dict': None,  # 学习率参数
                'ema_update': 0,  # ema参数
                'standard': 100,  # 评价指标
            }
        return model_dict

    def data_load(self):
        args = self.args
        # 读取数据
        try:
            df = pd.read_csv(args.data_path, encoding='utf-8', index_col=0)
        except:
            df = pd.read_csv(args.data_path, encoding='gbk', index_col=0)
        input_data = np.array(df[args.input_column]).astype(np.float32)
        output_data = np.array(df[args.output_column]).astype(np.float32)
        # 划分数据
        add = args.input_len + args.output_len - 1  # 输入数据后面的补足
        data_len = len(df) - add  # 输入数据的真实数量
        boundary = int(data_len * args.divide[0] / (args.divide[0] + args.divide[1]))  # 数据划分
        if args.divide_train == 1:  # 使用所有数据训练
            train_input = input_data  # 训练数据
            train_output = output_data  # 训练标签
        elif args.divide_train == 2:  # 使用验证集训练
            train_input = input_data[boundary:len(df)]  # 训练数据
            train_output = output_data[boundary:len(df)]  # 训练标签
        else:  # 使用训练集训练
            train_input = input_data[0:boundary + add]  # 训练数据
            train_output = output_data[0:boundary + add]  # 训练标签
        val_input = input_data[boundary:len(df)].copy()  # 验证数据
        val_output = output_data[boundary:len(df)].copy()  # 验证标签
        # 周期
        if args.z_score == 1:
            mean_input = np.mean(input_data, axis=0)
            mean_output = np.mean(output_data, axis=0)
            std_input = np.std(input_data, axis=0)
            std_output = np.std(output_data, axis=0)
        elif args.z_score == 2:
            mean_input = np.mean(val_input, axis=0)
            mean_output = np.mean(val_output, axis=0)
            std_input = np.std(val_input, axis=0)
            std_output = np.std(val_output, axis=0)
        else:
            mean_input = np.mean(train_input, axis=0)
            mean_output = np.mean(train_output, axis=0)
            std_input = np.std(train_input, axis=0)
            std_output = np.std(train_output, axis=0)
        # 归一化
        train_input = (train_input - mean_input) / std_input
        val_input = (val_input - mean_input) / std_input
        train_output = (train_output - mean_output) / std_output
        val_output = (val_output - mean_output) / std_output
        # 记录数据
        data_dict = {
            'train_input': train_input,
            'train_output': train_output,
            'val_input': val_input,
            'val_output': val_output,
            'mean_input': mean_input,
            'mean_output': mean_output,
            'std_input': std_input,
            'std_output': std_output,
        }
        return data_dict

    def dataloader_load(self):
        args = self.args
        train_dataset = torch_dataset(args, self.data_dict['train_input'], self.data_dict['train_output'])
        train_shuffle = False if args.distributed else True  # 分布式设置sampler后shuffle要为False
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=train_shuffle,
                                                       drop_last=True, pin_memory=args.latch,
                                                       num_workers=args.num_worker,
                                                       sampler=train_sampler, collate_fn=train_dataset.collate_fn)
        val_dataset = torch_dataset(args, self.data_dict['val_input'], self.data_dict['val_output'])
        val_sampler = None  # 分布式时数据合在主GPU上进行验证
        val_batch = args.batch // args.device_number  # 分布式验证时batch要减少为一个GPU的量
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch, shuffle=False,
                                                     drop_last=False, pin_memory=args.latch,
                                                     num_workers=args.num_worker,
                                                     sampler=val_sampler, collate_fn=val_dataset.collate_fn)
        return train_dataloader, val_dataloader

    def optimizer_load(self):
        args = self.args
        if args.regularization == 'L2':
            optimizer = torch.optim.Adam(self.model_dict['model'].parameters(),
                                         lr=args.lr_start, betas=(0.937, 0.999), weight_decay=args.r_value)
        else:
            optimizer = torch.optim.Adam(self.model_dict['model'].parameters(),
                                         lr=args.lr_start, betas=(0.937, 0.999))
        if self.model_dict['optimizer_state_dict'] is not None:
            optimizer.load_state_dict(self.model_dict['optimizer_state_dict'])
        step_epoch = ((len(self.data_dict['train_input']) - args.input_len - args.output_len + 1)
                      // args.batch // args.device_number * args.device_number)  # 每轮步数
        optimizer_adjust = lr_adjust(args, step_epoch, self.model_dict['epoch_finished'])  # 学习率调整函数
        optimizer = optimizer_adjust(optimizer)  # 学习率初始化
        return optimizer, optimizer_adjust

    def loss_load(self):
        choice_dict = {'mae': 'torch.nn.L1Loss()',
                       'mse': 'torch.nn.MSELoss()',
                       'mse_decay': 'mse_decay(self.args)'}
        loss = eval(choice_dict[self.args.loss])
        return loss

    def train(self):
        args = self.args
        model = self.model_dict['model']
        epoch_base = self.model_dict['epoch_finished'] + 1  # 新的一轮要+1
        for epoch in range(epoch_base, args.epoch + 1):
            if args.local_rank == 0 and args.print_info:
                info = f'-----------------------epoch:{epoch}-----------------------'
                print(info)
            model.train()
            train_loss = 0  # 记录损失
            for index, (series_batch, label_batch) in enumerate(self.train_dataloader):
                series_batch = series_batch.to(args.device, non_blocking=args.latch)
                label_batch = label_batch.to(args.device, non_blocking=args.latch)
                if args.amp:
                    with torch.cuda.amp.autocast():
                        pred_batch = model(series_batch)
                        loss_batch = self.loss(pred_batch, label_batch)
                    args.amp.scale(loss_batch).backward()
                    args.amp.step(self.optimizer)
                    args.amp.update()
                    self.optimizer.zero_grad()
                else:
                    pred_batch = model(series_batch)
                    loss_batch = self.loss(pred_batch, label_batch)
                    loss_batch.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                self.ema.update(model) if args.local_rank == 0 and args.ema else None  # 更新ema模型参数
                train_loss += loss_batch.item()  # 记录损失
                self.optimizer = self.optimizer_adjust(self.optimizer)  # 调整学习率
            # 计算平均损失
            train_loss /= index + 1
            # 日志
            if args.local_rank == 0 and args.print_info:
                info = f'| train | train_loss:{train_loss:.4f} | lr:{self.optimizer.param_groups[0]["lr"]:.6f} |'
                print(info)
            # 清理显存空间
            del series_batch, label_batch, pred_batch, loss_batch
            torch.cuda.empty_cache()
            # 验证和保存
            if args.local_rank == 0:
                val_loss, mae, rmse = self.validation()  # 验证
                self.model_dict['model'] = model.module if args.distributed else model
                self.model_dict['model'] = self.ema.ema_model if args.ema else self.model_dict['model']
                self.model_dict['epoch_finished'] = epoch
                self.model_dict['optimizer_state_dict'] = self.optimizer.state_dict()
                self.model_dict['ema_update'] = self.ema.update_total if args.ema else self.model_dict['ema_update']
                self.model_dict['mean_input'] = self.data_dict['mean_input']
                self.model_dict['mean_output'] = self.data_dict['mean_output']
                self.model_dict['std_input'] = self.data_dict['std_input']
                self.model_dict['std_output'] = self.data_dict['std_output']
                self.model_dict['train_loss'] = train_loss
                self.model_dict['val_loss'] = val_loss
                self.model_dict['val_mae'] = mae
                self.model_dict['val_rmse'] = rmse
                if epoch % args.save_epoch == 0 or epoch == args.epoch:  # 保存模型
                    torch.save(self.model_dict, args.save_path)
                if val_loss < 1 and val_loss < self.model_dict['standard']:  # 保存最佳模型
                    self.model_dict['standard'] = val_loss
                    torch.save(self.model_dict, args.save_best)
                    # 日志
                    info = (f'| best_model | val_loss:{val_loss:.4f} | val_mae:{mae:.4f} |'
                            f' val_rmse:{rmse:.4f} | {args.save_best} |')
                    print(info) if args.print_info else None
                    logging.info(info) if args.log else None
                # wandb
                args.wandb_run.log({'metric/train_loss': train_loss,
                                    'metric/val_loss': val_loss,
                                    'metric/val_mae': mae,
                                    'metric/val_rmse': rmse}) if args.wandb else None
            torch.distributed.barrier() if args.distributed else None  # 分布式时每轮训练后让所有GPU进行同步，快的GPU会在此等待

    def validation(self):
        args = self.args
        with torch.no_grad():
            model = self.ema.ema_model.eval() if args.ema else self.model_dict['model'].eval()
            pred = []
            label = []
            val_loss = 0
            for index, (series_batch, label_batch) in enumerate(self.val_dataloader):
                series_batch = series_batch.to(args.device, non_blocking=args.latch)
                label_batch = label_batch.to(args.device, non_blocking=args.latch)
                pred_batch = model(series_batch)
                loss_batch = self.loss(pred_batch, label_batch)
                val_loss += loss_batch.item()
                pred.append(pred_batch.cpu())
                label.append(label_batch.cpu())
            # 计算指标
            val_loss /= (index + 1)
            pred = torch.concat(pred, dim=0)
            label = torch.concat(label, dim=0)
            mae, rmse = self.metric(pred, label)
            # 计算各类别相对指标和真实指标
            for i, column in enumerate(args.output_column):
                mae_, rmse_ = self.metric(pred[:, i], label[:, i])
                pred[:, i] = pred[:, i] * self.data_dict['std_output'][i] + self.data_dict['mean_output'][i]
                label[:, i] = label[:, i] * self.data_dict['std_output'][i] + self.data_dict['mean_output'][i]
                true_mae, true_rmse = self.metric(pred[:, i], label[:, i])
                info = (f'| val | {column} | mae:{mae_:.4f} | rmse:{rmse_:.4f} | true_mae:{true_mae:.4f} |'
                        f' true_rmse:{true_rmse:.4f} |')
                print(info) if args.print_info else None
            # 日志
            if args.print_info:
                info = f'| val | all | val_loss:{val_loss:.4f} | val_mae:{mae:.4f} | val_rmse:{rmse:.4f} |'
                print(info)
        return val_loss, mae.item(), rmse.item()


class model_ema:
    def __init__(self, model, decay=0.9999, tau=2000, update_total=0):
        self.ema_model = copy.deepcopy(self._get_model(model)).eval()  # FP32 EMA
        self.update_total = update_total
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for param in self.ema_model.parameters():
            param.requires_grad_(False)  # 关闭梯度

    def update(self, model):
        with torch.no_grad():
            self.update_total += 1
            d = self.decay(self.update_total)
            state_dict = self._get_model(model).state_dict()
            for k, v in self.ema_model.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * state_dict[k].detach()

    def _get_model(self, model):
        if type(model) in (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel):
            return model.module
        else:
            return model


class lr_adjust:
    def __init__(self, args, step_epoch, epoch_finished):
        self.lr_start = args.lr_start  # 初始学习率
        self.lr_end = args.lr_end_ratio * args.lr_start  # 最终学习率
        self.lr_end_epoch = args.lr_end_epoch  # 最终学习率达到的轮数
        self.step_all = self.lr_end_epoch * step_epoch  # 总调整步数
        self.step_finished = epoch_finished * step_epoch  # 已调整步数
        self.warmup_step = max(5, int(args.warmup_ratio * self.step_all))  # 预热训练步数

    def __call__(self, optimizer):
        self.step_finished += 1
        step_now = self.step_finished
        decay = step_now / self.step_all
        lr = self.lr_end + (self.lr_start - self.lr_end) * math.cos(math.pi / 2 * decay)
        if step_now <= self.warmup_step:
            lr = lr * (0.1 + 0.9 * step_now / self.warmup_step)
        lr = max(lr, 0.000001)
        for i in range(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr
        return optimizer


class mse_decay:
    '''
        mse_decay: 使越靠近前面的数值的准确率越重要
    '''

    def __init__(self, args):
        self.decay = torch.linspace(1.5, 0.5, args.output_len).unsqueeze(1).to(args.device)
        self.mse = torch.nn.MSELoss()

    def __call__(self, pred, label):
        self.decay = self.decay
        pred_decay = pred * self.decay
        label_decay = label * self.decay
        loss = self.mse(pred_decay, label_decay)
        return loss


class torch_dataset(torch.utils.data.Dataset):
    def __init__(self, args, input_data, output_data):
        self.model = args.model
        self.input_len = args.input_len
        self.output_len = args.output_len
        self.input_data = input_data
        self.output_data = output_data

    def __len__(self):
        return len(self.input_data) - self.input_len - self.output_len + 1

    def __getitem__(self, index):
        boundary = index + self.input_len
        series = self.input_data[index:boundary]  # 输入数据
        series = torch.tensor(series, dtype=torch.float32)
        label = self.output_data[boundary:boundary + self.output_len]  # 输出标签
        label = torch.tensor(label, dtype=torch.float32)
        return series, label

    def collate_fn(self, getitem_list):  # 自定义__getitem__合成方式
        series_list = [_[0] for _ in getitem_list]
        label_list = [_[1] for _ in getitem_list]
        series_batch = torch.stack(series_list, dim=0)
        label_batch = torch.stack(label_list, dim=0)
        return series_batch, label_batch
