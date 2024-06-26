import tqdm
import torch
from block.metric_get import metric


def val_get(args, val_dataloader, model, loss, data_dict, ema, data_len):
    with torch.no_grad():
        model = ema.ema if args.ema else model.eval()
        pred = []
        true = []
        val_loss = 0
        tqdm_len = (data_len - args.input_size - args.output_size + 1 - 1) // (args.batch // args.device_number) + 1
        tqdm_show = tqdm.tqdm(total=tqdm_len)
        for index, (series_batch, true_batch, special) in enumerate(val_dataloader):
            series_batch = series_batch.to(args.device, non_blocking=args.latch)
            true_batch = true_batch.to(args.device, non_blocking=args.latch)
            pred_batch = model(series_batch) if 'special' not in args.model else model(series_batch, special)
            loss_batch = loss(pred_batch, true_batch)
            val_loss += loss_batch.item()
            pred.append(pred_batch.cpu())
            true.append(true_batch.cpu())
            # tqdm
            tqdm_show.set_postfix({'val_loss': loss_batch.item()})  # 添加loss显示
            tqdm_show.update(1)  # 更新进度条
        # tqdm
        tqdm_show.close()
        # 计算指标
        val_loss /= (index + 1)
        pred = torch.concat(pred, dim=0)
        true = torch.concat(true, dim=0)
        # 计算总相对指标
        mae, rmse = metric(pred, true)
        print(f'\n| 验证 | all | val_loss:{val_loss:.4f} | val_mae:{mae:.4f} | val_rmse:{rmse:.4f} |')
        # 计算各类别相对指标和真实指标
        for i in range(pred.shape[1]):
            column = args.output_column[i]
            _mae, _rmse = metric(pred[:, i], true[:, i])
            pred[:, i] = pred[:, i] * data_dict['std_output'][i] + data_dict['mean_output'][i]
            true[:, i] = true[:, i] * data_dict['std_output'][i] + data_dict['mean_output'][i]
            _mae_true, _rmse_true = metric(pred[:, i], true[:, i])
            print('| 验证 | {} | mae:{:.4f} | rmse:{:.4f} | mae_true:{:.4f} | rmse_true:{:.4f} |'
                  .format(column, _mae, _rmse, _mae_true, _rmse_true))
    return val_loss, mae.item(), rmse.item()
