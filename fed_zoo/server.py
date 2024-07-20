import time
import torch
import wandb
from collections import OrderedDict
from util.metric import psnr_slice, ssim_slice
clients = ['fastmri', 'cc359', 'siat']


def aggregate_weight(server_model, clients, aggregation_weights):
    update_state = OrderedDict()
    # aggregation
    for k, client in enumerate(clients):
        local_state = client.state_dict()
        for key in server_model.state_dict().keys():
            if k == 0:
                update_state[key] = local_state[key] * aggregation_weights[k]
            else:
                update_state[key] += local_state[key] * aggregation_weights[k]
    server_model.load_state_dict(update_state)

    # distribute
    for key in server_model.state_dict().keys():
        for k, client in enumerate(clients):
            client.state_dict()[key].data.copy_(server_model.state_dict()[key])

    return server_model, clients


def infer(model, valid_dataloaders, criterion, device, logger):
    model.to(device)
    model.eval()
    criterion.eval()
    start_time = time.time()
    psnrs = []
    ssims = []
    losss = []
    for idx, test_data in enumerate(valid_dataloaders):
        test_psnr = []
        test_ssim = []
        test_loss = []

        with torch.no_grad():
            for batch_idx, (x, target, mask, _, _) in enumerate(test_data):
                x = x.to(device, non_blocking=True)
                y = target.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                x = torch.view_as_real(x).permute(0, 3, 1, 2).contiguous()
                y = torch.abs(y)

                pred = model(x, mask)
                pred = torch.abs(torch.view_as_complex(pred.permute(0, 2, 3, 1).contiguous()))
                loss = criterion(y, pred)

                test_psnr.append(psnr_slice(y, pred))
                test_ssim.append(ssim_slice(y, pred))
                test_loss.append(loss.item())

        psnrs.append(sum(test_psnr) / len(test_psnr))
        ssims.append(sum(test_ssim) / len(test_ssim))
        losss.append(sum(test_loss) / len(test_loss))

    end_time = time.time()
    logger.info('server_psnr: {} | server_ssim: {} | server_loss: {}'.format(psnrs, ssims, losss))
    logger.info('server_infer time cost: {} s'.format(end_time-start_time))
    wandb.log({'server/psnr': sum(psnrs) / len(psnrs)})
    wandb.log({'server/ssim': sum(ssims) / len(ssims)})
    wandb.log({'server/loss': sum(losss) / len(losss)})
    return sum(psnrs) / len(psnrs), sum(ssims) / len(ssims), sum(losss) / len(losss)


def zf_infer(model, valid_dataloaders, device):
    model.to(device)
    model.eval()

    psnrs = []
    ssims = []
    losss = []
    zf_psnrs = []
    zf_ssims = []
    for _, test_data in enumerate(valid_dataloaders):
        test_psnr = []
        test_ssim = []

        zf_psnr = []
        zf_ssim = []

        with torch.no_grad():
            for batch_idx, (x, target, mask, _, _) in enumerate(test_data):
                x = x.to(device, non_blocking=True)
                y = target.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                input = torch.view_as_real(x).permute(0, 3, 1, 2).contiguous()
                y = torch.abs(y)

                pred = model(input, mask)
                pred = torch.abs(torch.view_as_complex(pred.permute(0, 2, 3, 1).contiguous()))

                test_psnr.append(psnr_slice(y, pred))
                test_ssim.append(ssim_slice(y, pred))

                zf_psnr.append(psnr_slice(y, torch.abs(x)))
                zf_ssim.append(ssim_slice(y, torch.abs(x)))
        print('test_psnr: {} | test_ssim: {}'.format(sum(test_psnr) / len(test_psnr), sum(test_ssim) / len(test_ssim)))
        print('zf_psnr: {} | zf_ssim: {}'.format(sum(zf_psnr) / len(zf_psnr), sum(zf_ssim) / len(zf_ssim)))
        psnrs.append(sum(test_psnr) / len(test_psnr))
        ssims.append(sum(test_ssim) / len(test_ssim))

        zf_psnrs.append(sum(zf_psnr) / len(zf_psnr))
        zf_ssims.append(sum(zf_ssim) / len(zf_ssim))
    return sum(zf_psnrs)/ len(zf_psnrs), sum(zf_ssims)/ len(zf_ssims), sum(psnrs) / len(psnrs), sum(ssims) / len(ssims)