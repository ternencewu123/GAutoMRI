import torch
import wandb

from util.metric import psnr_slice, ssim_slice


def train(name, model, train_dataloader, criterion, optimizer, args, device, logger):
    model.to(device)
    model.train()

    local_avg_train_psnr = []
    local_avg_train_ssim = []
    local_avg_train_loss = []
    for epoch in range(args.epochs):
        # training
        train_psnr, train_ssim, train_loss = local_train(train_dataloader, model, criterion, optimizer, device)
        logger.info('{} | epoch: {} | local_train_psnr: {:.4f} | local_train_ssim: {:.4f} | local_train_loss: '
                         '{:.4f}'.format(name, epoch + 1, train_psnr, train_ssim, train_loss))
        local_avg_train_psnr.append(train_psnr)
        local_avg_train_ssim.append(train_ssim)
        local_avg_train_loss.append(train_loss)

    wandb.log({'train/psnr/{}'.format(name): sum(local_avg_train_psnr) / len(local_avg_train_psnr)})
    wandb.log({'train/ssim/{}'.format(name): sum(local_avg_train_ssim) / len(local_avg_train_ssim)})
    wandb.log({'train/loss/{}'.format(name): sum(local_avg_train_loss) / len(local_avg_train_loss)})

    return sum(local_avg_train_psnr) / len(local_avg_train_psnr), sum(local_avg_train_ssim) / \
                                len(local_avg_train_ssim), sum(local_avg_train_loss) / len(local_avg_train_loss)


def local_train(train_queue, model, criterion, optimizer, device):
    psnrs = []
    ssims = []
    losss = []
    for step, (trn_X, trn_y, mask, _, _) in enumerate(train_queue):
        trn_X, trn_y = trn_X.to(device, non_blocking=True), trn_y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        trn_X = torch.view_as_real(trn_X).permute(0, 3, 1, 2).contiguous()
        trn_y = torch.abs(trn_y)

        optimizer.zero_grad()
        logits = model(trn_X, mask)
        logits = torch.abs(torch.view_as_complex(logits.permute(0, 2, 3, 1).contiguous()))
        loss = criterion(logits, trn_y)
        loss.backward()
        optimizer.step()

        psnr = psnr_slice(trn_y, logits)
        ssim = ssim_slice(trn_y, logits)
        psnrs.append(psnr)
        ssims.append(ssim)
        losss.append(loss.item())

    return sum(psnrs) / len(psnrs), sum(ssims) / len(ssims), sum(losss) / len(losss)


def local_infer(name, model, valid_dataloader, criterion, device, logger):
    model.to(device)
    model.eval()
    criterion.eval()
    psnrs = []
    ssims = []
    losss = []
    with torch.no_grad():
        for batch_idx, (trn_X, trn_y, mask, _, _) in enumerate(valid_dataloader):
            trn_X, trn_y = trn_X.to(device, non_blocking=True), trn_y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            trn_X = torch.view_as_real(trn_X).permute(0, 3, 1, 2).contiguous()
            trn_y = torch.abs(trn_y)

            pred = model(trn_X, mask)
            pred = torch.abs(torch.view_as_complex(pred.permute(0, 2, 3, 1).contiguous()))
            loss = criterion(pred, trn_y)

            psnr = psnr_slice(trn_y, pred)
            ssim = ssim_slice(trn_y, pred)

            psnrs.append(psnr)
            ssims.append(ssim)
            losss.append(loss.item())

    logger.info('{} | local_valid_psnr: {:.4f} | local_valid_ssim: {:.4f} | local_valid_loss: {:.4f}'.format(name,
                                sum(psnrs) / len(psnrs), sum(ssims) / len(ssims), sum(losss) / len(losss)))
    wandb.log({'valid/psnr/{}'.format(name): sum(psnrs) / len(psnrs)})
    wandb.log({'valid/ssim/{}'.format(name): sum(ssims) / len(ssims)})
    wandb.log({'valid/loss/{}'.format(name): sum(losss) / len(losss)})
    return sum(psnrs) / len(psnrs), sum(ssims) / len(ssims), sum(losss) / len(losss)


def local_test(name, model, valid_dataloader, criterion, device, logger):
    model.to(device)
    model.eval()
    criterion.eval()
    psnrs = []
    ssims = []
    losss = []
    with torch.no_grad():
        for batch_idx, (trn_X, trn_y, mask, _, _) in enumerate(valid_dataloader):
            trn_X, trn_y = trn_X.to(device, non_blocking=True), trn_y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            trn_X = torch.view_as_real(trn_X).permute(0, 3, 1, 2).contiguous()
            trn_y = torch.abs(trn_y)

            pred = model(trn_X, mask)
            pred = torch.abs(torch.view_as_complex(pred.permute(0, 2, 3, 1).contiguous()))
            loss = criterion(pred, trn_y)

            psnr = psnr_slice(trn_y, pred)
            ssim = ssim_slice(trn_y, pred)

            psnrs.append(psnr)
            ssims.append(ssim)
            losss.append(loss.item())

    logger.info('{} | local_test_psnr: {:.4f} | local_test_ssim: {:.4f} | local_test_loss: {:.4f}'.format(name,
                                sum(psnrs) / len(psnrs), sum(ssims) / len(ssims), sum(losss) / len(losss)))
    wandb.log({'test/psnr/{}'.format(name): sum(psnrs) / len(psnrs)})
    wandb.log({'test/ssim/{}'.format(name): sum(ssims) / len(ssims)})
    wandb.log({'test/loss/{}'.format(name): sum(losss) / len(losss)})
    return sum(psnrs) / len(psnrs), sum(ssims) / len(ssims), sum(losss) / len(losss)


def local_test1(name, model, valid_dataloader, criterion, device, logger):
    model.to(device)
    model.eval()
    criterion.eval()
    psnrs = []
    ssims = []
    losss = []
    with torch.no_grad():
        for batch_idx, (trn_X, trn_y, mask, _, _) in enumerate(valid_dataloader):
            trn_X, trn_y = trn_X.to(device, non_blocking=True), trn_y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            trn_X = torch.view_as_real(trn_X).permute(0, 3, 1, 2).contiguous()
            trn_y = torch.abs(trn_y)

            pred = model(trn_X, mask)
            pred = torch.abs(torch.view_as_complex(pred.permute(0, 2, 3, 1).contiguous()))
            loss = criterion(pred, trn_y)

            psnr = psnr_slice(trn_y, pred)
            ssim = ssim_slice(trn_y, pred)

            psnrs.append(psnr)
            ssims.append(ssim)
            losss.append(loss.item())

    return sum(psnrs) / len(psnrs), sum(ssims) / len(ssims), sum(losss) / len(losss)