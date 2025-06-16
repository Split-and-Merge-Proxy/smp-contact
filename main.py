import argparse
import sys
import torch
import torch.nn as nn
import os
import logging
from build import build_dataloader, build_model, build_optimizer, build_loss
from deepinter import DeepInter
from utils import set_seed, EarlyStopping, show_results, evaluate, save_checkpoint, setup_logger
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./deephomo', help='data dir')
    parser.add_argument('--data_list_dir', type=str, default='./deephomo', help='data dir')
    parser.add_argument('--s3_dir', type=str, default='./deephomo', help='s3 dir')
    parser.add_argument('--data_ratio', type=float, default=1.0, help='data ratio')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--bs', type=int, default=1, help='batch size')
    parser.add_argument('--epochs', type=int, default=50, help='number of epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='')
    parser.add_argument('--test_checkpoint_name', type=str, default=None, help='') 
    parser.add_argument('--output_dir', type=str, default='./output', help='output dir')
    parser.add_argument('--max_seq_len', type=int, default=256, help='max sequence length')
    parser.add_argument('--num_heads', type=int, default=4, help='num heads in Triangle Self-Attention')
    parser.add_argument('--in_channels_rec_lig1d', type=int, default=788, help='1D rec or lig dimensions')
    parser.add_argument('--in_channels_rec_lig2d', type=int, default=210, help='2D rec or lig dimensions')
    parser.add_argument('--in_channels_com2d', type=int, default=146, help='2D com dimensions')
    parser.add_argument('--hidden_channels', type=int, default=64, help='hidden dimensions')
    parser.add_argument('--launcher', type=str, choices=['none', 'slurm', 'pytorch'], default="none", help="job launcher")
    parser.add_argument('--num_classes', type=int, default=1, help='num classes')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropout')
    parser.add_argument('--gamma', type=float, default=2, help='')
    parser.add_argument('--alpha', type=float, default=0.25, help='')
    parser.add_argument('--name', type=str, choices=['deepinter', 'smp'], default="none", help="the name of method")

    args = parser.parse_args()

    return args


def main(args):

    set_seed(args.seed)
    torch.cuda.empty_cache()
    writer_dir = os.path.join(args.output_dir, 'runs')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(writer_dir):
        os.makedirs(writer_dir)
    
    writer = SummaryWriter(writer_dir)
    logger = setup_logger(args.name, args.output_dir)

    if args.train:
        # dataloader
        train_loader = build_dataloader(args, mode='train')
        val_loader = build_dataloader(args, mode='val')

        logger.info('start load model')
        start_epoch = -1
        # model
        model = build_model(args).cuda()
        logger.info(model)

        # loss
        loss_fn = build_loss(args)
        # optimize
        optimizer, scheduler = build_optimizer(args, model)

        if args.resume_checkpoint is not None and os.path.exists(args.resume_checkpoint):
            logger.info('load the checkpoint from the {}'.format(args.resume_checkpoint))
            checkpoint = torch.load(args.resume_checkpoint, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['net'])  
    
        logger.info('Start training')
        best_loss = float('inf')
        step = 0

        for epoch in range(start_epoch + 1, args.epochs + start_epoch + 1):
            model.train()
            tbar = tqdm(train_loader)
            # print(tbar[0])
            tbar.set_description('Epoch {}'.format(epoch + 1))
            for data in tbar:
                step += 1

                rec1d, rec2d, lig1d, lig2d = data['rec1d'].cuda(), data['rec2d'].cuda(), data['lig1d'].cuda(), data['lig2d'].cuda()
                com2d, intra_rec, intra_lig = data['com2d'].cuda(), data['intra_distA'].cuda(), data['intra_distB'].cuda()
                labels = data['flatten_contact_map'].cuda() # [B, L_rec*L_lig]

                logits = model(rec1d, rec2d, lig1d, lig2d, com2d, intra_rec, intra_lig)  # [B, 1, L_rec, L_lig]
                logits = torch.flatten(logits, start_dim=1) # [B, L_rec*L_lig]
                loss = loss_fn(logits, labels)  

                tbar.set_postfix(loss=loss.item())

                writer.add_scalar('Train Loss', loss.item(), step)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=0.5)   # clip gradient
                optimizer.step()
            
            logger.info('Start evaluation')

            results = evaluate(model, val_loader, loss_fn)
            val_loss = results['loss']
            scheduler.step(val_loss)

            writer.add_scalar('Val Loss', val_loss, epoch + 1)

            show_results(results, logger)

            save_checkpoint(model, optimizer, scheduler, epoch, args.output_dir, file_name='epoch_{}.pth'.format(epoch + 1))  # save each checkpoint

            if (val_loss + 5e-6) <= best_loss:
                best_loss = val_loss
                save_checkpoint(model, optimizer, scheduler, epoch, args.output_dir, file_name='best.pth')  # save the best checkpoint


    elif args.test:
        logger.info('Start test')
        # dataloader
        test_loader = build_dataloader(args, mode='test')
        logger.info('start load model')
        model = build_model(args)
        loss_fn = build_loss(args)
        model = model.cuda()
        ckpt = torch.load(os.path.join(args.output_dir, args.test_checkpoint_name))
        model.load_state_dict(ckpt['net'])

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params / 1e6:.2f} M")
        # import pdb; pdb.set_trace()

        logger.info('evaluation')

        results = evaluate(model, test_loader, loss_fn)

        show_results(results, logger)

    else:
        raise ValueError('Unknown mode!!!')


if __name__ == "__main__":
    args = parse_args()
    main(args)
