
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import DeepHomoDataset
from deepinter import DeepInter
from loss import FocalLoss
from utils import collate_fn

def build_dataloader(args, mode):
    # s3_dir, data_dir, data_list_dir, launcher, data_ratio, max_seq_len, mode
    if mode == 'train':
        dataset = DeepHomoDataset(args.s3_dir, args.data_dir, args.data_list_dir, args.launcher, args.data_ratio, max_seq_len=args.max_seq_len, mode='train')
        dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=8, drop_last=True, collate_fn=collate_fn, pin_memory=True)  # whether use drop_last?
    elif mode == 'val':
        dataset = DeepHomoDataset(args.s3_dir, args.data_dir, args.data_list_dir, args.launcher, args.data_ratio, max_seq_len=args.max_seq_len, mode='val')
        dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=8, drop_last=True, collate_fn=collate_fn, pin_memory=True)
    elif mode == 'test':
        dataset = DeepHomoDataset(args.s3_dir, args.data_dir, args.data_list_dir, args.launcher, args.data_ratio, max_seq_len=args.max_seq_len, mode='test')
        dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=False, num_workers=8, drop_last=False, collate_fn=collate_fn, pin_memory=True)
    
    return dataloader


def build_model(args):

    model = DeepInter(
        in_channels_rec_lig1d = args.in_channels_rec_lig1d,
        in_channels_rec_lig2d = args.in_channels_rec_lig2d,
        in_channels_com2d = args.in_channels_com2d,
        hidden_channels = args.hidden_channels,
        num_heads = args.num_heads,
        dropout_rate = args.dropout_rate,
        num_classes = args.num_classes,
    )
    
    return model


def build_loss(args):
    
    loss_fn = FocalLoss(
        gamma=args.gamma, 
        alpha=args.alpha,
    )

    return loss_fn


def build_optimizer(args, model):
    
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=3)

    return optimizer, scheduler
