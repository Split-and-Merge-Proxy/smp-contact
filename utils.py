import torch
import os
import numpy as np
import random
import logging
import sys
from terminaltables import AsciiTable
from tqdm import tqdm

def read_txt(file_):
    lines = []
    f = open(file_, 'r')
    for line in f:
        lines.append(line.split('\n')[0])
    
    return lines

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def collate_fn(data):
    batch_rec1d = [b['rec1d'] for b in data]
    batch_rec2d = [b['rec2d'] for b in data]
    batch_lig1d = [b['lig1d'] for b in data]
    batch_lig2d = [b['lig2d'] for b in data]
    batch_com2d = [b['com2d'] for b in data]
    batch_intra_distA = [b['intra_distA'] for b in data]
    batch_intra_distB = [b['intra_distB'] for b in data]

    batch_flatten_contact_map = [b['flatten_contact_map'] for b in data]
    batch_pdb_name = [b['pdb_name'] for b in data]

    batch_data = {
        'rec1d': torch.tensor(batch_rec1d),
        'rec2d': torch.tensor(batch_rec2d),
        'lig1d': torch.tensor(batch_lig1d),
        'lig2d': torch.tensor(batch_lig2d),
        'com2d': torch.tensor(batch_com2d),
        'intra_distA': torch.tensor(batch_intra_distA),
        'intra_distB': torch.tensor(batch_intra_distB),
        'flatten_contact_map': torch.tensor(batch_flatten_contact_map),
        'pdb_name': batch_pdb_name[0],
    }

    return batch_data



def seq2pairwise(feat1d_1, feat1d_2):
    # concatenate 1D feats to 2D feats
    device = feat1d_1.device
    b, c, L1 = feat1d_1.size()
    _, _, L2 = feat1d_2.size()

    out1 = feat1d_1.unsqueeze(3).to(device)
    repeat_idx = [1] * out1.dim()
    repeat_idx[3] = L2
    out1 = out1.repeat(*(repeat_idx))

    out2 = feat1d_2.unsqueeze(2).to(device)
    repeat_idx = [1] * out2.dim()
    repeat_idx[2] = L1
    out2 = out2.repeat(*(repeat_idx))

    return torch.cat([out1, out2], dim=1)

def show_results(data_dict, logger):
    # assert len(data_list) == len(metrics)
    metrics = data_dict.keys()
    data_list = data_dict.values()
    table = [[m for m in metrics]]
    table.append([d for d in data_list])
    table = AsciiTable(table)
    # print(table)
    for i in range(len(data_list)):
        table.justify_columns[i] = 'center'

    logger.info('\n' + table.table)

def save_checkpoint(net, optimizer, scheduler, epoch, save_dir, file_name):
    """
    save checkpoint, including net, optimizer
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    checkpoint_file = os.path.join(save_dir, file_name)
    checkpoint = {
        "net": net.state_dict(),
        'optimizer':optimizer.state_dict(),
        "epoch": epoch,
        'lr_schedule': scheduler.state_dict(),
    }
    torch.save(checkpoint, checkpoint_file)


class EarlyStopping(object):
    
    def __init__(self, mode, patience, logger):
        assert mode in ['higher', 'lower']
        if mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower
        
        self.patience = patience
        self.logger = logger
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def _check_higher(self, score, prev_best_score):
        return score > prev_best_score

    def _check_lower(self, score, prev_best_score):
        return score < prev_best_score

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._check(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            self.logger.info('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
                self.logger.info('EarlyStopping: patience reached. Stopping training ...')
        return self.early_stop


def calculate_top_k_prec(sorted_pred_indices, labels, k):
    """Calculate the top-k interaction precision."""
    num_interactions_to_score = k
    selected_pred_indices = sorted_pred_indices[:num_interactions_to_score]
    true_labels = labels[selected_pred_indices]
    num_correct = torch.sum(true_labels).item()
    prec = num_correct / (num_interactions_to_score + 1e-6)
    return prec


def calculate_top_k_recall(sorted_pred_indices, labels, k):
    """Calculate the top-k interaction recall."""
    num_interactions_to_score = k
    selected_pred_indices = sorted_pred_indices[:num_interactions_to_score]
    true_labels = labels[selected_pred_indices]
    num_correct = torch.sum(true_labels).item()
    num_pos_labels = torch.sum(labels).item()
    recall = num_correct / (num_pos_labels + 1e-6)
    return recall



def evaluate(model, eval_loader, loss_fn):

    model.eval()
    all_top_l_prec = 0.0
    all_top_25_prec, all_top_50_prec, all_top_l_by_5_prec, all_top_l_by_10_prec = 0.0, 0.0, 0.0, 0.0
    all_top_10_prec, all_top_1_prec = 0.0, 0.0

    all_loss = 0.0

    tbar = tqdm(eval_loader)
    tbar.set_description('Evaluation')

    dict_ = {}
    for data in tbar:
        
        rec1d, rec2d, lig1d, lig2d = data['rec1d'].cuda(), data['rec2d'].cuda(), data['lig1d'].cuda(), data['lig2d'].cuda()
        com2d, intra_rec, intra_lig = data['com2d'].cuda(), data['intra_distA'].cuda(), data['intra_distB'].cuda()
        labels = data['flatten_contact_map'].cuda() # [B, L_rec*L_lig]
        pdb_name = data['pdb_name']
        with torch.no_grad():

            logits = model(rec1d, rec2d, lig1d, lig2d, com2d, intra_rec, intra_lig)  # [B, 1, L_rec, L_lig]

            logits = torch.flatten(logits, start_dim=1) # [B, L_rec*L_lig]

            loss = loss_fn(logits, labels)  

            sorted_logits_indices = torch.argsort(logits, descending=True).squeeze()  # [B, L_rec*L_lig]
            labels = labels.squeeze()
            l = min(rec1d.shape[-1], lig1d.shape[-1])

            all_top_1_prec += calculate_top_k_prec(sorted_logits_indices, labels, k=1)
            all_top_10_prec += calculate_top_k_prec(sorted_logits_indices, labels, k=10)
            all_top_25_prec += calculate_top_k_prec(sorted_logits_indices, labels, k=25)
            all_top_50_prec += calculate_top_k_prec(sorted_logits_indices, labels, k=50)
            all_top_l_by_5_prec += calculate_top_k_prec(sorted_logits_indices, labels, k=(l // 5))
            all_top_l_by_10_prec += calculate_top_k_prec(sorted_logits_indices, labels, k=(l // 10))
            all_top_l_prec += calculate_top_k_prec(sorted_logits_indices, labels, k=l)

            all_loss += loss.item()
            dict_[pdb_name] = calculate_top_k_prec(sorted_logits_indices, labels, k=l)
        
    return {
            'top_1_prec': all_top_1_prec / len(eval_loader.dataset), 
            'top_10_prec': all_top_10_prec / len(eval_loader.dataset),
            'top_25_prec': all_top_25_prec / len(eval_loader.dataset),
            'top_50_prec': all_top_50_prec / len(eval_loader.dataset), 
            'top_l_by_10_prec': all_top_l_by_10_prec / len(eval_loader.dataset), 
            'top_l_by_5_prec': all_top_l_by_5_prec / len(eval_loader.dataset),  
            'top_l_prec': all_top_l_prec / len(eval_loader.dataset), 
            'loss': all_loss / len(eval_loader.dataset),
    }


def setup_logger(name, save_dir, filename="log.txt"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger