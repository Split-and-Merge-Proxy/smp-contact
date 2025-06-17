import torch
import numpy as np
from deepinter import DeepInter


def build_model():
    model = DeepInter(
        in_channels_rec_lig1d = 788,
        in_channels_rec_lig2d = 210,
        in_channels_com2d = 146,
        hidden_channels = 64,
        num_heads = 4,
        dropout_rate = 0.1,
        num_classes = 1,
    )
    
    return model

def inference(model, data):
    # data
    rec1d, rec2d, lig1d, lig2d = torch.tensor(data['rec1d']).cuda(), torch.tensor(data['rec2d']).cuda(), torch.tensor(data['lig1d']).cuda(), torch.tensor(data['lig2d']).cuda()
    com2d, intra_rec, intra_lig = torch.tensor(data['com2d']).cuda(), torch.tensor(data['intra_distA']).cuda(), torch.tensor(data['intra_distB']).cuda()

    rec1d, rec2d, lig1d, lig2d, com2d, intra_rec, intra_lig = rec1d.unsqueeze(0), rec2d.unsqueeze(0), lig1d.unsqueeze(0), lig2d.unsqueeze(0), com2d.unsqueeze(0), intra_rec.unsqueeze(0), intra_lig.unsqueeze(0)
    # labels = torch.tensor(data['flatten_contact_map']) # [B, L_rec*L_lig]
    # labels = labels.unsqueeze(0)

    with torch.no_grad():
        logits = model(rec1d, rec2d, lig1d, lig2d, com2d, intra_rec, intra_lig)  # [B, 1, L_rec, L_lig]
        logits = logits.squeeze().cpu()
        contact_map = (logits > 0.5).int().numpy()
    
    return contact_map


if __name__ == "__main__":

    data_path = './example/8SMQ.npz'
    ckpt_path = './ckpts/smp_hetero.pth'  # you can use smp_homo.pth for homodimers
    ckpt = torch.load(ckpt_path, map_location='cuda:0')

    # load model
    model = build_model()
    model = model.cuda()
    model.load_state_dict(ckpt['net'])
    model.eval()

    # load data
    data = dict(np.load(data_path))

    # inference
    contact_map = inference(model, data)
    np.save('./contact_map.npy', contact_map)