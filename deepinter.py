import torch
import torch.nn as nn

from resnet_inception import ResNet_Inception
from triangle import TriangleSelfAttention, TriangleMultiplication, Transition

class DeepInter(nn.Module):
    
    def __init__(self,
                 in_channels_rec_lig1d,
                 in_channels_rec_lig2d,
                 in_channels_com2d,
                 hidden_channels,
                 num_heads,
                 dropout_rate,
                 num_classes,
                 ):
        super(DeepInter, self).__init__()

        self.res_inception_1 = ResNet_Inception(
            in_channels_rec_lig1d,
            in_channels_rec_lig2d,
            hidden_channels,
        )
        self.res_inception_2 = ResNet_Inception(
            in_channels_rec_lig1d,
            in_channels_com2d,
            hidden_channels,
        )

        self.triangle_attention_rec = nn.ModuleList([TriangleSelfAttention(
            hidden_channels,
            hidden_channels // 2,
            num_heads,
        ) for _ in range(20)])
        
        self.triangle_attention_lig = nn.ModuleList([TriangleSelfAttention(
            hidden_channels,
            hidden_channels // 2,
            num_heads,
        ) for _ in range(20)])

        self.triangle_multi = nn.ModuleList([TriangleMultiplication(
            hidden_channels,
            hidden_channels,
        ) for _ in range(20)])

        self.transition = nn.ModuleList([Transition(
            hidden_channels,
            hidden_channels * 4,
        ) for _ in range(20)])
                
        self.norm = nn.LayerNorm(hidden_channels)

        self.Linear_final = nn.Linear(hidden_channels, num_classes)

        self.activ_fn = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, rec1d, rec2d, lig1d, lig2d, com2d, intra_rec, intra_lig):
        """
        rec: Receptor
        lig: Ligand
        com: Complex

        rec1d: [B, C, L_rec]
        rec2d: [B, C, L_rec, L_rec]
        lig1d: [B, C, L_lig]
        lig2d: [B, C, L_lig, L_lig]
        com2d: [B, C, L_rec, L_lig]
        intra_rec: [B, 1, L_rec, L_rec]
        intra_lig: [B, 1, L_lig, L_lig]
        """
        # print('rec1d', rec1d.shape)
        rec2d = self.res_inception_1(rec1d, rec1d, rec2d)  # [B, C, L_rec, L_rec]
        lig2d = self.res_inception_1(lig1d, lig1d, lig2d)  # [B, C, L_lig, L_lig]
        z_com = self.res_inception_2(rec1d, lig1d, com2d)  # [B, C, L_rec, L_lig]

        rec2d, lig2d, z_com = rec2d.permute(0, 2, 3, 1), lig2d.permute(0, 2, 3, 1), z_com.permute(0, 2, 3, 1)  # rec2d, lig2d: [B, L_rec/lig, L_rec/lig, C], z_com: [B, L_rec, L_lig, C]

        for i in range(20):
            
            z_com = z_com + self.dropout(self.triangle_multi[i](z_com, rec2d, lig2d))
            z_com = z_com + self.dropout(self.triangle_attention_rec[i](z_com, intra_lig))

            z_com_T = z_com.permute(0, 2, 1, 3)  # [B, L_lig, L_rec, C]
            z_com_T = self.triangle_attention_lig[i](z_com_T, intra_rec)
            z_com = z_com + self.dropout(z_com_T.permute(0, 2, 1, 3))
            z_com = z_com + self.transition[i](z_com)
        
        z_final_norm = self.norm(z_com)
        z_final = self.activ_fn(self.Linear_final(z_final_norm))
        z_final = z_final.permute(0, 3, 1, 2)  # [B, C, L_rec, L_lig]
        return z_final



if __name__ == "__main__":

    import numpy as np

    file_ = '/mnt/petrelfs/duhao.d/1A06.npz'
    with open(file_, 'rb') as f:
        data = np.load(file_)

    # import pdb; pdb.set_trace()
    rec1d, rec2d, lig1d, lig2d, com2d, intra_rec, intra_lig = data['rec1d'], data['rec2d'], data['lig1d'], data['lig2d'], data['com2d'], data['intra_distA'], data['intra_distB']
    rec1d, rec2d, lig1d, lig2d, com2d, intra_rec, intra_lig = torch.FloatTensor(rec1d), torch.FloatTensor(rec2d), torch.FloatTensor(lig1d), torch.FloatTensor(lig2d), torch.FloatTensor(com2d), torch.FloatTensor(intra_rec), torch.FloatTensor(intra_lig)
    rec1d, rec2d, lig1d, lig2d, com2d, intra_rec, intra_lig = rec1d.unsqueeze(0), rec2d.unsqueeze(0), lig1d.unsqueeze(0), lig2d.unsqueeze(0), com2d.unsqueeze(0), intra_rec.unsqueeze(0), intra_lig.unsqueeze(0)
    
    # import pdb; pdb.set_trace()

    model = DeepInter(
        in_channels_rec_lig1d = 788,
        in_channels_rec_lig2d = 210,
        in_channels_com2d = 146,
        hidden_channels = 64,
        num_heads = 4,
        dropout_rate = 0.1,
        num_classes = 1,
    )

    model(rec1d, rec2d, lig1d, lig2d, com2d, intra_rec, intra_lig)
    
