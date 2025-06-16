import torch
import torch.nn as nn
import torch.nn.functional as F

class TriangleSelfAttention(nn.Module):

    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 num_heads):
        super(TriangleSelfAttention, self).__init__()

        self.norm = nn.LayerNorm(in_channels)
        self.Linear_Q = nn.Linear(in_channels, out_channels)
        self.Linear_K = nn.Linear(in_channels, out_channels)
        self.Linear_V = nn.Linear(in_channels, out_channels)

        self.Linear_gate = nn.Linear(in_channels, out_channels)

        self.Linear_final = nn.Linear(out_channels, in_channels)

        self.num_heads = num_heads
        self.out_channels = out_channels


    def forward(self, com2d, intra_dist):
        """
        com2d: [B, L_rec, L_lig, C]
        instra_dist: [B, 1, L_rec/lig, L_rec/lig]
        """
        B, H, W, _ = com2d.shape

        com2d = self.norm(com2d)
        com2d_q = self.Linear_Q(com2d).view(B, H, W, self.num_heads, self.out_channels // self.num_heads)  # [B, H, W, num_heads, out_channels // num_heads]
        com2d_k = self.Linear_K(com2d).view(B, H, W, self.num_heads, self.out_channels // self.num_heads)
        com2d_v = self.Linear_V(com2d).view(B, H, W, self.num_heads, self.out_channels // self.num_heads)

        scalar = torch.sqrt(torch.tensor(1.0 / (self.out_channels // self.num_heads)))
        coef = torch.exp(-(intra_dist / 8.0)** 2.0 / 2.0).unsqueeze(2).type_as(com2d_q)  # [B, 1, 1, H, W]

        # atten_scores = torch.matmul(com2d_q.transpose(0, 3, 1, 2, 4), com2d_k.transpose(0, 3, 1, 4, 2))  # [B, num_heads, H, W, W]
        atten_scores = torch.einsum("bnihc, bnjhc->bhnij", com2d_q * scalar, com2d_k) # [B, num_heads, H, W, W]
        atten_scores = atten_scores * coef
        atten_scores = F.softmax(atten_scores, dim=-1)
        com2d_v = torch.einsum("bhnij, bnjhc->bnihc", atten_scores, com2d_v)  # [B, H, W, num_heads, out_channels // num_heads]
        com2d_gate = self.Linear_gate(com2d).view(B, H, W, self.num_heads, self.out_channels // self.num_heads)  # [B, H, W, num_heads, out_channels // num_heads]

        com2d_final = (com2d_v * com2d_gate).contiguous().view(B, H, W, -1)
        com2d_final = self.Linear_final(com2d_final)
        return com2d_final

class TriangleMultiplication(nn.Module):
    
    def __init__(self,
                 in_channels,
                 out_channels):
        super(TriangleMultiplication, self).__init__()

        self.norm_com = nn.LayerNorm(in_channels)
        self.norm_rec = nn.LayerNorm(in_channels)
        self.norm_lig = nn.LayerNorm(in_channels)

        self.Linear_com_1 = nn.Linear(in_channels, out_channels)
        self.gate_com_1 = nn.Linear(in_channels, out_channels)
        self.Linear_com_2 = nn.Linear(in_channels, out_channels)
        self.gate_com_2 = nn.Linear(in_channels, out_channels)

        self.Linear_rec = nn.Linear(in_channels, out_channels)
        self.gate_rec = nn.Linear(in_channels, out_channels)
        self.Linear_lig = nn.Linear(in_channels, out_channels)
        self.gate_lig = nn.Linear(in_channels, out_channels)

        self.norm_all = nn.LayerNorm(out_channels)
        self.Linear_all = nn.Linear(out_channels, in_channels)
        self.gate_all = nn.Linear(in_channels, in_channels)

    
    def forward(self, com2d, rec2d, lig2d):
        """
        com2d: [B, L_rec, L_lig, C]
        rec2d: [B, L_rec, L_rec, C]
        lig2d: [B, L_lig, L_lig, C]
        """
        z_com = self.norm_com(com2d)
        z_rec = self.norm_rec(rec2d)
        z_lig = self.norm_lig(lig2d)

        z_com_1 = self.Linear_com_1(z_com)
        z_com_self_atten_1 = torch.sigmoid(self.gate_com_1(z_com))
        z_com_1 = z_com_1 * z_com_self_atten_1 # [B, L_rec, L_lig, C]
        z_com_2 = self.Linear_com_2(z_com)
        z_com_self_atten_2 = torch.sigmoid(self.gate_com_2(z_com))
        z_com_2 = z_com_2 * z_com_self_atten_2 # [B, L_rec, L_lig, C]

        z_rec = self.Linear_rec(z_rec) * torch.sigmoid(self.gate_rec(z_rec)) # [B, L_rec, L_rec, C]
        z_lig = self.Linear_lig(z_lig) * torch.sigmoid(self.gate_lig(z_lig)) # [B, L_lig, L_lig, C]

        z_com_rec = torch.einsum("bikc, bkjc->bijc", z_rec, z_com_1)
        z_com_lig = torch.einsum("bikc, bjkc->bjic", z_lig, z_com_2)
        z_com_all = z_com_rec + z_com_lig

        z_com = self.Linear_all(self.norm_all(z_com_all)) * torch.sigmoid(self.gate_all(z_com))

        return z_com


class Transition(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Transition, self).__init__()

        self.norm = nn.LayerNorm(in_channels)
        self.transition = nn.Sequential(nn.Linear(in_channels, hidden_channels),
                                        nn.ReLU(),
                                        nn.Linear(hidden_channels, in_channels))
    
    def forward(self, x):
        """
        x: []
        """
        x = self.norm(x)
        x = self.transition(x)

        return x