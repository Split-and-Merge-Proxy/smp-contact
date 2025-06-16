import torch
import torch.nn as nn

from utils import seq2pairwise


class BasicBlock(nn.Module):

    def __init__(self, 
                 in_channels, 
                 out_channels):
        super(BasicBlock, self).__init__()

        self.conv_list_1 = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=(1,9), stride=1, padding=(0,4), bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=(9,1), stride=1, padding=(4,0), bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=1, padding=(1,1), bias=False)
        ])

        self.conv_list_2 = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=(1,9), stride=1, padding=(0,4), bias=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=(9,1), stride=1, padding=(4,0), bias=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), stride=1, padding=(1,1), bias=False)
        ])

        self.norm_list_1 = nn.ModuleList([nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=False) for _ in range(3)])
        self.norm_list_2 = nn.ModuleList([nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=False) for _ in range(3)])

        self.activ_fn_list = nn.ModuleList([nn.LeakyReLU(negative_slope=0.01, inplace=True) for _ in range(3)])

        self.activ_fn = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        
        res = None
        for i in range(3):
            
            xi = self.conv_list_1[i](x)
            xi = self.norm_list_1[i](xi)
            xi = self.activ_fn_list[i](xi)

            xi = self.conv_list_2[i](xi)
            xi = self.norm_list_2[i](xi)

            if res is None:
                res = xi
            else:
                res = res + xi
        
        return self.activ_fn(res + x)


class ResNet_Inception(nn.Module):
    
    def __init__(self, 
                 in_channels_1d, 
                 in_channels_2d,
                 hidden_channels):

        super(ResNet_Inception, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels_1d*2, hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(hidden_channels, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels_2d, hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(hidden_channels, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_channels*2, hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(hidden_channels, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )

        self.res_inception = nn.ModuleList([BasicBlock(hidden_channels, hidden_channels) for _ in range(4)])

        self.init_parameters()

    def init_parameters(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
    

    def forward(self, rec_lig1d_1, rec_lig1d_2, rec_lig_com2d):
        """
        rec_lig1d_1: rec/lig1d feature
        rec_lig1d_2: rec/lig1d feature
        rec_lig_com2d: rec/lig/com2d feature
        """
        pair_1 = seq2pairwise(rec_lig1d_1, rec_lig1d_2)
        pair_1 = self.conv1(pair_1)

        pair_2 = self.conv2(rec_lig_com2d)

        pair = torch.cat([pair_1, pair_2], dim=1)
        out = self.conv3(pair)

        for i in range(4):
            out = self.res_inception[i](out)

        return out
