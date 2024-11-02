import torch.nn
from torch import nn

import fvcore.nn.weight_init as weight_init
import torch.nn.functional as F

from mmcv.ops import ModulatedDeformConv2d

class Encoder(nn.Module):
    def __init__(self,
                 input_channels,
                 n_stages,
                 features_per_stage
                 ):
        super().__init__()

        Conv_layers = []
        Down_layers = []

        for s in range(n_stages):
            if s > 0:
                Down_layers.append(DCD(input_channels))
            Conv_layers.append(DCC(input_channels, features_per_stage[s]))

            input_channels = features_per_stage[s]

        self.n_stages = n_stages
        self.Down_layers = Down_layers
        self.Conv_layers = Conv_layers
        self.output_channels = features_per_stage

    def forward(self, x):
        ret = []
        for i in range(self.n_stages):
            if i > 0:
                x = self.Down_layers[i][x]
            x = self.Conv_layers[i](x)
            ret.append(x)
        return ret

class Decoder(nn.Module):
    def __init__(self,
                 encoder: Encoder,
                 num_classes,
                 deep_supervision):
                 
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)

        Conv_layers = []
        Up_layers = []
        seg_layers = []

        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]

            Up_layers.append(DCU(input_features_below, input_features_skip))
            Conv_layers.append(DCC(input_features_skip * 2, input_features_skip))
            seg_layers.append(nn.Conv2d(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.Conv_layers = nn.ModuleList(Conv_layers)
        self.Up_layers = nn.ModuleList(Up_layers)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for i in range(len(self.Conv_layers)):
            x = self.Up_layers[i](lres_input, skips[-(i+2)])
            x = self.Conv_layers[i](x)

            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[i](x))
            elif i == (len(self.Conv_layers) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r


class DCU(nn.Module):
    def __init__(self, 
                 input_channels: int,
                 output_channels: int):
        super().__init__()

        self.upsampling = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        
        kernel_size = (3, 3)
        self.kernel_field = kernel_size[0] * kernel_size[1]
        num_offset_channel = 3 * kernel_size[0] * kernel_size[1]
        self.offset_conv = nn.Conv2d(output_channels*2, num_offset_channel, kernel_size=3, stride=1, padding=1, bias=True)

        self.nonlin = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.align_conv = ModulatedDeformConv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1)
   
        weight_init.c2_xavier_fill(self.offset_conv)

    def forward(self, high_feature, low_feature):
        upsampled_feature = self.upsampling(high_feature)
        feature_offset = torch.cat((upsampled_feature, low_feature), dim=1)
        feature_offset = self.offset_conv(feature_offset)
        offset, mask = torch.split(feature_offset, [self.kernel_field * 2, self.kernel_field], dim=1)
        upsampled_feature = self.align_conv(upsampled_feature, offset.contiguous(), 2 * torch.sigmoid(mask.contiguous()))
        upsampled_feature = self.nonlin(upsampled_feature)
        output = torch.cat((upsampled_feature, low_feature), dim=1)
        return output

class DCD(nn.Module):
    def __init__(self, input_channels: int):
        super().__init__()
        
        kernel_size = (3, 3)
        self.kernel_field = kernel_size[0] * kernel_size[1]
        num_offset_channel = 3 * kernel_size[0] * kernel_size[1]
        self.offset_conv = nn.Conv2d(input_channels, num_offset_channel, kernel_size=3, stride=1, padding=1, bias=True)

        self.align_conv = ModulatedDeformConv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1)
        self.nonlin = nn.Sigmoid()

        self.pooling = nn.AvgPool2d(kernel_size=3, stride = 2, padding = 1)

        weight_init.c2_xavier_fill(self.offset_conv)
        
    def forward(self, x):
        offset = self.offset_conv(x)
        offset, mask = torch.split(offset, [self.kernel_field * 2, self.kernel_field], dim=1)
        w = self.align_conv(x, offset.contiguous(), 2 * torch.sigmoid(mask.contiguous()))
        w = self.nonlin(w)
        w = w.exp()
        output = self.pooling(w * x)/self.pooling(w)
        return output

class DCC(nn.Module):
    def __init__(self,
                 input_channels: int, 
                 output_channels: int
                 ):
        super().__init__()

        self.ge = nn.Sequential(
                    nn.AvgPool2d(kernel_size=4, stride=4), 
                    nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)
                    )

        self.att_conv = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1)

        self.nonlin = nn.Sigmoid()

        self.conv = nn.Sequential(
                    nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.BatchNorm2d(output_channels)
                    )

        self.next_conv = nn.Sequential(
                    nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.BatchNorm2d(output_channels)
                    )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(output_channels, output_channels), 
            nn.Sigmoid()
        )

    def forward(self, x):
        att1 = F.interpolate(self.ge(x), x.size()[2:])
        att2 = self.att_conv(x)
        att = torch.add(att1, att2)
        att = self.nonlin(att)
        output = self.conv(x)
        output = output * att
        output = self.next_conv(output)
        ch = self.fc(self.avg_pool(output).squeeze()).unsqueeze(dim=-1).unsqueeze(dim=-1)
        output = output * ch
        return output
