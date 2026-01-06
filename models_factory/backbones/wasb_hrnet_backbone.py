import torch
import torch.nn as nn
import logging
from ..builder import BACKBONES

# 基础配置
BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 卷积带 padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)

class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index], num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM),
            )
        layers = [block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample)]
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1: return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False),
                        nn.BatchNorm2d(num_inchannels[i]),
                        nn.Upsample(scale_factor=2**(j-i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_inchannels[i], 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_inchannels[i])))
                        else:
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_inchannels[j], 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_inchannels[j]),
                                nn.ReLU(True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                y = y + x[j] if i == j else y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse

@BACKBONES.register_module
class WASBHRNetBackbone(nn.Module):
    blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}

    def __init__(self, frames_in, MODEL): 
        super(WASBHRNetBackbone, self).__init__()
        extra = MODEL['EXTRA']
        
        # [cite_start]Modified Stem: 设置步长以保持分辨率 [cite: 108, 109]
        stem_strides = extra['STEM']['STRIDES']
        stem_inplanes = extra['STEM']['INPLANES']
        
        # [cite_start]输入为 3 * frames_in 通道 [cite: 110]
        self.conv1 = nn.Conv2d(3 * frames_in, stem_inplanes, kernel_size=3, stride=stem_strides[0], padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(stem_inplanes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(stem_inplanes, stem_inplanes, kernel_size=3, stride=stem_strides[1], padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(stem_inplanes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        # Stage 1: [cite_start]单分支 Bottleneck 提取特征 [cite: 106]
        s1_cfg = extra['STAGE1']
        self.layer1 = self._make_layer(self.blocks_dict[s1_cfg['BLOCK']], stem_inplanes, s1_cfg['NUM_CHANNELS'][0], s1_cfg['NUM_BLOCKS'][0])
        pre_channels = [self.blocks_dict[s1_cfg['BLOCK']].expansion * s1_cfg['NUM_CHANNELS'][0]]

        # Stage 2, 3, 4: [cite_start]并行流过渡与生成 [cite: 80, 107]
        self.transition1 = self._make_transition_layer(pre_channels, [c * self.blocks_dict[extra['STAGE2']['BLOCK']].expansion for c in extra['STAGE2']['NUM_CHANNELS']])
        self.stage2, pre_channels = self._make_stage(extra['STAGE2'], [c * self.blocks_dict[extra['STAGE2']['BLOCK']].expansion for c in extra['STAGE2']['NUM_CHANNELS']])

        self.transition2 = self._make_transition_layer(pre_channels, [c * self.blocks_dict[extra['STAGE3']['BLOCK']].expansion for c in extra['STAGE3']['NUM_CHANNELS']])
        self.stage3, pre_channels = self._make_stage(extra['STAGE3'], [c * self.blocks_dict[extra['STAGE3']['BLOCK']].expansion for c in extra['STAGE3']['NUM_CHANNELS']])

        self.transition3 = self._make_transition_layer(pre_channels, [c * self.blocks_dict[extra['STAGE4']['BLOCK']].expansion for c in extra['STAGE4']['NUM_CHANNELS']])
        self.stage4, pre_channels = self._make_stage(extra['STAGE4'], [c * self.blocks_dict[extra['STAGE4']['BLOCK']].expansion for c in extra['STAGE4']['NUM_CHANNELS']], multi_scale_output=True)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM))
        layers = [block(inplanes, planes, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(block(planes * block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_transition_layer(self, num_pre, num_cur):
        layers = []
        for i in range(len(num_cur)):
            if i < len(num_pre):
                if num_cur[i] != num_pre[i]:
                    layers.append(nn.Sequential(nn.Conv2d(num_pre[i], num_cur[i], 3, 1, 1, bias=False),
                                               nn.BatchNorm2d(num_cur[i]), nn.ReLU(True)))
                else: layers.append(None)
            else:
                convs = []
                for j in range(i + 1 - len(num_pre)):
                    in_c, out_c = num_pre[-1], num_cur[i] if j == i - len(num_pre) else num_pre[-1]
                    convs.append(nn.Sequential(nn.Conv2d(in_c, out_c, 3, 2, 1, bias=False),
                                              nn.BatchNorm2d(out_c), nn.ReLU(True)))
                layers.append(nn.Sequential(*convs))
        return nn.ModuleList(layers)

    def _make_stage(self, cfg, in_channels, multi_scale_output=True):
        modules = []
        for i in range(cfg['NUM_MODULES']):
            reset_mso = multi_scale_output or (i < cfg['NUM_MODULES'] - 1)
            modules.append(HighResolutionModule(cfg['NUM_BRANCHES'], self.blocks_dict[cfg['BLOCK']], cfg['NUM_BLOCKS'], 
                                                 in_channels, cfg['NUM_CHANNELS'], cfg['FUSE_METHOD'], reset_mso))
            in_channels = modules[-1].num_inchannels
        return nn.Sequential(*modules), in_channels

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.layer1(x)
        y_list = self.stage2([self.transition1[i](x) if self.transition1[i] else x for i in range(len(self.transition1))])
        y_list = self.stage3([self.transition2[i](y_list[-1]) if self.transition2[i] else y_list[i] for i in range(len(self.transition2))])
        y_list = self.stage4([self.transition3[i](y_list[-1]) if self.transition3[i] else y_list[i] for i in range(len(self.transition3))])
        return y_list[0]