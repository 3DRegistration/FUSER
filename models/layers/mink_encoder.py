import torch
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF

class MinkEncoder(ME.MinkowskiNetwork):
    NORM_TYPE = 'IN'
    CHANNELS = [None, 32, 64, 128, 256, 1024]
    BLOCK_NORM_TYPE = 'IN'
    def __init__(self, in_feats_dim, conv1_kernel_size, bn_momentum, normalize_feature, D=3):
        ME.MinkowskiNetwork.__init__(self, D)
        NORM_TYPE = self.NORM_TYPE
        BLOCK_NORM_TYPE = self.BLOCK_NORM_TYPE
        CHANNELS = self.CHANNELS
        bn_momentum = bn_momentum
        self.normalize_feature = normalize_feature

        self.conv1 = ME.MinkowskiConvolution(
            in_channels=in_feats_dim,
            out_channels=CHANNELS[1],
            kernel_size=conv1_kernel_size,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm1 = get_norm(NORM_TYPE, CHANNELS[1], bn_momentum=bn_momentum, D=D)

        self.block1 = get_block(
            BLOCK_NORM_TYPE, CHANNELS[1], CHANNELS[1], bn_momentum=bn_momentum, D=D)

        self.conv2 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[1],
            out_channels=CHANNELS[2],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm2 = get_norm(NORM_TYPE, CHANNELS[2], bn_momentum=bn_momentum, D=D)

        self.block2 = get_block(
            BLOCK_NORM_TYPE, CHANNELS[2], CHANNELS[2], bn_momentum=bn_momentum, D=D)

        self.conv3 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[2],
            out_channels=CHANNELS[3],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm3 = get_norm(NORM_TYPE, CHANNELS[3], bn_momentum=bn_momentum, D=D)

        self.block3 = get_block(
            BLOCK_NORM_TYPE, CHANNELS[3], CHANNELS[3], bn_momentum=bn_momentum, D=D)

        self.conv4 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[3],
            out_channels=CHANNELS[4],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm4 = get_norm(NORM_TYPE, CHANNELS[4], bn_momentum=bn_momentum, D=D)

        self.block4 = get_block(
            BLOCK_NORM_TYPE, CHANNELS[4], CHANNELS[4], bn_momentum=bn_momentum, D=D)

        self.conv5 = ME.MinkowskiConvolution(
            in_channels=CHANNELS[4],
            out_channels=CHANNELS[5],
            kernel_size=3,
            stride=2,
            dilation=1,
            bias=False,
            dimension=D)
        self.norm5 = get_norm(NORM_TYPE, CHANNELS[5], bn_momentum=bn_momentum, D=D)

        self.block5 = get_block(
            BLOCK_NORM_TYPE, CHANNELS[5], CHANNELS[5], bn_momentum=bn_momentum, D=D)

    def forward(self, stensor):

        ################################
        # encode src
        s1 = self.conv1(stensor)
        s1 = self.norm1(s1)
        s1 = self.block1(s1)
        src = MEF.relu(s1)

        s2 = self.conv2(src)
        s2 = self.norm2(s2)
        s2 = self.block2(s2)
        src = MEF.relu(s2)

        s4 = self.conv3(src)
        s4 = self.norm3(s4)
        s4 = self.block3(s4)
        src = MEF.relu(s4)

        s8 = self.conv4(src)
        s8 = self.norm4(s8)
        s8 = self.block4(s8)
        src = MEF.relu(s8)

        s16 = self.conv5(src)
        s16 = self.norm5(s16)
        s16 = self.block5(s16)
        src = MEF.relu(s16)

        x=src.C[:, 0]
        # counts = torch.bincount(torch.cumsum(torch.cat([torch.tensor([1]).to(x.device), (x[1:] != x[:-1]).int()]), dim=0))
        # slens_c = counts.tolist()[1:]

        slens_c = torch.bincount(x).tolist()
        pcd_c = src.C[:, 1:]
        return src.F, slens_c, pcd_c

def get_norm(norm_type, num_feats, bn_momentum=0.05, D=-1):
    if norm_type == 'BN':
        return ME.MinkowskiBatchNorm(num_feats, momentum=bn_momentum)
    elif norm_type == 'IN':
        return ME.MinkowskiInstanceNorm(num_feats)
    else:
        raise ValueError(f'Type {norm_type}, not defined')

class BasicBlockBase(torch.nn.Module):
    expansion = 1
    NORM_TYPE = 'BN'
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, bn_momentum=0.1, D=3):
        super(BasicBlockBase, self).__init__()
        self.conv1 = ME.MinkowskiConvolution(inplanes, planes, kernel_size=3, stride=stride, dimension=D)
        self.norm1 = get_norm(self.NORM_TYPE, planes, bn_momentum=bn_momentum, D=D)
        self.conv2 = ME.MinkowskiConvolution(planes, planes, kernel_size=3, stride=1, dilation=dilation, bias=False, dimension=D)
        self.norm2 = get_norm(self.NORM_TYPE, planes, bn_momentum=bn_momentum, D=D)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = MEF.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = MEF.relu(out)
        return out

class BasicBlockBN(BasicBlockBase):
  NORM_TYPE = 'BN'

class BasicBlockIN(BasicBlockBase):
  NORM_TYPE = 'IN'

def get_block(norm_type,
              inplanes,
              planes,
              stride=1,
              dilation=1,
              downsample=None,
              bn_momentum=0.1,
              D=3):
  if norm_type == 'BN':
    return BasicBlockBN(inplanes, planes, stride, dilation, downsample, bn_momentum, D)
  elif norm_type == 'IN':
    return BasicBlockIN(inplanes, planes, stride, dilation, downsample, bn_momentum, D)
  else:
    raise ValueError(f'Type {norm_type}, not defined')