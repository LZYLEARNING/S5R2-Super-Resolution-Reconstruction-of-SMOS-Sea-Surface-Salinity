# Structure: Unit => Layer => Box => Block => Part => Body => Model

import math
import torch

# from torchvision.transforms.functional import to_pil_image
# import numpy as np
# from PIL import Image
# from matplotlib import pyplot as plt

'''
# 保存中途输出
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
'''


# Unit #################################################################################################################
class ScaleUnit(torch.nn.Module):
    def __init__(self,
                 init_value=1e-3):
        super(ScaleUnit, self).__init__()
        self.scale = torch.nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, x):
        device = x.device
        return x * self.scale.to(device)


# Layer ################################################################################################################
class ConvLayer(torch.nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel_size,
                 stride,
                 padding,
                 groups):
        super(ConvLayer, self).__init__()

        self.conv = torch.nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    groups=groups)

    def forward(self, x):
        return self.conv(x)


class ActLayer(torch.nn.Module):
    def __init__(self,
                 act_type,
                 n_prelu=1,
                 inplace=True,
                 neg_slope=0.05):
        super(ActLayer, self).__init__()

        self.act_type = act_type.lower()
        self.inplace = inplace
        self.neg_slope = neg_slope
        self.n_prelu = n_prelu
        self.layer = self._get_activation_layer()

    def _get_activation_layer(self):
        if self.act_type == 'relu':
            return torch.nn.ReLU(self.inplace)
        elif self.act_type == 'lrelu':
            return torch.nn.LeakyReLU(self.neg_slope, self.inplace)
        elif self.act_type == 'prelu':
            return torch.nn.PReLU(num_parameters=self.n_prelu, init=self.neg_slope)
        else:
            raise NotImplementedError('activation layer [{:s}] is not found'.format(self.act_type))

    def forward(self, x):
        return self.layer(x)


class NormLayer(torch.nn.Module):
    def __init__(self,
                 norm_type,
                 in_ch,
                 num_groups=8):
        super(NormLayer, self).__init__()

        self.norm_type = norm_type.lower()
        self.in_ch = in_ch
        self.num_groups = num_groups
        self.layer = self._get_normalization_layer()

    def _get_normalization_layer(self):
        if self.norm_type == 'batch':
            return torch.nn.BatchNorm2d(self.in_ch, affine=True)
        elif self.norm_type == 'instance':
            return torch.nn.InstanceNorm2d(self.in_ch, affine=False)
        elif self.norm_type == 'group':
            return torch.nn.GroupNorm(self.num_groups, self.in_ch)  # num_groups指的是将输入通道分成几组，num_groups≤in_ch
        elif self.norm_type == 'layer':
            return torch.nn.LayerNorm(normalized_shape=self.in_ch)
        else:
            raise NotImplementedError('normalization layer [{:s}] is not found'.format(self.norm_type))

    def forward(self, x):
        return self.layer(x)


# Box ##################################################################################################################
class ConvBox(torch.nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel_size,
                 stride,
                 padding,
                 groups,
                 act_type,
                 norm_type):
        super(ConvBox, self).__init__()

        self.conv = ConvLayer(in_ch=in_ch,
                              out_ch=out_ch,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups)
        self.norm = NormLayer(norm_type=norm_type,
                              in_ch=out_ch) if norm_type else None
        self.act = ActLayer(act_type=act_type,
                            n_prelu=out_ch) if act_type else None

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)

        return x


class ReshapeBox(torch.nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch):
        super(ReshapeBox, self).__init__()

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_ch,
                            out_channels=out_ch,
                            kernel_size=2,
                            stride=2,
                            padding=1)) if in_ch < out_ch else None

        self.deconv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=in_ch,
                                     out_channels=out_ch,
                                     kernel_size=2,
                                     stride=2,
                                     padding=1)) if in_ch > out_ch else None

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        if self.deconv is not None:
            x = self.deconv(x)

        return x


class MlpBox(torch.nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_type='relu',
                 mlp_drop_rate=0.4):
        super(MlpBox, self).__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features // 4
        self.fc1 = torch.nn.Linear(in_features, hidden_features)
        self.act = ActLayer(act_type=act_type) if act_type else None
        self.fc2 = torch.nn.Linear(hidden_features, out_features)
        self.drop = torch.nn.Dropout(mlp_drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.drop(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class InceptionBox(torch.nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 groups,
                 kernel_sizes,
                 act_type,
                 norm_type):
        super(InceptionBox, self).__init__()

        assert out_ch % len(kernel_sizes) == 0, "InceptionBox: out_ch应当是kernel_sizes列表元素个数的非0整数倍"
        self.kernel_sizes = kernel_sizes
        self.branches = torch.nn.ModuleList()

        # [3组卷积]
        for k in range(len(kernel_sizes) - 1):
            kernel_size = kernel_sizes[k]
            branch_layers = torch.nn.ModuleList()
            branch_layers.append(ConvBox(in_ch=in_ch,
                                         out_ch=out_ch // len(kernel_sizes),
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         groups=groups[0],
                                         act_type=act_type,
                                         norm_type=norm_type))
            if k != 0:
                branch_layers.append(ConvBox(in_ch=out_ch // len(kernel_sizes),
                                             out_ch=out_ch // len(kernel_sizes),
                                             kernel_size=kernel_size,
                                             stride=1,
                                             padding=kernel_size // 2,
                                             groups=groups[1],
                                             act_type=act_type,
                                             norm_type=norm_type))
            self.branches.append(branch_layers)

        # [1组池化]
        branch_layers = torch.nn.ModuleList()
        branch_layers.append(
            torch.nn.MaxPool2d(kernel_size=kernel_sizes[-1], stride=1, padding=kernel_sizes[-1] // 2), )
        branch_layers.append(ConvBox(in_ch=in_ch,
                                     out_ch=out_ch // len(kernel_sizes),
                                     kernel_size=1,
                                     stride=1,
                                     padding=0,
                                     groups=groups[0],
                                     act_type=act_type,
                                     norm_type=norm_type))
        self.branches.append(branch_layers)

    def forward(self, x):
        branch_outputs = []
        for i, branch in enumerate(self.branches):
            branch_output = x
            for layer in branch:
                branch_output = layer(branch_output)
            branch_outputs.append(branch_output)
        output = torch.cat(branch_outputs, dim=1)

        return output


class HFBox(torch.nn.Module):  # 提取高频特征Hff
    def __init__(self):
        super(HFBox, self).__init__()

        self.avg_pool = torch.nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        x_pooled = self.avg_pool(x)
        y = torch.nn.functional.interpolate(x_pooled, size=(x.shape[2], x.shape[3]),
                                            mode='nearest')  # 最近邻插值效果类似于反池化，但是更适合与这些边长不是2的整数的
        return x - y, x_pooled


class EffAttentionBox(torch.nn.Module):
    def __init__(self,
                 in_ch,
                 num_heads=8,
                 qkv_bias=False,
                 linear_drop_rate=0.5,
                 attn_drop_rate=0.5,
                 cut_rate=4,
                 dim_down=5):
        super(EffAttentionBox, self).__init__()

        self.in_ch = in_ch
        self.dim_down = dim_down
        self.cut_rate = cut_rate
        self.num_heads = num_heads
        self.head_dim = in_ch // num_heads
        self.scale = self.head_dim ** -0.5
        self.linear_drop = torch.nn.Dropout(linear_drop_rate)
        self.attn_drop = torch.nn.Dropout(attn_drop_rate)
        self.linear_dim_down = torch.nn.Linear(in_ch, in_ch // dim_down, bias=qkv_bias)
        self.linear_qkv = torch.nn.Linear(in_ch // dim_down, in_ch // dim_down * 3, bias=qkv_bias)
        self.linear_dim_up = torch.nn.Linear(in_ch // dim_down, in_ch)

    def forward(self, x):
        x = self.linear_drop(x)
        x = self.linear_dim_down(x)
        B, N, C = x.shape
        # 输入self.linear_qkv(x)的x通道数C_inqkv为 k * k * x_in_ch // dim_down
        # 其中self.in_ch就是这里的k * k * x_in_ch
        # 应当为self.num_heads的整数倍, 例如当k=2, in_ch=15, dim_down=3,
        # 则C_inqkv=60
        # 所以, C_inqkv必须可以整除self.num_heads

        assert (self.in_ch * 3 // self.dim_down) % self.num_heads == 0, (
            f"输入qkv全连接层的通道数C_inqkv必须整除self.num_heads,"
            f"\nC_inqkv: {self.in_ch * 3 // self.dim_down},"
            f"\nself.num_heads: {self.num_heads}")

        qkv = self.linear_qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()

        q, k, v = qkv[0], qkv[1], qkv[2]
        q_all = torch.split(q, math.ceil(N // self.cut_rate), dim=-2)  # 最后，返回的结果是一个包含这些分割块的列表。
        k_all = torch.split(k, math.ceil(N // self.cut_rate), dim=-2)  # 所以，q_all，k_all，v_all是长度为4的列表
        v_all = torch.split(v, math.ceil(N // self.cut_rate), dim=-2)  # 每块大小为：math.ceil(N // 4)

        output = []
        for q, k, v in zip(q_all, k_all, v_all):
            attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale  # 16*8*37*37
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            trans_x = (attn @ v).transpose(1, 2).contiguous()  # .reshape(B, N, C)
            output.append(trans_x)

        # dropout，就是对此时输入dropout的x按照一定的几率，将一些特征维度的值设置为0
        x = torch.cat(output, dim=1)
        x = x.reshape(B, N, C)
        x = self.linear_drop(x)
        x = self.linear_dim_up(x)

        return x


# Block ################################################################################################################
class CNNStackBlock(torch.nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 groups,  # groups=(1, (1,1))
                 kernel_sizes,
                 incp_num=4,
                 ele_num=5,
                 act_type=(None, None),
                 norm_type=(None, None),
                 use_cc=None,
                 use_cc_ch=None):
        super(CNNStackBlock, self).__init__()

        # [some_params]
        self.incp_num = incp_num
        self.w1 = []  # ScaleUnit(1)
        self.w2 = []  # ScaleUnit(1)

        # [use_cc]
        if use_cc is None:
            use_cc = [True, "prelu"]
        self.use_cc = use_cc
        self.act_cc = ActLayer(act_type=self.use_cc[1], n_prelu=out_ch // ele_num) if self.use_cc[1] else None

        # [use_cc_ch]
        if use_cc_ch is None:
            use_cc_ch = [True, "batch", "prelu"]
        self.use_cc_ch = use_cc_ch
        self.norm = NormLayer(norm_type=self.use_cc_ch[1],
                              in_ch=(out_ch // ele_num) * (self.incp_num + 1)) if self.use_cc_ch[1] else None
        self.act = ActLayer(act_type=self.use_cc_ch[2], n_prelu=(out_ch // ele_num) * (self.incp_num + 1)) if \
            self.use_cc_ch[2] else None
        self.conv1x1 = None
        if self.use_cc_ch[0]:
            self.conv1x1 = ConvLayer(in_ch=(out_ch // ele_num) * (self.incp_num + 1),
                                     out_ch=out_ch // ele_num,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0,
                                     groups=groups[0])
        else:
            self.conv1x1 = ConvLayer(in_ch=out_ch // ele_num,
                                     out_ch=out_ch // ele_num,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0,
                                     groups=groups[0])

        # [cnn_stack_block]
        self.stack_blocks = torch.nn.ModuleList()
        for _ in range(ele_num):
            stack_block = torch.nn.ModuleList()
            stack_block.append(ConvBox(in_ch=in_ch // ele_num,  # in_ch必须等于ele_num, 这里实际上就是1
                                       out_ch=out_ch // ele_num,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       groups=1,
                                       act_type=act_type[0],
                                       norm_type=norm_type[0]))
            for _ in range(incp_num):
                stack_block.append(InceptionBox(in_ch=out_ch // ele_num,
                                                out_ch=out_ch // ele_num,
                                                groups=groups[1],
                                                kernel_sizes=kernel_sizes,
                                                act_type=act_type[1],
                                                norm_type=norm_type[1]))
                # 在inception中每层卷积，还会out_ch//ele_num//len(kernel_sizes)，
                # 所以out_ch必须是ele_num*len(kernel_sizes)的整数倍

                self.w1.append(ScaleUnit(1))
                self.w2.append(ScaleUnit(1))
            self.stack_blocks.append(stack_block)

        assert in_ch == ele_num, "CNNStackBlock: in_ch输入通道数等于ele_num要素个数"
        assert out_ch % ele_num == 0, "CNNStackBlock: out_ch应当是ele_num的非0整数倍"

    def forward(self, X):
        assert len(X) == len(self.stack_blocks), "X的要素个数要与stack_blocks的模块个数一致！"
        outputs = []

        for x, block in zip(X, self.stack_blocks):
            x = block[0](x)

            channel_concat = []
            if self.use_cc_ch[0]:
                channel_concat.append(x)

            for j in range(self.incp_num):
                if self.use_cc[0]:
                    x = self.w1[j](block[j + 1](x)) + self.w2[j](x)
                    if self.act_cc is not None:
                        x = self.act_cc(x)
                else:
                    x = block[j + 1](x)

                if self.use_cc_ch[0]:
                    channel_concat.append(x)

            if self.use_cc_ch[0]:
                x = torch.cat(channel_concat, 1)
                if self.norm is not None:
                    x = self.norm(x)
                if self.act is not None:
                    x = self.act(x)
                x = self.conv1x1(x)

            outputs.append(x)

        return outputs


class FMBlock(torch.nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 groups,  # groups=(1, 1, (1,1))
                 kernel_sizes,
                 incp_num=4,
                 act_type=None,
                 norm_type=None,
                 use_cc=None,
                 use_cc_ch=None):
        super(FMBlock, self).__init__()

        # [w权重]
        self.w1 = []
        self.w2 = []

        # [use_cc]
        if use_cc is None:
            use_cc = [True, "prelu"]
        self.use_cc = use_cc
        self.act_cc = ActLayer(act_type=self.use_cc[1], n_prelu=in_ch) if self.use_cc[1] else None

        # [use_cc_ch]
        if use_cc_ch is None:
            use_cc_ch = [True, "batch", "prelu"]
        self.use_cc_ch = use_cc_ch
        # [通道堆叠具体的3层]
        self.norm = NormLayer(norm_type=self.use_cc_ch[1],
                              in_ch=in_ch * (incp_num + 1)) if self.use_cc_ch[1] else None
        self.act = ActLayer(act_type=self.use_cc_ch[2],
                            n_prelu=in_ch * (incp_num + 1)) if self.use_cc_ch[2] else None
        self.conv1x1 = None
        if self.use_cc_ch[0]:
            self.conv1x1 = ConvLayer(in_ch=in_ch * (incp_num + 1),
                                     out_ch=out_ch,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0,
                                     groups=groups[0])
        else:
            self.conv1x1 = ConvLayer(in_ch=in_ch,
                                     out_ch=out_ch,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0,
                                     groups=groups[0])

        # [FMBlock]
        self.FMBlock = torch.nn.ModuleList()
        for _ in range(incp_num):
            self.FMBlock.append(InceptionBox(in_ch=in_ch,
                                             out_ch=in_ch,
                                             groups=groups[2],
                                             kernel_sizes=kernel_sizes,
                                             act_type=act_type,
                                             norm_type=norm_type))
            self.w1.append(ScaleUnit(1))
            self.w2.append(ScaleUnit(1))

    def forward(self, x):
        f = 0
        channel_concat = []
        if self.use_cc_ch[0]:
            channel_concat.append(x)

        for fm in self.FMBlock:
            if self.use_cc[0]:
                x = self.w1[f](fm(x)) + self.w2[f](x)
                if self.act_cc is not None:
                    x = self.act_cc(x)  # 果然残差连接后相加的结果需要激活一下
            else:
                x = fm(x)
            if self.use_cc_ch[0]:
                channel_concat.append(x)
            f = f + 1

        if self.use_cc_ch[0]:
            x = torch.cat(channel_concat, 1)
            if self.norm is not None:
                x = self.norm(x)
            if self.act is not None:
                x = self.act(x)
        x = self.conv1x1(x)

        return x


class CABlock(torch.nn.Module):
    def __init__(self,
                 in_ch,
                 relu_a=0.01,
                 r=2, ):
        super(CABlock, self).__init__()

        self.mlp_ops = [
            torch.nn.Linear(in_ch, in_ch // r),
            torch.nn.LeakyReLU(negative_slope=relu_a),
            torch.nn.Linear(in_ch // r, in_ch), ]
        self.mlp = torch.nn.Sequential(*self.mlp_ops)
        self.out_act = torch.nn.Sigmoid()

    def forward(self, x, olm, ret_att=False):
        # [提取海洋部分]
        x0 = x
        olm = olm.unsqueeze(0).unsqueeze(0).expand_as(x)
        x = x[olm.bool()].view(x.size(0), x.size(1), -1).contiguous()
        _max_out, _ = torch.max(x, -1, keepdim=False)  # 输入(16,3,64)，输出(16,3)
        _avg_out = torch.mean(x, -1, keepdim=False)  # 输入(16,3,64)，输出(16,3)

        _mlp_max = _max_out
        for layer in self.mlp:
            _mlp_max = layer(_mlp_max)  # 输入(16,3)，输出(16,3)

        _mlp_avg = _avg_out
        for layer in self.mlp:
            _mlp_avg = layer(_mlp_avg)  # 输入(16,3)，输出(16,3)

        _attention = self.out_act(_mlp_avg + _mlp_max)  # 输入(16,3)，输出(16,3)
        _attention = _attention.unsqueeze(-1)
        _attention = _attention.unsqueeze(-1)

        if ret_att:
            return _attention, _attention * x0
        else:
            return _attention * x0


class SABlock(torch.nn.Module):
    def __init__(self):
        super(SABlock, self).__init__()

        self.conv = ConvLayer(in_ch=2,
                              out_ch=1,
                              kernel_size=7,
                              stride=1,
                              padding=3,
                              groups=1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, olm, ret_att=False):
        _max_out, _ = torch.max(x, 1, keepdim=True)  # 若输入(16,64,64,128)，输出(16,1,64,128)
        _avg_out = torch.mean(x, 1, keepdim=True)  # 若输入(16,64,64,128)，输出(16,1,64,128)
        _out = torch.cat((_max_out, _avg_out), dim=1)  # 输出(16,2,64,128)
        _attention = self.conv(_out) * olm
        _attention = self.sigmoid(_attention)

        if ret_att:
            return _attention, _attention * x
        else:
            return _attention * x


class MaskMLABlock(torch.nn.Module):
    def __init__(self,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 padding=0,
                 in_ch=270,
                 num_heads=8,
                 qkv_bias=False,
                 linear_drop_rate=0.5,
                 attn_drop_rate=0.5,
                 mlp_drop_rate=0.5,
                 dim_down=5,
                 cut_rate=4,
                 num_register_tokens=3):
        super(MaskMLABlock, self).__init__()

        # Unfold与Fold的窗口参数
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride
        self.in_ch = in_ch * (self.kernel_size * self.kernel_size)
        self.num_register_tokens = num_register_tokens

        self.register_tokens = (
            torch.nn.Parameter(torch.zeros(1, self.num_register_tokens, self.in_ch)) if num_register_tokens else None)

        self.norm1 = torch.nn.LayerNorm(self.in_ch)
        self.atten = EffAttentionBox(self.in_ch, num_heads=num_heads, qkv_bias=qkv_bias,
                                     linear_drop_rate=linear_drop_rate, attn_drop_rate=attn_drop_rate,
                                     cut_rate=cut_rate, dim_down=dim_down)
        self.norm2 = torch.nn.LayerNorm(self.in_ch)
        self.mlp = MlpBox(in_features=self.in_ch,
                          hidden_features=self.in_ch // 4,
                          act_type='relu',
                          mlp_drop_rate=mlp_drop_rate)

        self.CA = CABlock(in_ch=in_ch)
        self.SA = SABlock()

    def forward(self, x, olm):
        olm0 = olm

        # [CBAM注意力]
        x = self.CA(x, olm0)
        x = self.SA(x, olm0)

        device = x.device
        dtype = x.dtype
        _, _, h, w = x.shape

        # [unfold为序列]
        unfold = torch.nn.Unfold(kernel_size=self.kernel_size,
                                 dilation=self.dilation,
                                 padding=self.padding,
                                 stride=self.stride)
        x = unfold(x)
        x = x.permute(0, 2, 1).contiguous()
        b, n, _ = x.shape

        olm = unfold(olm.unsqueeze(0).unsqueeze(0))
        olm = olm.permute(0, 2, 1).contiguous()[0]

        # [删去纯陆地行+olm编码]]
        land_indices = (olm != 0).all(dim=1)
        x = x[:, land_indices, :].contiguous()

        # [添加寄存器reg]
        if self.num_register_tokens != 0:
            x = torch.cat((x[:, :1].contiguous(),
                           self.register_tokens.expand(x.shape[0], -1, -1), x[:, 1:].contiguous(),), dim=1, )

        # [多头注意力]
        y = self.norm1(x)
        x = x + self.atten(y)

        # [前馈神经网络]
        x = x + self.mlp(self.norm2(x))

        # [舍弃寄存器reg]
        if self.num_register_tokens != 0:
            x = torch.cat((x[:, :1].contiguous(),
                           x[:, 1 + self.num_register_tokens:].contiguous(),), dim=1)

        # [补全纯陆地项]
        result = torch.zeros(b, n, x.shape[2], dtype=dtype).to(device)
        result[:, land_indices, :] = x.contiguous()

        # [删除不适用参量]
        del x, olm

        # [恢复尺寸(不含mask)]
        result = result.permute(0, 2, 1).contiguous()
        fold = torch.nn.Fold(output_size=(h, w),
                             kernel_size=self.kernel_size,
                             dilation=self.dilation,
                             padding=self.padding,
                             stride=self.stride)
        result = fold(result)

        # [CBAM注意力]
        result = self.CA(result, olm0)
        result = self.SA(result, olm0)

        return result


# Part #######################################################
class HPPart(torch.nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 groups,  # groups=((1, 1, (1,1)), 1, 1)  (FM, conv1x1, conv1x1_up)
                 kernel_sizes,
                 incp_num=4,
                 fm_num=4,
                 act_type=None,
                 norm_type=None,
                 use_cc=None,
                 use_cc_ch=None):
        super(HPPart, self).__init__()

        self.fm_num = fm_num
        self.FM = FMBlock(
            in_ch=in_ch,
            out_ch=in_ch,
            groups=groups[0],
            kernel_sizes=kernel_sizes,
            incp_num=incp_num,
            act_type=act_type,
            norm_type=norm_type,
            use_cc=use_cc,
            use_cc_ch=use_cc_ch)

        self.FMs = torch.nn.ModuleList()
        for i in range(fm_num):
            self.FMs.append(FMBlock(in_ch=in_ch,
                                    out_ch=in_ch,
                                    groups=groups[0],
                                    kernel_sizes=kernel_sizes,
                                    incp_num=incp_num,
                                    act_type=act_type,
                                    norm_type=norm_type,
                                    use_cc=use_cc,
                                    use_cc_ch=use_cc_ch))

        self.HF = HFBox()

        self.conv1x1 = ConvLayer(in_ch=2 * in_ch,
                                 out_ch=in_ch,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 groups=groups[1])
        self.conv1x1_up = ConvLayer(in_ch=in_ch,
                                    out_ch=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=groups[2]) if in_ch != out_ch else None

        self.w1 = ScaleUnit(1)
        self.w2 = ScaleUnit(1)

    def forward(self, x):
        x0 = x
        x = self.FM(x)
        x1, x2 = self.HF(x)

        x1 = self.FM(x1)
        for fm in self.FMs:
            x2 = fm(x2)
        x2 = torch.nn.functional.interpolate(x2, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2), dim=1)
        x = self.conv1x1(x)
        x = self.FM(x)
        x = self.w1(x) + self.w2(x0)

        if self.conv1x1_up is not None:
            x = self.conv1x1_up(x)
        return x


class BodyPart(torch.nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 groups,  # groups=((1, 1, (1,1)), 1, 1)  (FM, conv1x1, conv1x1_up) 与HPPart一致
                 kernel_sizes,
                 incp_num,
                 fm_num,
                 act_type,
                 norm_type,
                 use_cc,
                 use_cc_ch,
                 use_mla,
                 mla_args):
        super(BodyPart, self).__init__()

        self.MLA = torch.nn.ModuleList() if "M" in use_mla else None
        if "M" in use_mla:
            All = []
            result = []
            for i in range(len(mla_args[0])):
                for item in mla_args:
                    result.append(item[i])
                All.append(result)
                result = []
            for args in All:
                self.MLA.append(MaskMLABlock(*args))

        self.HP = HPPart(in_ch=in_ch,
                         out_ch=out_ch,
                         groups=groups,
                         kernel_sizes=kernel_sizes,
                         incp_num=incp_num,
                         fm_num=fm_num,
                         act_type=act_type,
                         norm_type=norm_type,
                         use_cc=use_cc,
                         use_cc_ch=use_cc_ch) if "H" in use_mla else None

    def forward(self, x, olm):
        if self.MLA is not None:
            for mla in self.MLA:
                x = mla(x, olm)
        if self.HP is not None:
            x = self.HP(x)
        return x


# Body #######################################################
class CHBody(torch.nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 groups,  # (((1, 1, (1,1)), 1, 1), 1)
                 kernel_sizes,
                 incp_num,
                 fm_num,
                 act_type,
                 norm_type,
                 use_cc,
                 use_cc_ch,
                 use_mla,
                 mla_args):
        super(CHBody, self).__init__()

        # [串联式-CHBody]
        body_parts = []
        for p in range(len(in_ch)):
            body_part = BodyPart(in_ch=in_ch[p][1],
                                 out_ch=out_ch[p][1],
                                 groups=groups[0],
                                 kernel_sizes=kernel_sizes[p],
                                 incp_num=incp_num[p],
                                 fm_num=fm_num[p],
                                 act_type=act_type[p],
                                 norm_type=norm_type[p],
                                 use_cc=use_cc[p],
                                 use_cc_ch=use_cc_ch[p],
                                 use_mla=use_mla[p],
                                 mla_args=mla_args[p])
            body_parts.append(body_part)
        self.Body = torch.nn.ModuleList(body_parts)

        # [conv1x1]
        self.conv1x1 = ConvLayer(in_ch=sum(sub[1] for sub in out_ch) + out_ch[0][1],
                                 out_ch=out_ch[0][1],
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 groups=groups[1])

    def forward(self, x, olm_l, olm_h):
        x_h = torch.nn.functional.interpolate(x,
                                              size=(olm_h.shape[0], olm_h.shape[1]),
                                              mode='bilinear', align_corners=True)
        x_hp_list = [x_h]
        for b in self.Body:
            x = b(x, olm_l)
            x_h = torch.nn.functional.interpolate(x,
                                                  size=(olm_h.shape[0], olm_h.shape[1]),
                                                  mode='bilinear', align_corners=True)
            x_hp_list.append(x_h)
        x = torch.cat(x_hp_list, dim=1)
        x = self.conv1x1(x, olm_h)

        return x


class UBody(torch.nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 groups,  # (((1, 1, (1,1)), 1, 1), 1) 与CHBody一致
                 kernel_sizes,
                 incp_num,
                 fm_num,
                 act_type,
                 norm_type,
                 use_cc,
                 use_cc_ch,
                 use_mla,
                 use_btn_mla,
                 mla_args,
                 btn_mla_args):
        super(UBody, self).__init__()

        self.use_mla = use_mla

        # [U型-UBody]
        self.in_ch = in_ch
        self.Encoder = torch.nn.ModuleList()
        self.BottleNeck = torch.nn.ModuleList()
        self.BottleTail = torch.nn.ModuleList()
        self.Decoder = torch.nn.ModuleList()
        self.ConvBilinear = torch.nn.ModuleList()

        # [编码]
        for p in range(len(in_ch) // 2):
            en = torch.nn.ModuleList()
            en.append(BodyPart(in_ch=in_ch[p][0],
                               out_ch=out_ch[p][0],
                               groups=groups[0],
                               kernel_sizes=kernel_sizes[p],
                               incp_num=incp_num[p],
                               fm_num=fm_num[p],
                               act_type=act_type[p],
                               norm_type=norm_type[p],
                               use_cc=use_cc[p],
                               use_cc_ch=use_cc_ch[p],
                               use_mla=use_mla[p],
                               mla_args=mla_args[p]))
            en.append(ReshapeBox(in_ch=in_ch[p][0], out_ch=in_ch[p][0] * 2))
            self.Encoder.append(en)
        self.n_en = len(in_ch) // 2

        # [瓶颈]
        self.use_btn = use_btn_mla[0]
        if use_btn_mla[0]:
            All = []
            result = []
            for i in range(len(btn_mla_args[0][0])):
                for item in btn_mla_args[0]:
                    result.append(item[i])
                All.append(result)
                result = []
            for args in All:
                self.BottleNeck.append(MaskMLABlock(*args))
            self.n_btn = len(btn_mla_args[0][0])

        # [解码]
        for p in range((len(in_ch) // 2), len(in_ch)):
            de = torch.nn.ModuleList()
            de.append(ReshapeBox(in_ch=in_ch[p][0], out_ch=out_ch[p][0]))
            de.append(BodyPart(in_ch=in_ch[p][0],
                               out_ch=out_ch[p][0],
                               groups=groups[0],
                               kernel_sizes=kernel_sizes[p],
                               incp_num=incp_num[p],
                               fm_num=fm_num[p],
                               act_type=act_type[p],
                               norm_type=norm_type[p],
                               use_cc=use_cc[p],
                               use_cc_ch=use_cc_ch[p],
                               use_mla=use_mla[p],
                               mla_args=mla_args[p]))
            if "_M_" in use_mla:
                # 通道变换
                de.append(ConvLayer(in_ch=in_ch[p][0],
                                    out_ch=out_ch[p][0],
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=groups[1]))
                de.append(torch.nn.BatchNorm2d(num_features=out_ch[p][0]))
                de.append(ActLayer(act_type="prelu", n_prelu=out_ch[p][0]))
            self.Decoder.append(de)

        self.conv_up = ConvLayer(in_ch=out_ch[-1][0],
                                 out_ch=out_ch[-1][0] * (3 ** 2),
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 groups=groups[1])
        self.bn_up = torch.nn.BatchNorm2d(num_features=out_ch[-1][0] * (3 ** 2))
        self.act_up = ActLayer(act_type="prelu", n_prelu=out_ch[-1][0] * (3 ** 2))
        self.pixel_shuffle = torch.nn.PixelShuffle(3)
        self.n_de = len(in_ch) // 2

        # [堆叠]
        for p in range((len(in_ch) // 2), len(in_ch)):
            cb = torch.nn.ModuleList()
            cb.append(ConvLayer(in_ch=in_ch[p][0],
                                out_ch=out_ch[-1][0],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                groups=groups[1]))
            cb.append(torch.nn.BatchNorm2d(num_features=out_ch[-1][0]))
            cb.append(ActLayer(act_type="prelu", n_prelu=out_ch[-1][0]))
            self.ConvBilinear.append(cb)
        cb = torch.nn.ModuleList()
        cb.append(ConvLayer(in_ch=out_ch[-1][0],
                            out_ch=out_ch[-1][0],
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            groups=groups[1]))
        cb.append(torch.nn.BatchNorm2d(num_features=out_ch[-1][0]))
        cb.append(ActLayer(act_type="prelu", n_prelu=out_ch[-1][0]))
        self.ConvBilinear.append(cb)
        self.n_cb = len(in_ch) // 2 + 1

        # [融合]
        self.conv_fuse = ConvLayer(in_ch=out_ch[-1][0] * (self.n_cb + 1),
                                   out_ch=out_ch[-1][0],
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   groups=groups[1])
        self.bn_fuse = torch.nn.BatchNorm2d(num_features=out_ch[-1][0])
        self.act_fuse = ActLayer(act_type="prelu", n_prelu=out_ch[-1][0])

        # [瓶尾]
        self.use_btt = use_btn_mla[1]
        if use_btn_mla[1]:
            All = []
            result = []
            for i in range(len(btn_mla_args[1][0])):
                for item in btn_mla_args[1]:
                    result.append(item[i])
                All.append(result)
                result = []
            for args in All:
                self.BottleTail.append(MaskMLABlock(*args))
            self.n_btt = len(btn_mla_args[1][0])

        # [残差加权]
        self.w1 = ScaleUnit(1)
        self.w2 = ScaleUnit(1)

    def forward(self, x, olm_l, olm_h):
        x0 = torch.nn.functional.interpolate(x,
                                             size=(olm_h.shape[0], olm_h.shape[1]),
                                             mode='bilinear', align_corners=True)

        En = []
        De = []
        Olm = []

        # [编码]
        for i in range(self.n_en):
            x = self.Encoder[i][0](x, olm_l)
            En.append(x)
            Olm.append(olm_l)
            x = self.Encoder[i][1](x)
            olm_l = (torch.nn.functional.max_pool2d(olm_l.unsqueeze(0).unsqueeze(0), kernel_size=2, stride=2, padding=1)
                     .squeeze(0).squeeze(0))
        Olm.append(olm_l)

        # [瓶颈]
        if self.use_btn:
            for i in range(self.n_btn):
                x = self.BottleNeck[i](x, Olm[-1])

        # [解码]
        for i in range(self.n_de):
            De.append(x)
            x = self.Decoder[i][0](x)
            x = torch.nn.functional.interpolate(x, size=(En[-1 - i].shape[2], En[-1 - i].shape[3]),
                                                mode='bilinear', align_corners=True)
            x = torch.cat([x, En[-1 - i]], dim=1)
            x = self.Decoder[i][1](x, Olm[-2 - i])
            if "_M_" in self.use_mla:
                x = self.Decoder[i][2](x)
                x = self.Decoder[i][3](x)
                x = self.Decoder[i][4](x)
        De.append(x)
        x_up = self.conv_up(x)
        x_up = self.bn_up(x_up)
        x_up = self.act_up(x_up)
        x_up = self.pixel_shuffle(x_up)
        x_up = torch.nn.functional.interpolate(x_up,
                                               size=(olm_h.shape[0], olm_h.shape[1]),
                                               mode='bilinear', align_corners=True)

        # [堆叠]
        for i in range(self.n_cb):
            x = self.ConvBilinear[i][0](De[i])
            x = self.ConvBilinear[i][1](x)
            x = self.ConvBilinear[i][2](x)
            De[i] = torch.nn.functional.interpolate(x,
                                                    size=(olm_h.shape[0], olm_h.shape[1]),
                                                    mode='bilinear', align_corners=True)
        De.append(x_up)
        x = torch.cat(De, dim=1)

        # [融合]
        x = self.conv_fuse(x)
        x = self.bn_fuse(x)
        x = self.act_fuse(x)

        # [瓶尾]
        if self.use_btt:
            for i in range(self.n_btt):
                x = self.BottleTail[i](x, olm_h)

        return self.w1(x0) + self.w2(x)


# Model ##############################################################################################
class S4R2(torch.nn.Module):
    def __init__(self, kwargs):
        super(S4R2, self).__init__()

        # 使用get方法获取字典中的值，如果键不存在则使用默认值
        self.train_param = kwargs.get('train_param',
                                      {"gyh_type": "Norm"})
        self.in_ch = kwargs.get('in_ch',
                                {"CNNStackBlock": [(7,), (1,)],
                                 "FMBlock": [(24,), (3,)],
                                 "HPPart": [(24, 24),
                                            (48, 24),
                                            (48, 24),
                                            (24, 24), ]})
        self.out_ch = kwargs.get('out_ch',
                                 {"CNNStackBlock": [(21,), (3,)],
                                  "FMBlock": [(24,), (3,)],
                                  "HPPart": [(24, 24),
                                             (48, 24),
                                             (48, 24),
                                             (24, 24), ]})
        self.groups = kwargs.get("groups",  # Inception含两个groups一个是1x1部分的, 一个是后续的多尺度卷积部分的
                                 {"CNNStackBlock": [(1, (1, 1)), (1, (1, 1))],  # (use_cc_ch的conv1x1, Inception)
                                  "FMBlock": [(1, 1, (1, 1)), (1, 1, (1, 1))],
                                  # (use_cc_ch的conv1x1, conv1x1_up变化衔接处通道数, Inception)
                                  "HPPart": [((1, 1, (1, 1)), 1, 1)]})  # (FM/FMS, conv1x1合并, conv1x1_up变化衔接处通道数)
        self.kernel_sizes = kwargs.get("kernel_sizes",
                                       {"CNNStackBlock": [[1, 3, 5, 3], [1, 3, 5, 3]],
                                        "FMBlock": [[1, 3, 5, 3], [1, 3, 5, 3]],
                                        "HPPart": [[1, 3, 5, 3] for _ in range(4)]})
        self.incp_num = kwargs.get('incp_num',
                                   {"CNNStackBlock": [3, 3],
                                    "FMBlock": [1, 1],
                                    "HPPart": [1, 1, 1, 1]})
        self.upper_num = kwargs.get('upper_num',
                                    {"CNNStackBlock": [7, 1],
                                     "FMBlock": None,
                                     "HPPart": [4, 4, 4, 4]})
        self.act_type = kwargs.get('act_type',
                                   {"CNNStackBlock": [("prelu", "prelu"), ("prelu", "prelu")],
                                    "FMBlock": ["prelu", "prelu"],
                                    "HPPart": ["prelu", "prelu", "prelu", "prelu"]})
        self.norm_type = kwargs.get('norm_type',
                                    {"CNNStackBlock": [("batch", "batch"), ("batch", "batch")],
                                     "FMBlock": ["batch", "batch"],
                                     "HPPart": ["batch", "batch",
                                                "batch", "batch"]})
        self.use_cc = kwargs.get('use_cc',
                                 {"CNNStackBlock": [[True, None], [True, None]],
                                  "FMBlock": [[True, None], [True, None]],
                                  "HPPart": [[True, None], [True, None],
                                             [True, None], [True, None]]})
        self.use_cc_ch = kwargs.get('use_cc_ch',
                                    {"CNNStackBlock": [[True, None, None], [True, None, None]],
                                     "FMBlock": [[True, None, None], [True, None, None]],
                                     "HPPart": [[True, None, None],
                                                [True, None, None],
                                                [True, None, None],
                                                [True, None, None]]})
        self.u_mlablock = kwargs.get('u_mlablock',
                                     {"use_mla": (True, True, True, True, True, True),
                                      "kernel_size": [(2,), (2,), (2,), (2,)],
                                      "stride": [(2,), (2,), (2,), (2,)],
                                      "dilation": [(1,), (1,), (1,), (1,)],
                                      "padding": [(0,), (0,), (0,), (0,)],
                                      "in_ch": [(24,),
                                                (48,),
                                                (48,),
                                                (24,)],
                                      "num_heads": [(8,), (8,),
                                                    (8,), (8,)],
                                      "qkv_bias": [(True,), (True,),
                                                   (True,), (True,)],
                                      "linear_drop_rate": [(0.5,), (0.5,), (0.5,), (0.5,)],
                                      "attn_drop_rate": [(0.5,), (0.5,), (0.5,), (0.5,)],
                                      "mlp_drop_rate": [(0.5,), (0.5,), (0.5,), (0.5,)],
                                      "dim_down": [(3,), (3,), (3,), (3,)],
                                      "cut_rate": [(4,), (4,), (4,), (4,)],
                                      "num_register_tokens": [(6,), (6,), (6,), (6,)]})
        self.ch_mlablock = kwargs.get('ch_mlablock',
                                      {"use_mla": (True, True, True, True, True, True),
                                       "kernel_size": [(2,), (2,), (2,), (2,)],
                                       "stride": [(2,), (2,), (2,), (2,)],
                                       "dilation": [(1,), (1,), (1,), (1,)],
                                       "padding": [(0,), (0,), (0,), (0,)],
                                       "in_ch": [(24,),
                                                 (24,),
                                                 (24,),
                                                 (24,)],
                                       "num_heads": [(8,), (8,),
                                                     (8,), (8,)],
                                       "qkv_bias": [(True,), (True,),
                                                    (True,), (True,)],
                                       "linear_drop_rate": [(0.5,), (0.5,), (0.5,), (0.5,)],
                                       "attn_drop_rate": [(0.5,), (0.5,), (0.5,), (0.5,)],
                                       "mlp_drop_rate": [(0.5,), (0.5,), (0.5,), (0.5,)],
                                       "dim_down": [(3,), (3,), (3,), (3,)],
                                       "cut_rate": [(4,), (4,), (4,), (4,)],
                                       "num_register_tokens": [(6,), (6,), (6,), (6,)]})
        self.btn_mlablock = kwargs.get("btn_mlablock",
                                       {"use_btn_mla": (True, False),
                                        "kernel_size": [(2,), (2,)],
                                        "stride": [(2,), (2,)],
                                        "dilation": [(1,), (1,)],
                                        "padding": [(0,), (0,)],
                                        "in_ch": [(96,),
                                                  (24,)],
                                        "num_heads": [(8,), (8,)],
                                        "qkv_bias": [(True,), (True,)],
                                        "linear_drop_rate": [(0.5,), (0.5,)],
                                        "attn_drop_rate": [(0.5,), (0.5,)],
                                        "mlp_drop_rate": [(0.5,), (0.5,)],
                                        "dim_down": [(3,), (3,)],
                                        "cut_rate": [(4,), (16,)],
                                        "num_register_tokens": [(6,), (6,)]})
        self.body = kwargs.get('body',
                               {'shape': "U", 'groups': [1, 1]})
        self.tail = kwargs.get('tail',
                               {"act_type": "prelu", "groups": [1, 1]})

        # [Head]
        self.CNN_l = CNNStackBlock(in_ch=self.in_ch["CNNStackBlock"][0][0],
                                   out_ch=self.out_ch["CNNStackBlock"][0][0],
                                   groups=self.groups["CNNStackBlock"][0],
                                   kernel_sizes=self.kernel_sizes["CNNStackBlock"][0],
                                   incp_num=self.incp_num["CNNStackBlock"][0],
                                   ele_num=self.upper_num["CNNStackBlock"][0],
                                   act_type=self.act_type["CNNStackBlock"][0],
                                   norm_type=self.norm_type["CNNStackBlock"][0],
                                   use_cc=self.use_cc["CNNStackBlock"][0],
                                   use_cc_ch=self.use_cc_ch["CNNStackBlock"][0])

        self.CNN_h = CNNStackBlock(in_ch=self.in_ch["CNNStackBlock"][1][0],
                                   out_ch=self.out_ch["CNNStackBlock"][1][0],
                                   groups=self.groups["CNNStackBlock"][1],
                                   kernel_sizes=self.kernel_sizes["CNNStackBlock"][1],
                                   incp_num=self.incp_num["CNNStackBlock"][1],
                                   ele_num=self.upper_num["CNNStackBlock"][1],
                                   act_type=self.act_type["CNNStackBlock"][1],
                                   norm_type=self.norm_type["CNNStackBlock"][1],
                                   use_cc=self.use_cc["CNNStackBlock"][1],
                                   use_cc_ch=self.use_cc_ch["CNNStackBlock"][1]) if self.in_ch["CNNStackBlock"][1][
                                                                                        0] != 0 else None

        self.FM_l = FMBlock(
            in_ch=self.in_ch["FMBlock"][0][0],
            out_ch=self.out_ch["FMBlock"][0][0],
            groups=self.groups["FMBlock"][0],
            kernel_sizes=self.kernel_sizes["FMBlock"][0],
            incp_num=self.incp_num["FMBlock"][0],
            act_type=self.act_type["FMBlock"][0],
            norm_type=self.norm_type["FMBlock"][0],
            use_cc=self.use_cc["FMBlock"][0],
            use_cc_ch=self.use_cc_ch["FMBlock"][0])

        self.FM_h = FMBlock(
            in_ch=self.in_ch["FMBlock"][1][0],
            out_ch=self.out_ch["FMBlock"][1][0],
            groups=self.groups["FMBlock"][1],
            kernel_sizes=self.kernel_sizes["FMBlock"][1],
            incp_num=self.incp_num["FMBlock"][1],
            act_type=self.act_type["FMBlock"][1],
            norm_type=self.norm_type["FMBlock"][1],
            use_cc=self.use_cc["FMBlock"][1],
            use_cc_ch=self.use_cc_ch["FMBlock"][1]) if self.in_ch["FMBlock"][1][0] != 0 else None

        # [Body]
        self.CHBody = CHBody(in_ch=self.in_ch["HPPart"],
                             out_ch=self.out_ch["HPPart"],
                             groups=(self.groups["HPPart"][0], self.body["groups"][1]),
                             kernel_sizes=self.kernel_sizes["HPPart"],
                             incp_num=self.incp_num["HPPart"],
                             fm_num=self.upper_num["HPPart"],
                             act_type=self.act_type["HPPart"],
                             norm_type=self.norm_type["HPPart"],
                             use_cc=self.use_cc["HPPart"],
                             use_cc_ch=self.use_cc_ch["HPPart"],
                             use_mla=self.ch_mlablock["use_mla"],
                             mla_args=list(zip(self.ch_mlablock["kernel_size"],
                                               self.ch_mlablock["stride"],
                                               self.ch_mlablock["dilation"],
                                               self.ch_mlablock["padding"],
                                               self.ch_mlablock["in_ch"],
                                               self.ch_mlablock["num_heads"],
                                               self.ch_mlablock["qkv_bias"],
                                               self.ch_mlablock["linear_drop_rate"],
                                               self.ch_mlablock["attn_drop_rate"],
                                               self.ch_mlablock["mlp_drop_rate"],
                                               self.ch_mlablock["dim_down"],
                                               self.ch_mlablock["cut_rate"],
                                               self.ch_mlablock["num_register_tokens"]
                                               ))) if self.body["shape"] == "CH" else None

        self.UBody = UBody(in_ch=self.in_ch["HPPart"],
                           out_ch=self.out_ch["HPPart"],
                           groups=(self.groups["HPPart"][0], self.body["groups"][1]),
                           kernel_sizes=self.kernel_sizes["HPPart"],
                           incp_num=self.incp_num["HPPart"],
                           fm_num=self.upper_num["HPPart"],
                           act_type=self.act_type["HPPart"],
                           norm_type=self.norm_type["HPPart"],
                           use_cc=self.use_cc["HPPart"],
                           use_cc_ch=self.use_cc_ch["HPPart"],
                           use_mla=self.u_mlablock["use_mla"],
                           mla_args=list(zip(self.u_mlablock["kernel_size"],
                                             self.u_mlablock["stride"],
                                             self.u_mlablock["dilation"],
                                             self.u_mlablock["padding"],
                                             self.u_mlablock["in_ch"],
                                             self.u_mlablock["num_heads"],
                                             self.u_mlablock["qkv_bias"],
                                             self.u_mlablock["linear_drop_rate"],
                                             self.u_mlablock["attn_drop_rate"],
                                             self.u_mlablock["mlp_drop_rate"],
                                             self.u_mlablock["dim_down"],
                                             self.u_mlablock["cut_rate"],
                                             self.u_mlablock["num_register_tokens"])),
                           use_btn_mla=self.btn_mlablock["use_btn_mla"],
                           btn_mla_args=list(zip(self.btn_mlablock["kernel_size"],
                                                 self.btn_mlablock["stride"],
                                                 self.btn_mlablock["dilation"],
                                                 self.btn_mlablock["padding"],
                                                 self.btn_mlablock["in_ch"],
                                                 self.btn_mlablock["num_heads"],
                                                 self.btn_mlablock["qkv_bias"],
                                                 self.btn_mlablock["linear_drop_rate"],
                                                 self.btn_mlablock["attn_drop_rate"],
                                                 self.btn_mlablock["mlp_drop_rate"],
                                                 self.btn_mlablock["dim_down"],
                                                 self.btn_mlablock["cut_rate"],
                                                 self.btn_mlablock["num_register_tokens"]
                                                 ))) if self.body["shape"] == "U" else None

        # [Tail]
        self.conv_down_ch_cc = ConvLayer(in_ch=self.out_ch["HPPart"][-1][0],
                                         out_ch=1,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1,
                                         groups=self.tail["groups"][0])

        self.conv_up_og = ConvLayer(in_ch=1,
                                    out_ch=3 * 3 ** 2,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=self.tail["groups"][0])
        self.bn_up_og = torch.nn.BatchNorm2d(num_features=3 * 3 ** 2)
        self.act_up_og = ActLayer(act_type="prelu", n_prelu=3 * 3 ** 2)
        self.pixel_shuffle = torch.nn.PixelShuffle(3)
        self.conv_down_ch_og = ConvLayer(in_ch=3,
                                         out_ch=1,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1,
                                         groups=self.tail["groups"][0])

        self.w1 = ScaleUnit(1)
        self.w2 = ScaleUnit(1)

        self.CA = CABlock(in_ch=self.in_ch["HPPart"][0][0])
        self.SA = SABlock()

    def forward(self, x, OLM):
        # [origin_lr_sss]
        x_og = x[0]

        # [Mask]
        olm_l = OLM[0][0, 0]
        olm_h = OLM[1][0, 0]

        # [分辨率分组]
        x_l = []
        x_h = []

        # [加噪noise补全]
        i = 0
        for tensor in x:
            if tensor.shape[2] == olm_l.shape[0]:  # 假设 size2 是第二类尺寸
                x_l.append(tensor)
            elif tensor.shape[2] == olm_h.shape[0]:  # 假设 size1 是第一类尺寸
                x_h.append(tensor)
            i += 1

        # [Head]
        x_l = self.CNN_l(x_l)  # 此时还是列表

        x_h_l = None
        if self.CNN_h is not None:
            x_h = self.CNN_h(x_h)  # 此时还是列表
            x_h_l = [torch.nn.functional.interpolate(x_h_one, size=(x_l[0].shape[2], x_l[0].shape[3]),
                                                     mode='bilinear', align_corners=True) for x_h_one in x_h]
            x_h_l = torch.cat(x_h_l, dim=1)
            x_h_l = self.FM_h(x_h_l)

        x_l = torch.cat(x_l, dim=1)
        if self.CNN_h is not None:
            x_l = torch.cat([x_l, x_h_l], dim=1)

        x_l = self.FM_l(x_l)
        x_l = self.CA(x_l, olm_l)
        x_l = self.SA(x_l, olm_l)

        # [Body]
        x_body = None
        if self.body["shape"] == "U":
            x_body = self.UBody(x_l, olm_l, olm_h)
        if self.body["shape"] == "CH":
            x_body = self.CHBody(x_l, olm_l, olm_h)
        if self.body["shape"] == "None":
            x_body = torch.nn.functional.interpolate(x_l,
                                                     size=(olm_h.shape[0], olm_h.shape[1]),
                                                     mode='bilinear', align_corners=True)

        # [Tail]
        x_body = self.conv_down_ch_cc(x_body)

        x_og = self.conv_up_og(x_og)
        x_og = self.bn_up_og(x_og)
        x_og = self.act_up_og(x_og)
        x_og = self.pixel_shuffle(x_og)
        x_og = torch.nn.functional.interpolate(x_og,
                                               size=(olm_h.shape[0], olm_h.shape[1]),
                                               mode='bilinear', align_corners=True)
        x_og = self.conv_down_ch_og(x_og)

        return self.w1(x_og) + self.w2(x_body)
