from typing import Any, List, Tuple
from torch import Tensor
import models.basicblock as B

from math import ceil
from .utils import *
import models.dct_head_d as dct_head_d
from scipy.interpolate import griddata
import time
from torch import autograd
import time
import torch
import torch.nn as nn
from torch import autograd
from thop import profile


class LayerNorm(nn.Module):

    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = nn.LayerNorm(dim, elementwise_affine=False)
        else:
            self.body = nn.LayerNorm(dim, elementwise_affine=True)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)  # [B, C, H*W] -> [B, H*W, C]
        x = self.body(x)
        x = x.transpose(1, 2).view(-1, x.size(-1), h, w)   
        return x


class LightweightMDTA(nn.Module):

    def __init__(self, dim, num_heads=2, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias, groups=3)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                    stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(b, self.num_heads, -1, h * w)
        k = k.view(b, self.num_heads, -1, h * w)
        v = v.view(b, self.num_heads, -1, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).reshape(b, c, h, w)
        return self.project_out(out)


class EfficientGDFN(nn.Module):

    def __init__(self, dim, ffn_expansion_factor=1.5, bias=False):   
        super().__init__()
        hidden_dim = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_dim * 2, 1, bias=bias)
         
        self.dwconv = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 3,
                                padding=1, groups=hidden_dim * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_dim, dim, 1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)

class RestormerBlock(nn.Module):

    def __init__(self, dim, num_heads=2, ffn_expansion_factor=1.5):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = LightweightMDTA(dim, num_heads)
        self.norm2 = LayerNorm(dim)
        self.ffn = EfficientGDFN(dim, ffn_expansion_factor)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class NetD(nn.Module):

    def __init__(self, atom_channels, d_size, num_heads=2, num_blocks=2, ffn_expansion_factor=1.5):
        super().__init__()

        self.d_size = d_size
        self.atom_channels = atom_channels

        self.init_conv = nn.Conv2d(atom_channels + 1, atom_channels, kernel_size=1, padding=0)

        self.refine_blocks = nn.Sequential(
            *[RestormerBlock(atom_channels, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor) for _ in
              range(num_blocks)]
        )

        self.final_conv = nn.Conv2d(atom_channels, atom_channels, kernel_size=1, padding=0)
        self.activation = nn.GELU()

    def forward(self, d_flat):
        x = self.init_conv( d_flat)
        x = self.activation(x)
        x = self.refine_blocks(x)
        x = self.final_conv(x)
        out = x + d_flat[:,:-1,:,:]
        return out


class HeadNet(nn.Module):
    def __init__(self, in_nc: int, nc_x: List[int], out_nc: int, d_size: int):
        super(HeadNet, self).__init__()
        self.head_x = nn.Sequential(
            nn.Conv2d(in_nc,
                      nc_x[0],
                      d_size,
                      padding=(d_size - 1) // 2,
                      bias=False), nn.ReLU(inplace=True),
            B.ResBlock(nc_x[0], nc_x[0], bias=False, mode='CRC'),
            B.ResBlock(nc_x[0], nc_x[0], bias=False, mode='CRC'),
            nn.Conv2d(nc_x[0], nc_x[0], 3, padding=1, bias=False))
        self.d_size = d_size
        self.nc_x = nc_x
        self.out_nc=out_nc

    def forward(self, y: Any) -> Tuple[Tensor, Tensor]:
        x = self.head_x(y)
        d = dct_head_d.dct_dict_initialization(y, self.nc_x[0], self.d_size, 'True',self.out_nc)
        return x, d


class BodyNet(nn.Module):
    def __init__(self, in_nc: int, nc_x: List[int], nc_d: List[int],
                 out_nc: int, nb: int,d_size:int):
        super(BodyNet, self).__init__()

        self.net_x = NetX(in_nc=in_nc, nc_x=nc_x, nb=nb)
        self.solve_fft = SolveFFT()

        self.net_d = NetD(nc_x[0]*out_nc, d_size, num_heads=2, num_blocks=2, ffn_expansion_factor=1.5)
        self.solve_ls = SolveLS()
        # self.solve_ls = MonitoredSolveLS()
        self.tail=TailNet()
        self.module_time = {}

    def forward(self, x: Tensor, d: Tensor, y: Tensor, Y: Tensor,
                alpha_x: Tensor, beta_x: Tensor, alpha_d: float, beta_d: float,
                reg: float):
        """
            x: N, C_in, H, W
            d: N, C_out, C_in, d_size, d_size
            Y: N, C_out, 1, H, W, 2
            y: N, C_out, H, W
            alpha/beta: 1, 1, 1, 1
            reg: float
        """
        # Solve X
        X, D = self.rfft_xd(x, d)
        size_x = np.array(list(x.shape[-2:]))
        x = self.solve_fft(X, D, Y, alpha_x, size_x)
        beta_x = (1 / beta_x.sqrt()).repeat(1, 1, x.size(2), x.size(3))
        in_netx = torch.cat([x, beta_x], dim=1)
  
        x = self.net_x(torch.cat([x, beta_x], dim=1))

        if self.net_d is not None:
            d = self.solve_ls(x.unsqueeze(1), d, y.unsqueeze(2), alpha_d, reg)
            beda_d = (1 / beta_d.sqrt()).repeat(1, 1, d.size(3), d.size(4))
            size_d = [d.size(1), d.size(2)]
            d = d.view(d.size(0), d.size(1) * d.size(2), d.size(3), d.size(4))
            in_netd = torch.cat([d, beda_d], dim=1)
            d = self.net_d(torch.cat([d, beda_d], dim=1))
            d = d.view(d.size(0), size_d[0], size_d[1], d.size(2), d.size(3))
        return x, d
    def rfft_xd(self, x: Tensor, d: Tensor):
        X =compatible_rfft(x, signal_ndim=2).unsqueeze(1)
        D = p2o(d, x.shape[-2:])
        return X, D

class NetX(nn.Module):
    def __init__(self,
                 in_nc: int = 65,
                 nc_x: List[int] = [64, 128, 256, 512],
                 nb: int = 4):
        super(NetX, self).__init__()

        self.m_down1 = B.sequential(
            *[
                B.ResBlock(in_nc, in_nc, bias=False, mode='CRC')
                for _ in range(nb)
            ], B.downsample_strideconv(in_nc, nc_x[1], bias=False, mode='2'))
        self.m_down2 = B.sequential(
            *[
                B.ResBlock(nc_x[1], nc_x[1], bias=False, mode='CRC')
                for _ in range(nb)
            ], B.downsample_strideconv(nc_x[1], nc_x[2], bias=False, mode='2'))
        self.m_down3 = B.sequential(
            *[
                B.ResBlock(nc_x[2], nc_x[2], bias=False, mode='CRC')
                for _ in range(nb)
            ], B.downsample_strideconv(nc_x[2], nc_x[3], bias=False, mode='2'))

        self.m_body = B.sequential(*[
            B.ResBlock(nc_x[-1], nc_x[-1], bias=False, mode='CRC')
            for _ in range(nb)
        ])

        self.m_up3 = B.sequential(
            B.upsample_convtranspose(nc_x[3], nc_x[2], bias=False, mode='2'),
            *[
                B.ResBlock(nc_x[2], nc_x[2], bias=False, mode='CRC')
                for _ in range(nb)
            ])
        self.m_up2 = B.sequential(
            B.upsample_convtranspose(nc_x[2], nc_x[1], bias=False, mode='2'),
            *[
                B.ResBlock(nc_x[1], nc_x[1], bias=False, mode='CRC')
                for _ in range(nb)
            ])
        self.m_up1 = B.sequential(
            B.upsample_convtranspose(nc_x[1], nc_x[0], bias=False, mode='2'),
            *[
                B.ResBlock(nc_x[0], nc_x[0], bias=False, mode='CRC')
                for _ in range(nb)
            ])

        self.m_tail = B.conv(nc_x[0], nc_x[0], bias=False, mode='C')

    def forward(self, x):
        x1 = x
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        # x = self.att_body(x)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1[:, :-1, :, :])
        return x

class SolveFFT(nn.Module):
    def __init__(self):
        super(SolveFFT, self).__init__()

    def forward(self, X: Tensor, D: Tensor, Y: Tensor, alpha: Tensor,
                x_size: np.ndarray):
        """
            X: N, 1, C_in, H, W, 2
            D: N, C_out, C_in, H, W, 2
            Y: N, C_out, 1, H, W, 2
            alpha: N, 1, 1, 1
        """
        alpha = alpha.unsqueeze(-1).unsqueeze(-1) / X.size(2)

        _D = cconj(D)
        m= alpha * X
        Z = cmul(Y, D) + alpha * X

        factor1 = Z / alpha

        numerator = cmul(_D, Z).sum(2, keepdim=True)
        denominator = csum(alpha * cmul(_D, D).sum(2, keepdim=True),
                           alpha.squeeze(-1)**2)
        factor2 = cmul(D, cdiv(numerator, denominator))
        X = (factor1 - factor2).mean(1)
        x_recon = compatible_irfft(
            X,
            signal_ndim=2,
            signal_sizes=list(x_size)
        )

        return x_recon
class CholeskySolve(autograd.Function):
    @staticmethod
    def forward(ctx, Q, P):

        L = torch.linalg.cholesky(Q)
        D = torch.cholesky_solve(P, L)  # D = Q-1 @ P
        ctx.save_for_backward(L, D)
        return D

    @staticmethod
    def backward(ctx, dLdD):
        L, D = ctx.saved_tensors
        dLdP = torch.cholesky_solve(dLdD, L)
        dLdQ = -dLdP.matmul(D.transpose(-2, -1))

        return dLdQ, dLdP

class SolveLS(nn.Module):
    def __init__(self):
        super(SolveLS, self).__init__()
        self.cholesky_solve = CholeskySolve.apply
        self.register_buffer("xtx_index", None)

    def _get_xtx_index(self, C_in, d_size, device):
        size = (C_in * d_size) ** 2
        idx = self.xtx_index
        if idx is None or idx.numel() != size:
            idx = torch.arange(size, device=device).view(
                C_in, C_in, d_size, d_size
            ).permute(0, 2, 3, 1).reshape(-1)
            self.xtx_index = idx
        return idx

    def forward(self, x, d, y, alpha, reg):
        """
            x: N, 1, C_in, H, W
            d: N, C_out, C_in, d_size, d_size
            y: N, C_out, 1, H, W
            alpha: N, 1, 1, 1
            reg: float
        """
        x=x.float()
        C_in = x.shape[2]
        d_size = d.shape[-1]
        N = x.shape[0]
        xtx_raw = self.cal_xtx(x, d_size)  # N, C_in, C_in, d_size, d_size
        xtx_unfold = F.unfold(
            xtx_raw.view(
                xtx_raw.size(0) * xtx_raw.size(1), xtx_raw.size(2),
                xtx_raw.size(3), xtx_raw.size(4)), d_size)
        xtx_unfold = xtx_unfold.view(xtx_raw.size(0), xtx_raw.size(1),
                                     xtx_unfold.size(1), xtx_unfold.size(2))

        xtx = xtx_unfold.view(xtx_unfold.size(0), xtx_unfold.size(1),
                              xtx_unfold.size(1), -1, xtx_unfold.size(3))
        xtx.copy_(xtx[:, :, :, torch.arange(xtx.size(3) - 1, -1, -1), ...])
        xtx = xtx.view(xtx.size(0), -1, xtx.size(-1))  # TODO

        index = self._get_xtx_index(C_in, d_size, xtx.device)
        xtx = xtx[:, index, :]
        xtx = xtx.view(xtx.size(0), d_size**2 * C_in, -1)
        xtx = (xtx + xtx.transpose(1, 2)) *0.5

        xty = self.cal_xty(x, y, d_size)

        xty = xty.reshape(xty.size(0), xty.size(1), -1).permute(0, 2, 1)

        # reg
        alpha = alpha * x.size(3) * x.size(4) * reg / (d_size**2 * d.size(2))


        alpha_squeezed = alpha.squeeze(-1).squeeze(-1)
        diagonal_update = (xtx[:, range(len(xtx[0])), range(len(xtx[0]))] + alpha_squeezed).to(xtx.dtype)

        xtx[:, range(len(xtx[0])), range(len(xtx[0]))] = diagonal_update

        d_reshaped = d.reshape(d.size(0), d.size(1), -1).permute(0, 2, 1)
        xty_update = (alpha_squeezed.unsqueeze(1) * d_reshaped).to(xty.dtype)
        xty += xty_update
        xtx=xtx.float()
        xty = xty.float()

        # solve
        try:
            d = self.cholesky_solve(xtx, xty).view(d.size(0), C_in, d_size,
                                                   d_size, d.size(1)).permute(
                                                       0, 4, 1, 2, 3)

        except RuntimeError as e:
            print(f"[ERROR] Cholesky failed：{e}")
            d = torch.linalg.solve(xtx, xty).view(d.size(0), C_in, d_size, d_size, d.size(1)).permute(0, 4, 1, 2, 3)
        return d

    def cal_xtx(self, x, d_size):
        padding = d_size - 1
        xtx = conv3d(x,
                     x.view(x.size(0), x.size(2), 1, 1, x.size(3), x.size(4)),
                     padding,
                     sample_wise=True)
        return xtx

    def cal_xty(self, x, y, d_size):
        padding = (d_size - 1) // 2
        xty = conv3d(x, y.unsqueeze(3).contiguous(), padding, sample_wise=True)
        return xty

class TailNet(nn.Module):
    def __init__(self):
        super(TailNet, self).__init__()

    def forward(self, x, d):
        y = conv2d(F.pad(x, [
            (d.size(-1) - 1) // 2,
        ] * 4, mode='circular'),
                   d,
                   sample_wise=True)

        return y

class HyPaNet(nn.Module):
    def __init__(self, in_nc=3, nf=32, num_stages=1):
        super(HyPaNet, self).__init__()
        self.num_stages = num_stages

        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d(16),
            nn.Conv2d(in_nc, nf, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.param_predictor = nn.Sequential(
            nn.Linear(nf, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_stages * 4),   
            nn.Softplus()
        )

    def forward(self, x):
        """
        Args:
            x:[B, C, H, W]
        Returns:
            params: [B, num_stages, 4, 1, 1]
        """
        batch_size = x.size(0)

        feat = self.feature_extractor(x)
        feat = feat.view(batch_size, -1)

        params = self.param_predictor(feat)
        params = params.view(batch_size, self.num_stages, 4, 1, 1)

        return params + 1e-6

class DULR(nn.Module):
    def __init__(self,
                 n_iter: int = 1,
                 in_nc: int = 1,
                 nc_x: List[int] = [64, 128, 256, 512],
                 out_nc: int = 12,
                 nb: int = 1,
                 d_size: int = 5,
                 **kargs):
        super(DULR, self).__init__()

        self.head_p = HeadNet(in_nc=12, nc_x=nc_x, out_nc=out_nc, d_size=d_size)
        self.pdm = BodyNet(in_nc=nc_x[0] + 1,
                            nc_x=nc_x,
                            nc_d=nc_x,
                            out_nc=out_nc,
                            nb=nb,d_size=d_size)
        self.tail = TailNet()
        self.hypa_p = HyPaNet(in_nc=12, nf=32, num_stages=n_iter)
        self.n_iter = n_iter

    def forward(self, y):
        h, w = y.size()[-2:]
        paddingBottom = int(ceil(h / 8) * 8 - h)
        paddingRight = int(ceil(w / 8) * 8 - w)

        pred = None
        preds = []

        y_s1 = F.pad(y, [0, paddingRight, 0, paddingBottom], mode='circular')
        Y_s1 = compatible_rfft(y_s1, signal_ndim=2)
        Y_s1 = Y_s1.unsqueeze(2)

        x_s2, d_s2 = self.head_p(y_s1)
        params_s2 = self.hypa_p(y_s1)

        for i in range(self.n_iter):
            stage_params = params_s2[:, i, :, :, :]
            alpha_x_s2 = stage_params[:, 0:1]
            beta_x_s2 = stage_params[:, 1:2]
            alpha_d_s2 = stage_params[:, 2:3]
            beta_d_s2 = stage_params[:, 3:4]

            x_s2, d_s2 = self.pdm(x_s2, d_s2, y_s1, Y_s1, alpha_x_s2, beta_x_s2, alpha_d_s2, beta_d_s2,
                                      0.001)
            dx_s2 = self.tail(x_s2 , d_s2)
            dx_s2= dx_s2 [..., :h, :w]

            pred = dx_s2
            preds.append(pred)

        if self.training:
            return preds,d_s2,x_s2
        else:
            return pred, d_s2,x_s2


    def Gen_reverse_subpol_with_interp(self, subpol):

        B, C, sub_h, sub_w = subpol.shape
        H = sub_h * 2
        W = sub_w * 2

        output = torch.zeros((B, C, H, W), dtype=subpol.dtype, device=subpol.device)

        # (channel_start, channel_end, y_offset, x_offset)
        sub_images_info = [
            (0, 3, 1, 1),   
            (3, 6, 0, 1),   
            (6, 9, 0, 0),   
            (9, 12, 1, 0)   
        ]
        for b in range(B):
            for (ch_start, ch_end, y_off, x_off) in sub_images_info:
                sub_img_data = subpol[b, ch_start:ch_end, :, :].detach().cpu().numpy()
                y_sub, x_sub = np.where(sub_img_data[0, :, :] != 0)   

                if len(y_sub) == 0:
                     
                    continue

                y_orig = y_sub * 2 + y_off
                x_orig = x_sub * 2 + x_off

                points = np.column_stack((y_orig, x_orig))
                values_r = sub_img_data[0, y_sub, x_sub]
                values_g = sub_img_data[1, y_sub, x_sub]
                values_b = sub_img_data[2, y_sub, x_sub]

                Y, X = np.indices((H, W), dtype=np.float64)
                grid_points = (Y, X)

                interpolated_r = griddata(points, values_r, grid_points, method='linear', fill_value=0.0)
                interpolated_g = griddata(points, values_g, grid_points, method='linear', fill_value=0.0)
                interpolated_b = griddata(points, values_b, grid_points, method='linear', fill_value=0.0)

                interpolated_rgb = np.stack([interpolated_r, interpolated_g, interpolated_b], axis=0)
                output[b, ch_start:ch_end, :, :] = torch.from_numpy(interpolated_rgb).to(subpol.device)
        return output
