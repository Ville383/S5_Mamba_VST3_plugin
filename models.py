import math
import numpy as np
#from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from config import ModelParams
from init_ssm import apply_ssm, make_DPLR_HiPPO, discretize_bilinear
from film import FiLMGenerator

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1, num_layers=1):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, )
        self.linear = nn.Linear(hidden_size, output_size)
        self.hidden = None # Hidden state(s) of the LSTM

    def detach_hidden(self):
        self.hidden = tuple([h.detach() for h in self.hidden])

    def reset_hidden(self):
        self.hidden = None

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, input_size).

        Returns:
            Tensor: Output tensor with skip connection applied, shape (batch_size, seq_length, output_size).
        """
        # LSTM layer
        lstm_out, self.hidden = self.lstm(x, self.hidden)  # lstm_out: (batch_size, seq_length, hidden_size)
        # Linear layer
        linear_out = self.linear(lstm_out)  # linear_out: (batch_size, seq_length, output_size)
        # Skip connection: Add input `x` to the output of the linear layer
        output = x + linear_out  # Skip connection
        return output


class CausalConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode):
        super(CausalConv1D, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)
        self.buffer = None # Register buffer to persist state without being a parameter

    def forward(self, x):
        # x shape: (batch_size, in_channels, sequence_length)
        if self.buffer is None:
            # Initialize buffer with zeros for the first chunk
            self.buffer = torch.zeros(x.size(0), x.size(1), self.kernel_size - 1, device=x.device)
        # Concatenate the buffer with the current input
        x = torch.cat([self.buffer, x], dim=-1)
        
        out = self.conv(x)

        # Update buffer with the last kernel_size - 1 samples from the current input
        self.buffer = x[:, :, -self.kernel_size + 1:].detach()

        return out
    
    def reset_state(self):
        self.buffer = None


class S5SSM(nn.Module):
    def __init__(self, Lambda, B, C, D, inv_dt, conj_sym, step_rescale):
        super(S5SSM, self).__init__()
        self.A_real_log = nn.Parameter(Lambda.real)
        self.A_imag = nn.Parameter(Lambda.imag)
        self.B = nn.Parameter(B)
        self.C = nn.Parameter(C)
        self.D = nn.Parameter(D)
        self.inv_dt = nn.Parameter(inv_dt)
        self.conj_sym = conj_sym
        self.h = None
        self.step_rescale = step_rescale

    def forward(self, x):
        """ Input  (float32): (L, H)
            Output (float32): (L, H)
        """
        # left-half plane condition (inside unit circle in discrete time)
        #A = -torch.exp(self.A_real_log) + 1j * self.A_imag
        A = self.A_real_log + 1j * self.A_imag

        # discretize using the bilinear transform
        dt = self.step_rescale * F.softplus(self.inv_dt)
        dA, dB = discretize_bilinear(A, self.B, dt)

        # Apply: h[n] = Ah[n-1] + Bx[n]
        #        y[n] = real(Ch[n]) (+ Dx[n])
        y, h = apply_ssm(dA, dB, self.C, x, self.h, self.conj_sym)

        self.h = h.detach()

        Du = torch.vmap(lambda u: self.D * u)(x)
        return y + Du
    
    def reset_state(self):
        self.h = None

    def change_scale(self, step_rescale):
        self.step_rescale = step_rescale


class S5(nn.Module):
    def __init__(self, Lambda, B, C, D, inv_dt, conj_sym, step_rescale):
        super(S5, self).__init__()
        self.ssm = S5SSM(Lambda, B, C, D, inv_dt, conj_sym, step_rescale)

    def forward(self, x):
        # x (float32): (B, L, H)
        # Abstract for batch dimension
        return torch.vmap(lambda x: self.ssm(x))(x)
    
    def reset_state(self):
        self.ssm.reset_state()

    def change_scale(self, step_rescale):
        self.ssm.change_scale(step_rescale)


class MambaBlock(nn.Module):
    def __init__(self, args: ModelParams):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super(MambaBlock, self).__init__()
        self.step_rescale = args.step_rescale

        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)

        '''
        self.conv = CausalConv1D(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            kernel_size=args.kernel_size,
            stride=args.stride,
            padding=args.padding,
            dilation=args.dilation,
            groups=args.d_inner,
            bias=args.conv_bias,
            padding_mode=args.padding_mode
        )
        '''

        # Initialize state-space parameters A, B, C, D, dt
        block_size = int(args.d_state / args.blocks)

        Lambda, V = make_DPLR_HiPPO(block_size)

        if args.conj_sym:
            block_size = block_size // 2

        Lambda = Lambda[:block_size]
        V = V[:, :block_size]
        Vc = V.conj().T

        Lambda = (Lambda * torch.ones((args.blocks, block_size))).ravel()
        V = torch.block_diag(*([V] * args.blocks))
        Vinv = torch.block_diag(*([Vc] * args.blocks))

        B_init = torch.randn(args.d_state, args.d_inner, dtype=torch.float32) * math.sqrt(1.0 / args.d_state)
        B_tilde = Vinv @ B_init.type(Vinv.dtype)

        C_init = torch.randn(args.d_inner, args.d_state, dtype=torch.complex64) * math.sqrt(1.0 / args.d_inner)
        C_tilde = C_init @ V

        D = torch.randn(args.d_inner, dtype=torch.float32)

        
        dt = torch.exp(
            torch.rand(args.d_state, dtype=torch.float32) * (math.log(args.dt_max) - math.log(args.dt_min))
            + math.log(args.dt_min)
        ).clamp(min=args.dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))

        self.ssm = S5(Lambda, B_tilde, C_tilde, D, inv_dt, args.conj_sym, args.step_rescale)

        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)


    def forward(self, x):
        """
        Args:
            x: shape (b, l, d)
    
        Returns:
            output: shape (b, l, d)
        """
        x_and_res = self.in_proj(x) # (B, L, H) -> (B, L, d_in)
        u, res = x_and_res.chunk(chunks=2, dim=-1)
        """
        u = rearrange(u, 'b l d_in -> b d_in l')
        u = self.conv(u)
        u = rearrange(u, 'b d_in l -> b l d_in')
        """

        u = F.silu(u)
        y = self.ssm(u)

        y = y * F.silu(res)

        return self.out_proj(y)

    def reset_state(self):
        #self.conv.reset_state()
        self.ssm.reset_state()

    def change_scale(self, step_rescale):
        self.ssm.change_scale(step_rescale)


class ResidualBlock(nn.Module):
    def __init__(self, args: ModelParams):
        """Simple block wrapping Mamba block with FiLM conditioning, RMSNorm, and residual connection."""
        super(ResidualBlock, self).__init__()
        self.mamba = MambaBlock(args)
        #self.res = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.norm = nn.RMSNorm(args.d_model, eps = 1e-5)
    
    def forward(self, x, gamma, beta):
        """
        Args:
            x: shape (b, l, d)
            gamma, beta: shape (b, d)
    
        Returns:
            output: shape (b, l, d)
        """
        tmp = gamma.unsqueeze(1) * x + beta.unsqueeze(1) # FiLM conditioning
        return self.mamba(self.norm(tmp)) + x
    
    def reset_state(self):
        self.mamba.reset_state()

    def change_scale(self, step_rescale):
        self.mamba.change_scale(step_rescale)


class Mamba(nn.Module):
    def __init__(self, args: ModelParams):
        super(Mamba, self).__init__()
        self.n_layers = args.n_layers

        self.film_gen = FiLMGenerator(args.c_dim, args.d_model, args.n_layers)

        self.in_proj = nn.Linear(args.input_size, args.d_model, bias=args.bias)
        #with torch.no_grad():
        #    self.in_proj.weight.copy_(torch.ones_like(self.in_proj.weight))
        #    if args.bias:
        #        self.in_proj.bias.copy_(torch.zeros_like(self.in_proj.bias))

        self.mamba_blocks = nn.ModuleList([])
        for _ in range(0, args.n_layers):
            self.mamba_blocks.append(ResidualBlock(args))

        self.out_proj = nn.Linear(args.d_model, args.output_size, bias=args.bias)
        #with torch.no_grad():
        #    self.out_proj.weight.copy_(torch.ones_like(self.out_proj.weight))
        #    if args.bias:
        #        self.out_proj.bias.copy_(torch.zeros_like(self.out_proj.bias))

    def forward(self, x, c):
        # x (B, L, 1), c (B, c_dim)
        gamma, beta = self.film_gen(c) # tensor of tuples (len(n_layers)) with shape (B, 2*d_model)
        x = self.in_proj(x)

        for i in range(self.n_layers):
            x = self.mamba_blocks[i](x, gamma, beta)

        return self.out_proj(x)
    
    def reset_state(self):
        for mamba_block in self.mamba_blocks:
            mamba_block.reset_state()

    def change_scale(self, step_rescale):
        for i in range(self.n_layers):
            self.mamba_blocks[i].change_scale(step_rescale)