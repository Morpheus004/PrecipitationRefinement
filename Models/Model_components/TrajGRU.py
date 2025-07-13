import torch
import torch.nn as nn
import torch.nn.functional as F

# Utility function for warping hidden states with flow fields
def wrap(input, flow):
    # input: (B, C, H, W), flow: (B, 2, H, W)
    B, C, H, W = input.size()
    device = input.device
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1).to(device)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W).to(device)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(input, vgrid, align_corners=True)
    return output

class TrajGRUCell(nn.Module):
    """
    Trajectory GRU Cell for spatiotemporal sequence modeling
    """
    def __init__(self, input_channels, hidden_channels, kernel_size, L=5, zoneout=0.0):
        super(TrajGRUCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.L = L
        self.zoneout = zoneout
        padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)

        # Input to hidden
        self.i2h = nn.Conv2d(
            in_channels=input_channels,
            out_channels=3 * hidden_channels,
            kernel_size=self.kernel_size,
            padding=padding
        )
        # Input to flow
        self.i2f_conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=5,
            padding=2
        )
        # Hidden to flow
        self.h2f_conv1 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=32,
            kernel_size=5,
            padding=2
        )
        # Generate L flows (each 2 channels)
        self.flows_conv = nn.Conv2d(
            in_channels=32,
            out_channels=L * 2,
            kernel_size=5,
            padding=2
        )
        # 1x1 conv to combine L warped hidden states
        self.ret = nn.Conv2d(
            in_channels=hidden_channels * L,
            out_channels=3 * hidden_channels,
            kernel_size=1
        )

    def _flow_generator(self, x, h):
        # x: (B, C, H, W) or None, h: (B, Hc, H, W)
        i2f = self.i2f_conv1(x) if x is not None else 0
        h2f = self.h2f_conv1(h)
        f = torch.tanh(i2f + h2f) if x is not None else torch.tanh(h2f)
        flows = self.flows_conv(f)
        flows = torch.split(flows, 2, dim=1)  # L tensors of (B, 2, H, W)
        return flows

    def forward(self, x, h_prev):
        # x: (B, C, H, W), h_prev: (B, Hc, H, W)
        B, _, H, W = h_prev.size()
        if x is not None:
            i2h = self.i2h(x)  # (B, 3*Hc, H, W)
            i2h_chunks = torch.chunk(i2h, 3, dim=1)
        else:
            i2h_chunks = [0, 0, 0]
        flows = self._flow_generator(x, h_prev)
        warped = [wrap(h_prev, -flow) for flow in flows]  # L tensors (B, Hc, H, W)
        wrapped_cat = torch.cat(warped, dim=1)  # (B, L*Hc, H, W)
        h2h = self.ret(wrapped_cat)  # (B, 3*Hc, H, W)
        h2h_chunks = torch.chunk(h2h, 3, dim=1)
        reset_gate = torch.sigmoid(i2h_chunks[0] + h2h_chunks[0])
        update_gate = torch.sigmoid(i2h_chunks[1] + h2h_chunks[1])
        new_mem = torch.tanh(i2h_chunks[2] + reset_gate * h2h_chunks[2])
        next_h = update_gate * h_prev + (1 - update_gate) * new_mem
        if self.zoneout > 0.0 and self.training:
            mask = F.dropout2d(torch.ones_like(next_h), p=self.zoneout)
            next_h = mask * next_h + (1 - mask) * h_prev
        return next_h

class TrajGRU(nn.Module):
    """
    Multi-step TrajGRU module for sequence modeling
    """
    def __init__(self, input_channels, hidden_channels, kernel_size, L=5, zoneout=0.0):
        super(TrajGRU, self).__init__()
        self.cell = TrajGRUCell(input_channels, hidden_channels, kernel_size, L, zoneout)
        self.hidden_channels = hidden_channels
        # Set output channels for refinement (default: input_channels)

        # Refinement layers similar to RefinementModel
        self.refine_layers = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, x, h0=None ):
        # x: (B, S, C, H, W)
        if x.dim() == 4:
            x = x.unsqueeze(1)  # (B, 1, C, H, W)
        B, S, C, H, W = x.size()
        x = x.permute(1, 0, 2, 3, 4)  # (S, B, C, H, W)
        if h0 is None:
            h = torch.zeros(B, self.hidden_channels, H, W, device=x.device, dtype=x.dtype)
        else:
            h = h0
        outputs = []
        for t in range(S):
            h = self.cell(x[t], h)
            outputs.append(h)
        outputs = torch.stack(outputs, dim=0)  # (S, B, hidden_channels, H, W)
        # if refine:
            # Apply refinement to the last hidden state (or all, as needed)
        refined = self.refine_layers(outputs[-1])  # (B, out_channels, H, W)
        return refined
        # return outputs, h
