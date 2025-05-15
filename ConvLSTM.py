import torch.nn as nn
import torch

class ConvLSTMCell(nn.Module):
    """
    ConvLSTM Cell implementation for spatial temporal sequence learning
    """
    def __init__(self,input_channels,hidden_channels,kernel_size,padding):
        super(ConvLSTMCell,self).__init__()

        self.input_channels=input_channels
        self.hidden_channels=hidden_channels
        self.kernel_size=kernel_size
        self.padding=padding

        self.conv=nn.Conv2d(
            in_channels=input_channels+hidden_channels,
            out_channels=4*hidden_channels,
            padding=padding,
            kernel_size=kernel_size,
            bias=True
        )

    def forward(self,x,h_prev,c_prev):
        """
        Forward pass of ConvLSTM cell
        """
        combined=torch.cat([x,h_prev],dim=1)

        gates=self.conv(combined)
        
        i,f,o,g=torch.split(gates,self.hidden_channels,dim=1)
        # Apply activations
        i = torch.sigmoid(i)  # input gate
        f = torch.sigmoid(f)  # forget gate
        o = torch.sigmoid(o)  # output gate
        g = torch.tanh(g)     # candidate cell state

        # Update cell state and hidden state
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next