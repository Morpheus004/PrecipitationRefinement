import torch
import torch.nn as nn
from .Model_components import ConvLSTMCell

class RefinementModel(nn.Module):
    """
    ConvLSTM model for refining precipitation predictions to match IMERG data.
    This model takes prediction data and refines the second timestep to better match IMERG ground truth.
    """
    def __init__(self,input_channels=1,hidden_channels=[32,64,32],kernel_size=3):
        super(RefinementModel,self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # Feature extraction from input predictions
        self.conv_in = nn.Conv2d(
            in_channels=input_channels,
            out_channels=hidden_channels[0],
            kernel_size=kernel_size,
            padding=self.padding,
        )
        self.convlstm1 = ConvLSTMCell(
            input_channels=hidden_channels[0],
            hidden_channels=hidden_channels[1],
            kernel_size=kernel_size,
            padding=self.padding,
        )
        # self.spatial_attention = nn.Sequential(
        #     nn.Conv2d(hidden_channels[1], hidden_channels[2], kernel_size=1),
        #     nn.ReLU(),
        #     nn.Conv2d(hidden_channels[2], 1, kernel_size=1),
        #     nn.Sigmoid()
        # )
        self.refine_layers = nn.Sequential(
            nn.Conv2d(hidden_channels[1],hidden_channels[2],kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels[2],hidden_channels[2],kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels[2],input_channels,kernel_size=1),
        )

    def forward(self,pred_sequence):
        """
        Forward pass of the refinement model.
        
        Args:
            pred_sequence: prediction sequence tensor of shape (batch_size, seq_len, channels, height, width)
                            where seq_len=2 for current and next prediction timesteps
                            
        Returns:
            refined_prediction: refined precipitation prediction for the second timestep
        """
        batch_size,seq_len,channels,height,width=pred_sequence.size()
        device=pred_sequence.device

        h = torch.zeros(batch_size, self.hidden_channels[1], height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_channels[1], height, width, device=device)
        
        # Process first timestep (current)
        x_t1 = self.conv_in(pred_sequence[:, 0])
        h, c = self.convlstm1(x_t1, h, c)
        
        # Process second timestep (next)
        x_t2 = self.conv_in(pred_sequence[:, 1])
        h, c = self.convlstm1(x_t2, h, c)
        
        # Directly refine the hidden state features without attention
        refined_prediction = self.refine_layers(h)
        
        return refined_prediction
