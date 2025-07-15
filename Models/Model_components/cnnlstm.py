import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTM(nn.Module):
    """
    CNN-LSTM Network for spatial-temporal sequence learning (regression version).
    Applies CNN layers (no pooling) to extract spatial features, then a 1x1 conv to combine channels,
    then flattens and passes to LSTM for temporal processing.
    """
    
    def __init__(self, 
                 input_channels=1,
                 cnn_features=[64, 128, 256],
                 lstm_hidden_size=128,
                 lstm_num_layers=1,
                 output_size=1,
                 dropout=0.2,
                 input_height=36,
                 input_width=41):
        super(CNNLSTM, self).__init__()
        
        self.input_channels = input_channels
        self.cnn_features = cnn_features
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.output_size = output_size
        self.input_height = input_height
        self.input_width = input_width
        
        # CNN layers for spatial feature extraction (no pooling)
        self.cnn_layers = nn.ModuleList()
        
        # First CNN layer
        self.cnn_layers.append(
            nn.Sequential(
                nn.Conv2d(input_channels, cnn_features[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(cnn_features[0]),
                nn.ReLU(inplace=True),
                nn.Conv2d(cnn_features[0], cnn_features[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(cnn_features[0]),
                nn.ReLU(inplace=True)
            )
        )
        
        # Additional CNN layers (no pooling)
        for i in range(1, len(cnn_features)):
            self.cnn_layers.append(
                nn.Sequential(
                    nn.Conv2d(cnn_features[i-1], cnn_features[i], kernel_size=3, padding=1),
                    nn.BatchNorm2d(cnn_features[i]),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(cnn_features[i], cnn_features[i], kernel_size=3, padding=1),
                    nn.BatchNorm2d(cnn_features[i]),
                    nn.ReLU(inplace=True)
                )
            )
        
        # 1x1 convolution to combine channels
        self.conv1x1 = nn.Conv2d(cnn_features[-1], 1, kernel_size=1)
        
        # LSTM layer for temporal processing
        self.lstm_input_size = input_height * input_width  # after 1x1 conv, channels=1
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Output layer: map LSTM hidden state to flattened spatial map
        self.output_layer = nn.Sequential(
            nn.Linear(lstm_hidden_size, input_height * input_width)
        )
        
    def forward(self, x):
        """
        Forward pass of CNN-LSTM network (regression version)
        Args:
            x: Input tensor of shape (batch_size, time_steps, channels, height, width)
        Returns:
            Output tensor of shape (batch_size, 1, output_size)
        """
        batch_size, time_steps, channels, height, width = x.shape
        assert height == self.input_height and width == self.input_width, "Input H/W must match model initialization."
        
        cnn_outputs = []
        for t in range(time_steps):
            x_t = x[:, t, :, :, :]  # (batch_size, channels, height, width)
            cnn_out = x_t
            for cnn_layer in self.cnn_layers:
                cnn_out = cnn_layer(cnn_out)
            cnn_out = self.conv1x1(cnn_out)  # (batch_size, 1, H, W)
            cnn_out = cnn_out.view(batch_size, -1)  # flatten (batch_size, H*W)
            cnn_outputs.append(cnn_out)
        # Stack for LSTM: (batch_size, time_steps, H*W)
        lstm_input = torch.stack(cnn_outputs, dim=1)
        lstm_out, (hidden, cell) = self.lstm(lstm_input)
        last_output = lstm_out[:, -1, :]  # (batch_size, lstm_hidden_size)
        output = self.output_layer(last_output)  # (batch_size, height*width)
        output = output.view(batch_size, 1, self.input_height, self.input_width)  # (batch_size, 1, height, width)
        return output
