# from https://github.com/haoxiangsnr/A-Convolutional-Recurrent-Neural-Network-for-Real-Time-Speech-Enhancement/blob/master/model/crn.py
import torch
from torch import nn
from third_party.grouped_gru import GroupedGRU

INIT_BIAS_ZERO = False

class CausalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, fstride, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, fstride),
            padding=(kernel_size[0]-1, 0),
            bias=bias
        )
        self.kernel_size = kernel_size

        if INIT_BIAS_ZERO:
            nn.init.zeros_(self.conv.bias)
       

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, T, F]
        Returns:
            [B, C, T, F]
        """
        x = self.conv(x)
        
        x = x[:, :, :x.shape[2]-self.kernel_size[0]+1, :]  # chomp size
        return x


class CausalTransConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, fstride, output_padding_f_axis=0, bias=True):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, fstride),
            output_padding=(0, output_padding_f_axis),
            bias=bias
        )
        self.kernel_size = kernel_size
        if INIT_BIAS_ZERO:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, T, F]
        Returns:
            [B, C, T, F]
        """
        x = self.conv(x)
        x = x[:, :, :x.shape[2]-self.kernel_size[0]+1, :]  # chomp size
        return x


class CausalGatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, fstride, nonlin, gate=True, batch_norm=False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, fstride),
            padding=(kernel_size[0]-1, 0),
            bias=not batch_norm
        )
        self.gate = gate
        if gate:
            self.conv_gate = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=(1, fstride),
                padding=(kernel_size[0]-1, 0),
                bias=True
            )

        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = nn.Identity()

        self.kernel_size = kernel_size
        self.elu = nonlin()


    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, T, F]
        Returns:
            [B, C, T, F]
        """
        if self.gate:
            xg = torch.sigmoid(self.conv_gate(x))
            x = self.conv(x)
            
            
            return self.elu(self.bn(xg * x)[:, :, :x.shape[2]-self.kernel_size[0]+1, :])

        else:
            x = self.conv(x)
            return self.elu(self.bn(x)[:, :, :x.shape[2]-self.kernel_size[0]+1, :])

class CausalTransGatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, fstride, nonlin, 
            gate=True, output_padding_f_axis=0, batch_norm=False):

        super().__init__()
        
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, fstride),
            output_padding=(0, output_padding_f_axis),
            bias=not batch_norm
        )
        self.gate = gate
        if gate:
            self.conv_gate = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=(1, fstride),
                output_padding=(0, output_padding_f_axis),
                bias=True
            )
        self.kernel_size = kernel_size
        self.elu = nonlin()
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = nn.Identity()


    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, T, F]
        Returns:
            [B, C, T, F]
        """
        if self.gate:
            xg = torch.sigmoid(self.conv_gate(x))
            x = self.conv(x)
            return self.elu(self.bn(xg * x)[:, :, :x.shape[2]-self.kernel_size[0]+1, :])
        else:
            x = self.conv(x)
            return self.elu(self.bn(x)[:, :, :x.shape[2]-self.kernel_size[0]+1, :])


class GCRN(nn.Module):
    def __init__(self, fsize_input, num_channels_encoder, num_channels_decoder,
        num_decoders, kernel_size, fstride, n_gru_layers, n_gru_groups, 
        nonlinearity, output_nonlinearity,
        batch_norm):
        
        super().__init__()

        # get from nn
        nonlinearity = getattr(nn, nonlinearity)
        output_nonlinearity = getattr(nn, output_nonlinearity)
        
        # Encoder
        fsize = fsize_input
        self.encoder_blocks = nn.ModuleList()
        for i in range(len(num_channels_encoder)-1):
            conv_layer = CausalGatedConvBlock(num_channels_encoder[i], num_channels_encoder[i+1],
                kernel_size, fstride,nonlin=nn.Identity)
            
            if batch_norm:
                self.encoder_blocks.append(nn.Sequential(
                    conv_layer,
                    nn.BatchNorm2d(num_channels_encoder[i+1]),
                    nonlinearity()))
            else:
                self.encoder_blocks.append(nn.Sequential(
                    conv_layer,
                    nonlinearity()))
            
            fsize = (fsize - kernel_size[1]) / fstride + 1
            assert fsize == int(fsize)

        recurr_input_size = int(fsize * num_channels_encoder[-1])

        # gru
        self.group_gru = GroupedGRU(input_size=recurr_input_size,
            hidden_size=recurr_input_size,
            num_layers=n_gru_layers,
            groups = n_gru_groups,
            bidirectional=False)

        # decoder
        #if type(num_channels_decoder[0]) == int:
        #    num_channels_decoder = [num_channels_decoder] * num_decoders
        self.decoders = nn.ModuleList()
        for h in range(num_decoders):
            decoder_blocks = nn.ModuleList()
            for i in range(len(num_channels_decoder[h])-1):
                conv_layer = CausalTransGatedConvBlock(
                    num_channels_decoder[h][i] + num_channels_encoder[-1-i],
                    num_channels_decoder[h][i+1],
                    kernel_size, fstride, nonlin=nn.Identity)
                
                if batch_norm:
                    block_wo_nonlin = nn.Sequential(
                        conv_layer,
                        nn.BatchNorm2d(num_channels_decoder[h][i+1]))
                else:
                    block_wo_nonlin = conv_layer

                if i < (len(num_channels_decoder[h]) - 2):
                    decoder_blocks.append(nn.Sequential(
                        block_wo_nonlin,
                        nonlinearity()))
                else:
                    decoder_blocks.append(nn.Sequential(
                        block_wo_nonlin,
                        output_nonlinearity()))
                    
                fsize = (fsize - 1) * fstride + kernel_size[1]
                assert fsize == int(fsize)
            self.decoders.append(decoder_blocks)




    def forward(self, x):
        encoder_outputs = []
        for l in self.encoder_blocks:
            
            x = l(x)
            encoder_outputs.append(x)

        batch_size, n_channels, n_frames, n_bins = x.shape

        #print(x[0, 0, 100:110, 20])
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size, n_frames, n_channels * n_bins)
        x, _ = self.group_gru(x)
        
        x = x.reshape(batch_size, n_frames, n_channels, n_bins)
        x = x.permute(0, 2, 1, 3)

        decoder_outputs = []
        for h in range(len(self.decoders)):
            d = x
            for i in range(len(self.decoders[h])):
                skip_connected = torch.cat([d, encoder_outputs[-1-i]], 1)
                d = self.decoders[h][i](skip_connected)
            decoder_outputs.append(d)

        return decoder_outputs
    

class IGCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            padding=(kernel_size[0]-1, (kernel_size[1]-1)//2),
            bias=True
        )

        self.conv_gate = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size[0]-1, (kernel_size[1]-1)//2),
            bias=True
        )
        self.kernel_size = kernel_size
        self.act = activation()

    def forward(self, x):
        """
        Args:
            x: [B, C, T, F]
        Returns:
            [B, C, T, F]
        """
        xg = torch.sigmoid(self.conv_gate(x))
        x = self.conv(x)
        
        return self.act((xg * x)[:, :, :x.shape[2]-self.kernel_size[0]+1, :])


class IGCRN(nn.Module):
    def __init__(self, num_input_channels, num_channels, kernel_size, 
        num_decoders, num_output_channels, depth, num_recurrent_layers,
        nonlinearity, output_nonlinearity):
        
        # get from nn
        nonlinearity = getattr(nn, nonlinearity)
        output_nonlinearity = getattr(nn, output_nonlinearity)

        super().__init__()

        encoder_layers = []
        encoder_layers.append(IGCBlock(num_input_channels, num_channels, kernel_size, nonlinearity))

        for _ in range(depth - 1):
            encoder_layers.append(IGCBlock(num_channels, num_channels, kernel_size, nonlinearity))
        
        self.encoder_layers = nn.ModuleList(encoder_layers)
        
        if num_decoders == 1 and type(num_output_channels) != list:
            num_output_channels = [num_output_channels]
        
        decoders = []
        for i in range(num_decoders):
            decoder_layers = []
            for _ in range(depth - 1):
                decoder_layers.append(IGCBlock(2 * num_channels, num_channels, kernel_size, nonlinearity))
            decoder_layers.append(IGCBlock(2 * num_channels, num_output_channels[i], kernel_size, output_nonlinearity))
            decoder_layers = nn.ModuleList(decoder_layers)  
            decoders.append(decoder_layers)

        self.decoders = nn.ModuleList(decoders)
        self.recurr = nn.GRU(num_channels, num_channels, num_layers = num_recurrent_layers, batch_first=True)
        
    def forward(self, x, hr = None):
        encoder_outputs = []
        for l in self.encoder_layers:
            x = l(x)
            encoder_outputs.append(x)
        xr = x.permute(0, 3, 2, 1)
        xr = xr.reshape(xr.shape[0] * xr.shape[1], xr.shape[2], xr.shape[3])
        xro, hrout = self.recurr(xr, hr)
        xro = xro.reshape(x.shape[0], x.shape[3], x.shape[2], x.shape[1]).permute(0, 3, 2, 1)
        

        outputs = []
        for d in self.decoders:
            y = xro
            for dli, dl in enumerate(d):
                y = dl(torch.concat([encoder_outputs[len(self.encoder_layers) - dli - 1], y], axis=1))
            outputs.append(y)
        if hr == None:
            return outputs
        else:
            return outputs, hrout