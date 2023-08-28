import torch

def sqrt_hann_win_fn(winlen):
    return torch.sqrt(torch.hann_window(winlen))

class MultichannelSTFTLayer(torch.nn.Module):
    def __init__(self, stft_framelen, stft_hopsize, win_fn):
        super().__init__()
        self.stft_framelen = stft_framelen
        self.stft_hopsize = stft_hopsize
        if win_fn == None:
            self.window = torch.nn.parameter.Parameter(torch.ones((self.stft_framelen,)), requires_grad=False)
        else:
            self.window = torch.nn.parameter.Parameter(win_fn(self.stft_framelen), requires_grad=False)
        
        self.normalization = torch.sqrt(
            self.stft_framelen / 2 * 
            torch.sum(self.window**2))

    def forward(self, x):
        '''
        input: multichannel signal (batch_size x samples x channels)

        returns: multichannel STFT (batch_size x frames x bins x channels)
        '''

        (batch_size, nsamples, nchannels) = x.size()
        x = x.permute(2, 0, 1) #channels first
        x = x.reshape(nchannels * batch_size, nsamples)
        y = torch.permute(torch.stft(x / self.normalization, self.stft_framelen,
                self.stft_hopsize, self.stft_framelen, 
                self.window.to(x.device), return_complex=True), [0, 2, 1])
        y = y.reshape(nchannels, batch_size, y.shape[1], y.shape[2])
        y = y.permute(1, 2, 3, 0) # channels last
        return y


class MultichannelISTFTLayer(torch.nn.Module):
    def __init__(self, stft_framelen, stft_hopsize, win_fn):
        super().__init__()
        self.stft_framelen = stft_framelen
        self.stft_hopsize = stft_hopsize
        if win_fn == None:
            self.window = torch.nn.parameter.Parameter(torch.ones((self.stft_framelen,)), requires_grad=False)
        else:
            self.window = torch.nn.parameter.Parameter(win_fn(self.stft_framelen), requires_grad=False)
        
        self.normalization = torch.sqrt(
            self.stft_framelen / 2 * 
            torch.sum(self.window**2))

    def forward(self, x, length):
        (batch_size, nframes, nbins, nchannels) = x.size()
        x = x.permute(3, 0, 1, 2) #channels first
        x = x.reshape(nchannels * batch_size, nframes, nbins)
        y =  torch.istft(torch.permute(x, [0, 2, 1]), 
            self.stft_framelen,
            self.stft_hopsize, self.stft_framelen, 
            self.window.to(x.device), length=length) * self.normalization
        y = y.reshape(nchannels, batch_size, y.shape[1])
        y = y.permute(1, 2, 0) # channels last
        return y
