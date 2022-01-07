"""Conv block and NN models."""
import torch
from torch import nn

class ConvBlock(nn.Module):
    """Convolutional block.

    Parameters
    ----------
    in_channels : int
        Input channels size.
    layout : str
        Layout of layers. Can contain "c" for convolution, "a" for activation, "n" for batchnorm,
        "p" for maxpooling, "d" for dropout. E.g. of layout: "ccnapd".
    filters : int, list or None
        Number of filters for convolutions. Can be a single number (all convolutions will
        have the same number of filters), a list of the same length as a count of letters "c"
        in the layout, or None if the layout contains no "c".
    rate : float
        Dropout rate parameter. Default to 0.1.
    activation : function
        Activation function. If not specified activation is tf.nn.elu.
    """
    def __init__(self, layout, in_channels=None, filters=None, kernel_size=3, stride=1,#pylint:disable=too-many-branches
                 padding='same', rate=0.1, activation=None):
        super().__init__()
        seq = []
        i = 0
        k = kernel_size
        try:
            iter(filters)
        except TypeError:
            filters = list([filters] * (layout.count('c') + layout.count('t')))

        if isinstance(padding, str):
            if padding == 'same':
                padding = k // 2
            else:
                raise NotImplementedError('padding = {}'.format(padding))

        for s in layout:
            if s == 'c':
                seq.append(nn.Conv2d(in_channels, filters[i], (k, k), padding=padding, stride=stride))
                in_channels = filters[i]
                i += 1
            elif s == 't':
                seq.append(nn.ConvTranspose2d(in_channels, filters[i], (k, k), stride=2,
                                              padding=k//2, output_padding=1))
                in_channels = filters[i]
                i += 1
            elif s == 'a':
                if activation is None:
                    activation = nn.ELU()
                seq.append(activation)
            elif s == 'p':
                seq.append(nn.MaxPool2d((2, 2), stride=(2, 2)))
            elif s == 'n':
                seq.append(nn.BatchNorm2d(in_channels, momentum=0.9))
            elif s == 'd':
                seq.append(nn.Dropout2d(p=rate))
            else:
                raise KeyError('unknown letter {0}'.format(s))
        self.layers = nn.Sequential(*seq)

    def forward(self, x):
        """Forward pass."""
        return self.layers(x)


class Up(nn.Module):
    """Upsample block."""
    def __init__(self, layout, in_channels, out_channels, **kwargs):
        super().__init__()
        self.upsample = ConvBlock('tad', in_channels, in_channels, **kwargs)
        self.conv = ConvBlock(layout, 3*out_channels, out_channels, **kwargs)

    def forward(self, x, y):
        """Forward pass."""
        x = self.upsample(x)
        x = torch.cat([x, y], dim=1)
        return self.conv(x)


class Unet(nn.Module):
    """U-net implementation.

    Parameters
    ----------
    in_channels : int
        Input channels size.
    depth : int
        Depth of the U-net.
    init_filters : int
        Number of filters in the first conv block.
    output : dict
        Output conv block config.
    """
    def __init__(self, in_channels, depth, init_filters, kernel_size=3, output=None, norm=True):
        super().__init__()
        self.down = nn.ModuleList()
        self.bottom = None
        self.up = nn.ModuleList()
        self.out = None

        layout = 'cancan' if norm else 'caca'
        for d in range(depth):
            out_channels = init_filters * (2 ** d)
            self.down.append(ConvBlock(layout, in_channels, out_channels, kernel_size=kernel_size))
            self.down.append(ConvBlock('pd'))
            in_channels = out_channels

        self.bottom = ConvBlock(layout, in_channels, init_filters*(2**depth),
                                kernel_size=kernel_size)

        in_channels = init_filters * (2 ** depth)

        for d in range(depth - 1, -1, -1):
            out_channels = init_filters*(2**d)
            self.up.append(Up(layout, in_channels, out_channels, kernel_size=kernel_size))
            in_channels = out_channels

        if output is not None:
            self.out = ConvBlock(**output, in_channels=in_channels)

    def forward(self, x):
        """Forward pass."""
        down = [x]
        for layer in self.down:
            down.append(layer(down[-1]))
        x = self.bottom(down[-1])
        for i, layer in enumerate(self.up):
            y = down[-2*i-2]
            x = layer(x, y)

        return self.out(x) if self.out is not None else x


class Encoder(nn.Module):
    """Encoder module.

    Parameters
    ----------
    in_channels : int
        Input channels size.
    filters : array
        Sequence of filters in convolution layers.
    kernel_size : int
        Kernel size in convolution layers. Default 3.
    norm : bool
        Include normalization layers. Default True.
    output : dict
        Output conv block config.
    """
    def __init__(self, in_channels, filters, output=None,
                 kernel_size=3, norm=True):
        super().__init__()
        self.down = nn.ModuleList()
        self.out = None

        layout = 'can' if norm else 'ca'
        for f in filters:
            self.down.append(ConvBlock(layout, in_channels, f, kernel_size=kernel_size))
            self.down.append(ConvBlock(layout, f, f,
                                       kernel_size=kernel_size,
                                       stride=2))
            in_channels = f

        if output is not None:
            self.out = ConvBlock(**output, in_channels=in_channels)

    def forward(self, x):
        """Forward pass."""
        for layer in self.down:
            x = layer(x)
        return self.out(x) if self.out is not None else x


class Decoder(nn.Module):
    """Decoder module.

    Parameters
    ----------
    in_channels : int
        Input channels size.
    filters : array
        Sequence of filters in convolution layers.
    kernel_size : int
        Kernel size in convolution layers. Default 3.
    norm : bool
        Include normalization layers. Default True.
    output : dict
        Output conv block config.
    """
    def __init__(self, in_channels, filters, output=None,
                 kernel_size=3, norm=True):
        super().__init__()
        self.up = nn.ModuleList()
        self.out = None

        layout = 'can' if norm else 'ca'
        for f in filters:
            self.up.append(nn.Upsample(scale_factor=2, mode='bilinear',
                                       align_corners=False))
            self.up.append(ConvBlock(layout, in_channels, f, kernel_size=kernel_size))
            in_channels = f

        if output is not None:
            self.out = ConvBlock(**output, in_channels=in_channels)

    def forward(self, x):
        """Forward pass."""
        if x.ndim == 2:
            x = x.view(x.size(0), x.size(1), 1, 1)
        for layer in self.up:
            x = layer(x)
        return self.out(x) if self.out is not None else x


class VAE(nn.Module):
    """Convolutional VAE model.

    Parameters
    ----------
    in_channels : int
        Input channels size.
    filters_enc : array
        Sequence of filters for encoder.
    filters_enc : array
        Sequence of filters for decoder.
    z_dim : int
        Number of filters in the latent space.
    output : dict
        Output conv block config for decoder.
    kernel_size : int
        Kernel size in convolution layers. Default 3.
    norm : bool
        Include normalization layers. Default True.
    variational : bool
        If False, implemnt ordinary AE scheme. Default True.
    """
    def __init__(self, in_channels, filters_enc, filters_dec, z_dim, output,
                 kernel_size=3, norm=True, variational=True):
        super().__init__()
        self.variational = variational
        self.z_dim = z_dim
        self.last_f = filters_enc[-1]
        self.enc = Encoder(in_channels, filters=filters_enc, norm=norm, kernel_size=kernel_size)
        self.dec = Decoder(self.last_f, filters=filters_dec, norm=norm, kernel_size=kernel_size,
                           output=output)
        self.hid_mu = ConvBlock('c', self.last_f, self.z_dim)
        self.hid_var = ConvBlock('c', self.last_f, self.z_dim) if variational else None
        self.hid_z = ConvBlock('c', self.z_dim, self.last_f)

    def forward(self, x):
        """Forward pass."""
        enc = self.enc(x)
        mu = self.hid_mu(enc)
        if self.variational:
            logvar = self.hid_var(enc)
            z = self.reparameterize(mu, logvar) if self.training else mu
            return self.decode(z), mu, logvar
        return self.decode(mu), mu

    def encode(self, x):
        """Encoder."""
        return self.hid_mu(self.enc(x))

    def decode(self, z):
        """Decoder."""
        return self.dec(self.hid_z(z))

    def reparameterize(self, mu, logvar):
        """Reparametrization trick."""
        std = logvar.mul(0.5).exp_()
        eps = torch.randn(*mu.size()).to(mu.device)
        z = mu + std * eps
        return z.view(mu.shape)
