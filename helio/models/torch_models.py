"""Conv block and U-net model."""
import numpy as np
import torch
import torch.nn as nn

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
    transpose : bool
        If true, transposed convolutions are used.
    rate : float
        Dropout rate parameter. Default to 0.1.
    activation : function
        Activation function. If not specified activation is tf.nn.elu.
    """
    def __init__(self, layout, in_channels=None, filters=None, kernel_size=3, stride=1,
                 padding='same', transpose=False, rate=0.1, activation=None):
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
                if np.all(np.unique(stride) == 1):
                    padding = k // 2
                else:
                    raise ValueError('Stride should be 1 for "same" padding.')
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
        return self.layers(x)


class Up(nn.Module):
    """Upsample block."""
    def __init__(self, layout, in_channels, out_channels, **kwargs):
        super().__init__()
        self.up = ConvBlock('tad', in_channels, in_channels, **kwargs)
        self.conv = ConvBlock(layout, 3*out_channels, out_channels, **kwargs)

    def forward(self, x, y):
        x = self.up(x)
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

        self.bottom = ConvBlock(layout, in_channels, init_filters * (2 ** depth), kernel_size=kernel_size)

        in_channels = init_filters * (2 ** depth)

        for d in range(depth - 1, -1, -1):
            out_channels = init_filters * (2 ** d)
            self.up.append(Up(layout, in_channels, out_channels, kernel_size=kernel_size))
            in_channels = out_channels

        if output is not None:
            self.out = ConvBlock(**output, in_channels=in_channels)


    def forward(self, x):
        down = [x]
        for layer in self.down:
            down.append(layer(down[-1]))
        x = self.bottom(down[-1])
        for i, layer in enumerate(self.up):
            y = down[-2*i-2]
            x = layer(x, y)

        return self.out(x) if self.out is not None else x

class DownScale(nn.Module):
    """Downsample block."""
    def __init__(self, in_channels, filters, kernel_size=3, **kwargs):
        super().__init__()
        self.conv = ConvBlock('ca', in_channels, filters, kernel_size=2, stride=2, padding=0)
        self.out = ConvBlock('c', filters, in_channels, kernel_size=kernel_size, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return self.out(x)

class UpScale(nn.Module):
    """Upsample block."""
    def __init__(self, in_channels, filters, skip_channels, **kwargs):
        super().__init__()
        self.up = ConvBlock('tad', in_channels, filters, **kwargs)
        self.conv = ConvBlock('cancan', in_channels + filters + skip_channels, in_channels, **kwargs)

    def forward(self, x, y, z):
        x = self.up(x)
        x = torch.cat([x, y, z], dim=1)
        return self.conv(x)
    
class Snet(nn.Module):
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
    def __init__(self, segmentation_model, in_channels, out_channels, scales=4, scale_filters=8,
                 kernel_size=3):
        super().__init__()
        self.down = nn.ModuleList()
        self.seg_model = segmentation_model
        self.up = nn.ModuleList()

        for d in range(scales - 1):
            self.down.append(DownScale(in_channels, scale_filters, kernel_size=kernel_size))
            self.up.append(UpScale(in_channels=out_channels, filters=scale_filters,
                                   skip_channels=in_channels, kernel_size=kernel_size))

        self.sigmoid = nn.Sigmoid()

    def scale_mask(self, x):
        scaled = [x]
        for layer in self.down:
            scaled.append(layer(scaled[-1]))
        masks = [self.seg_model(z) for z in scaled]
        return scaled, masks

    def forward(self, x):
        scaled, masks = self.scale_mask(x)
        y = masks[-1]
        for i in range(len(masks)-1):
            y = self.up[i](y, masks[-i-2], scaled[-i-2])
        return y

