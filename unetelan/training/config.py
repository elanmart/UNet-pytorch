class Config:
    __bn           = True
    __dp           = 0.2
    __n_layers     = 3
    __filters_base = 32
    __patch_border = 16
    __pad          = 1
    __n_classes    = 2
    __n_channels   = 3

    def __init__(self, bn=__bn, dp=__dp, n_layers=__n_layers, filters_base=__filters_base,
                 patch_border=__patch_border, pad=__pad, n_classes=__n_classes, n_channels=__n_channels):

        self.bn           = bn
        self.dp           = dp
        self.n_layers     = n_layers
        self.filters_base = filters_base
        self.patch_border = patch_border
        self.pad          = pad
        self.n_classes    = n_classes
        self.n_channels   = n_channels
