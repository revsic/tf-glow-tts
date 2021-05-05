
class Config:
    """Configuration
    """
    def __init__(self, mel: int, vocabs: int):
        """Initializer.
        Args:
            mel: channels of the mel-spectrogram.
            vocabs: the number of the vocabularies.
        """
        self.factor = 2
        self.neck = mel * self.factor
        self.vocabs = vocabs

        # standard deviation of isotropic gaussian assumption
        self.temperature = 0.333

        # model
        self.channels = 192
        # prenet
        self.prenet_kernel = 5
        self.prenet_layers = 3
        self.prenet_groups = 4
        self.prenet_dropout = 0.5

        # encoder
        self.block_num = 6
        self.block_ffn = self.channels * 4
        self.block_heads = 2
        self.block_dropout = 0.1

        # decoder
        self.flow_groups = 4
        self.flow_block_num = 12
        self.wavenet_block_num = 4
        self.wavenet_cycle = 1
        self.wavenet_kernel_size = 5
        self.wavenet_dilation = 1

        # durator model
        self.dur_kernel = 3
        self.dur_layers = 2
        self.dur_dropout = self.block_dropout
