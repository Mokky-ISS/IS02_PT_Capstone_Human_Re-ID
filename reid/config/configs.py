from .default import DefaultConfig

class Config(DefaultConfig):
    def __init__(self):
        super(Config, self).__init__()
        self.CFG_NAME = 'baseline'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = 'model/resnet50-state.pth'
        self.LOSS_TYPE = 'triplet+softmax+center'
        self.TEST_WEIGHT = 'model/reid_R26.pth'



