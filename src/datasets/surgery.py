from .thumos import *

@DATASET_REGISTRY.register()
class Surgery(Thumos):
    def __init__(self, cfg, split):
        super().__init__(cfg, split)
