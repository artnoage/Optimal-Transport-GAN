from enum import Enum

class CostType(Enum):
    WGAN_SQUARE = 1
    WGAN_WEIGHT_CLIPPING = 2
    WGAN_GRADIENT_PENALTY = 3
    WGAN_SP = 4
    ASSIGNMENT = 5
    SAMPLE_SENSITIVE = 6
    KERNEL = 7
    GAN = 8
    WGAN_WEIGTHED_NORM = 9
    WGAN_COSINE = 10