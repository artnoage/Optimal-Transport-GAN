from enum import Enum


class SummaryType(Enum):
    TEXT = 1
    SCALAR = 2
    IMAGE = 3
    HISTOGRAM = 4
    NON_TENSOR = 5