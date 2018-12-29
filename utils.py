import math
import torchvision


def make_grid(tensor):
    return torchvision.utils.make_grid(tensor, nrow=math.round(math.sqrt(tensor.size(0))))
