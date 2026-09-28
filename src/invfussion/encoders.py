import torch

from invfussion.utils import persistence


@persistence.persistent_class
class Encoder:
    def __init__(self):
        pass

    def init(self, device): # force lazy init to happen now
        pass

    def __getstate__(self):
        return self.__dict__

    def encode(self, x): # raw pixels => final latents
        return self.encode_latents(self.encode_pixels(x))

    def encode_pixels(self, x): # raw pixels => raw latents
        raise NotImplementedError # to be overridden by subclass

    def encode_latents(self, x): # raw latents => final latents
        raise NotImplementedError # to be overridden by subclass

    def decode(self, x): # final latents => raw pixels
        raise NotImplementedError # to be overridden by subclass

@persistence.persistent_class
class StandardRGBEncoder(Encoder):
    def __init__(self):
        super().__init__()

    def encode_pixels(self, x): # raw pixels => raw latents
        return x

    def encode_latents(self, x): # raw latents => final latents
        return x.to(torch.float32) / 127.5 - 1

    def decode(self, x): # final latents => raw pixels
        return (x.to(torch.float32) * 127.5 + 128).clip(0, 255).to(torch.uint8)
