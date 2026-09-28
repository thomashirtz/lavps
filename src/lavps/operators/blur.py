from abc import ABC, abstractmethod
from math import pi
from pathlib import Path

import numpy as np
import scipy
import torch
from PIL import Image, ImageDraw, ImageFilter
from einops import rearrange
from scipy.signal import convolve
from torch import Tensor, nn

from lavps.operators.base import H_functions

eps = 0.1


# NOTE this is a copy/paste of https://github.com/LeviBorodenko/motionblur/blob/master/motionblur.py
# NOTE adapted to fix the seed as the path of the motion blur is randomly generated
def polar2z(r: np.ndarray, θ: np.ndarray) -> np.ndarray:
    """[summary]
    Takes a list of radii and angles (radians) and
    converts them into a corresponding list of complex
    numbers x + yi.
    [description]

    Arguments:
        r {np.ndarray} -- radius
        θ {np.ndarray} -- angle

    Returns:
        [np.ndarray] -- list of complex numbers r e^(i theta) as x + iy
    """
    return r * np.exp(1j * θ)

class Blur(H_functions):
    """Wrapper around ``GaussialBlurOperator`` that follows ``H_functions`` API.

    Notes
    -----
    Default parameters were borrowed from
    https://github.com/DPS2022/diffusion-posterior-sampling/blob/main/configs/gaussian_deblur_config.yaml
    """

    def __init__(self, kernel_size=61, intensity=3.0, device="cpu", num_channels: int = 3):
        super().__init__()

        self.degradation = GaussialBlurOperator(kernel_size, intensity, device, num_channels=num_channels)
        self.degradation.conv.requires_grad_(False)

    def H(self, x: Tensor):
        n_samples = x.shape[0]
        return self.degradation.forward(x).view(n_samples, -1)

    def Ht(self, x: Tensor):
        n_samples = x.shape[0]
        return self.degradation.transpose(x).view(n_samples, -1)

    # HACK: this is not the pseudo inverse of the operator
    # As long as we are using this method to display the degraded image (observation)
    def H_pinv(self, x: Tensor):
        return x


class MotionBlur(H_functions):
    """Wrapper around ``MotionBlurOperator`` that follows ``H_functions`` API.

    Notes
    -----
    Default parameters were borrowed from
    https://github.com/DPS2022/diffusion-posterior-sampling/blob/main/configs/motion_deblur_config.yaml
    """

    def __init__(self, kernel_size=61, intensity=0.5, device="cpu", seed=0, imshape=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.intensity = intensity
        self.device = device
        self.seed = seed  # just the initial/default seed

        self.degradation = MotionBlurOperator(kernel_size, intensity, device, seed=seed)
        self.degradation.conv.requires_grad_(False)

        self.imshape =imshape
        if imshape is not None:
            *_, self.c, self.h, self.w = imshape

    def update_with_seed(self, seed=None): # change to update with seed when updating git
        self.degradation = MotionBlurOperator(self.kernel_size, self.intensity, self.device, seed=seed)
        self.degradation.conv.requires_grad_(False)

    def H(self, x: Tensor):
        # n_samples = x.shape[0]
        batch_shape = x.shape[:-3]
        # invfussion expects the output of the form (*batch_shape, ...) I hope it doesn't break the code elsewhere
        return self.degradation.forward(x).view(*batch_shape, -1)

    def Ht(self, x: Tensor):
        # n_samples = x.shape[0]
        batch_shape = x.shape[:-3]
        return self.degradation.transpose(x).view(*batch_shape, -1)

    # HACK: this is not the pseudo inverse of the operator
    # As long as we are using this method to display the degraded image (observation)
    def H_pinv(self, x: Tensor):
        # invfussion except the shape to be (*batch_shape, c, h, w), exactly the same shape as the original image
        if self.imshape is not None:
            x = rearrange(x, "... (c h w) -> ... c h w", c=self.c, h=self.h, w=self.w)
        return x


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass

    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)


class MotionBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device, seed=None):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(
            blur_type="motion", kernel_size=kernel_size, std=intensity, device=device, seed=seed,
        ).to(
            device
        )  # should we keep this device term?

        self.kernel = Kernel(size=(kernel_size, kernel_size), intensity=intensity, seed=seed)
        kernel = torch.tensor(self.kernel.kernelMatrix, dtype=torch.float32)
        self.conv.update_weights(kernel)

    def forward(self, data, **kwargs):
        # A^T * A
        return self.conv(data)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        kernel = self.kernel.kernelMatrix.type(torch.float32).to(self.device)
        return kernel.view(1, 1, self.kernel_size, self.kernel_size)


class GaussialBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device, num_channels: int = 3):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(
            blur_type="gaussian", kernel_size=kernel_size, std=intensity, device=device, num_channels=num_channels
        ).to(device)
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))

    def forward(self, data, **kwargs):
        return self.conv(data)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        return self.kernel.view(1, 1, self.kernel_size, self.kernel_size)


class Kernel(object):
    """[summary]
    Class representing a motion blur kernel of a given intensity.

    [description]
    Keyword Arguments:
            size {tuple} -- Size of the kernel in px times px
            (default: {(100, 100)})

            intensity {float} -- Float between 0 and 1.
            Intensity of the motion blur.

            :   0 means linear motion blur and 1 is a highly non linear
                and often convex motion blur path. (default: {0})

    Attribute:
    kernelMatrix -- Numpy matrix of the kernel of given intensity

    Properties:
    applyTo -- Applies kernel to image
               (pass as path, pillow image or np array)

    Raises:
        ValueError
    """

    def __init__(self, size: tuple = (100, 100), intensity: float = 0, seed: int = None):

        # checking if size is correctly given
        if not isinstance(size, tuple):
            raise ValueError("Size must be TUPLE of 2 positive integers")
        elif len(size) != 2 or type(size[0]) != type(size[1]) != int:
            raise ValueError("Size must be tuple of 2 positive INTEGERS")
        elif size[0] < 0 or size[1] < 0:
            raise ValueError("Size must be tuple of 2 POSITIVE integers")

        # check if intensity is float (int) between 0 and 1
        if type(intensity) not in [int, float, np.float32, np.float64]:
            raise ValueError("Intensity must be a number between 0 and 1")
        elif intensity < 0 or intensity > 1:
            raise ValueError("Intensity must be a number between 0 and 1")

        # saving args
        self.SIZE = size
        self.INTENSITY = intensity

        # deriving quantities
        self.rng = np.random.RandomState(seed)

        # we super size first and then downscale at the end for better
        # anti-aliasing
        self.SIZEx2 = tuple([2 * i for i in size])
        self.x, self.y = self.SIZEx2

        # getting length of kernel diagonal
        self.DIAGONAL = (self.x**2 + self.y**2) ** 0.5

        # flag to see if kernel has been calculated already
        self.kernel_is_generated = False

    def _createPath(self):
        """[summary]
        creates a motion blur path with the given intensity.
        [description]
        Proceede in 5 steps
        1. Get a random number of random step sizes
        2. For each step get a random angle
        3. combine steps and angles into a sequence of increments
        4. create path out of increments
        5. translate path to fit the kernel dimensions

        NOTE: "random" means random but might depend on the given intensity
        """

        # first we find the lengths of the motion blur steps
        def getSteps():
            """[summary]
            Here we calculate the length of the steps taken by
            the motion blur
            [description]
            We want a higher intensity lead to a longer total motion
            blur path and more different steps along the way.

            Hence we sample

            MAX_PATH_LEN =[U(0,1) + U(0, intensity^2)] * diagonal * 0.75

            and each step: beta(1, 30) * (1 - self.INTENSITY + eps) * diagonal)
            """

            # getting max length of blur motion
            self.MAX_PATH_LEN = (
                0.75
                * self.DIAGONAL
                * (self.rng.uniform() + self.rng.uniform(0, self.INTENSITY**2))
            )

            # getting step
            steps = []

            while sum(steps) < self.MAX_PATH_LEN:

                # sample next step
                step = self.rng.beta(1, 30) * (1 - self.INTENSITY + eps) * self.DIAGONAL
                if step < self.MAX_PATH_LEN:
                    steps.append(step)

            # note the steps and the total number of steps
            self.NUM_STEPS = len(steps)
            self.STEPS = np.asarray(steps)

        def getAngles():
            """[summary]
            Gets an angle for each step
            [description]
            The maximal angle should be larger the more
            intense the motion is. So we sample it from a
            U(0, intensity * pi)

            We sample "jitter" from a beta(2,20) which is the probability
            that the next angle has a different sign than the previous one.
            """

            # same as with the steps

            # first we get the max angle in radians
            self.MAX_ANGLE = self.rng.uniform(0, self.INTENSITY * pi)

            # now we sample "jitter" which is the probability that the
            # next angle has a different sign than the previous one
            self.JITTER = self.rng.beta(2, 20)

            # initialising angles (and sign of angle)
            angles = [self.rng.uniform(low=-self.MAX_ANGLE, high=self.MAX_ANGLE)]

            while len(angles) < self.NUM_STEPS:

                # sample next angle (absolute value)
                angle = self.rng.triangular(
                    0, self.INTENSITY * self.MAX_ANGLE, self.MAX_ANGLE + eps
                )

                # with jitter probability change sign wrt previous angle
                if self.rng.uniform() < self.JITTER:
                    angle *= -np.sign(angles[-1])
                else:
                    angle *= np.sign(angles[-1])

                angles.append(angle)

            # save angles
            self.ANGLES = np.asarray(angles)

        # Get steps and angles
        getSteps()
        getAngles()

        # Turn them into a path
        ####

        # we turn angles and steps into complex numbers
        complex_increments = polar2z(self.STEPS, self.ANGLES)

        # generate path as the cumsum of these increments
        self.path_complex = np.cumsum(complex_increments)

        # find center of mass of path
        self.com_complex = sum(self.path_complex) / self.NUM_STEPS

        # Shift path s.t. center of mass lies in the middle of
        # the kernel and a apply a random rotation
        ###

        # center it on COM
        center_of_kernel = (self.x + 1j * self.y) / 2
        self.path_complex -= self.com_complex

        # randomly rotate path by an angle a in (0, pi)
        self.path_complex *= np.exp(1j * self.rng.uniform(0, pi))

        # center COM on center of kernel
        self.path_complex += center_of_kernel

        # convert complex path to final list of coordinate tuples
        self.path = [(i.real, i.imag) for i in self.path_complex]

    def _createKernel(self, save_to: Path = None, show: bool = False):
        """[summary]
        Finds a kernel (psf) of given intensity.
        [description]
        use displayKernel to actually see the kernel.

        Keyword Arguments:
            save_to {Path} -- Image file to save the kernel to. {None}
            show {bool} -- shows kernel if true
        """

        # check if we haven't already generated a kernel
        if self.kernel_is_generated:
            return None

        # get the path
        self._createPath()

        # Initialise an image with super-sized dimensions
        # (pillow Image object)
        self.kernel_image = Image.new("RGB", self.SIZEx2)

        # ImageDraw instance that is linked to the kernel image that
        # we can use to draw on our kernel_image
        self.painter = ImageDraw.Draw(self.kernel_image)

        # draw the path
        self.painter.line(xy=self.path, width=int(self.DIAGONAL / 150))

        # applying gaussian blur for realism
        self.kernel_image = self.kernel_image.filter(
            ImageFilter.GaussianBlur(radius=int(self.DIAGONAL * 0.01))
        )

        # Resize to actual size
        self.kernel_image = self.kernel_image.resize(self.SIZE, resample=Image.LANCZOS)

        # convert to gray scale
        self.kernel_image = self.kernel_image.convert("L")

        # flag that we have generated a kernel
        self.kernel_is_generated = True

    def displayKernel(self, save_to: Path = None, show: bool = True):
        """[summary]
        Finds a kernel (psf) of given intensity.
        [description]
        Saves the kernel to save_to if needed or shows it
        is show true

        Keyword Arguments:
            save_to {Path} -- Image file to save the kernel to. {None}
            show {bool} -- shows kernel if true
        """

        # generate kernel if needed
        self._createKernel()

        # save if needed
        if save_to is not None:

            save_to_file = Path(save_to)

            # save Kernel image
            self.kernel_image.save(save_to_file)
        else:
            # Show kernel
            self.kernel_image.show()

    @property
    def kernelMatrix(self) -> np.ndarray:
        """[summary]
        Kernel matrix of motion blur of given intensity.
        [description]
        Once generated, it stays the same.
        Returns:
            numpy ndarray
        """

        # generate kernel if needed
        self._createKernel()
        kernel = np.asarray(self.kernel_image, dtype=np.float32)
        kernel /= np.sum(kernel)

        return kernel

    @kernelMatrix.setter
    def kernelMatrix(self, *kargs):
        raise NotImplementedError("Can't manually set kernel matrix yet")

    def applyTo(self, image, keep_image_dim: bool = False) -> Image:
        """[summary]
        Applies kernel to one of the following:

        1. Path to image file
        2. Pillow image object
        3. (H,W,3)-shaped numpy array
        [description]

        Arguments:
            image {[str, Path, Image, np.ndarray]}
            keep_image_dim {bool} -- If true, then we will
                    conserve the image dimension after blurring
                    by using "same" convolution instead of "valid"
                    convolution inside the scipy convolve function.

        Returns:
            Image -- [description]
        """
        # calculate kernel if haven't already
        self._createKernel()

        def applyToPIL(image: Image, keep_image_dim: bool = False) -> Image:
            """[summary]
            Applies the kernel to an PIL.Image instance
            [description]
            converts to RGB and applies the kernel to each
            band before recombining them.
            Arguments:
                image {Image} -- Image to convolve
                keep_image_dim {bool} -- If true, then we will
                    conserve the image dimension after blurring
                    by using "same" convolution instead of "valid"
                    convolution inside the scipy convolve function.

            Returns:
                Image -- blurred image
            """
            # convert to RGB
            image = image.convert(mode="RGB")

            conv_mode = "valid"
            if keep_image_dim:
                conv_mode = "same"

            result_bands = ()

            for band in image.split():

                # convolve each band individually with kernel
                result_band = convolve(band, self.kernelMatrix, mode=conv_mode).astype(
                    "uint8"
                )

                # collect bands
                result_bands += (result_band,)

            # stack bands back together
            result = np.dstack(result_bands)

            # Get image
            return Image.fromarray(result)

        # If image is Path
        if isinstance(image, str) or isinstance(image, Path):

            # open image as Image class
            image_path = Path(image)
            image = Image.open(image_path)

            return applyToPIL(image, keep_image_dim)

        elif isinstance(image, Image.Image):

            # apply kernel
            return applyToPIL(image, keep_image_dim)

        elif isinstance(image, np.ndarray):

            # ASSUMES we have an array of the form (H, W, 3)
            ###

            # initiate Image object from array
            image = Image.fromarray(image)

            return applyToPIL(image, keep_image_dim)

        else:

            raise ValueError("Cannot apply kernel to this type.")


class Blurkernel(nn.Module):
    def __init__(self, blur_type="gaussian", kernel_size=31, std=3.0, device=None, num_channels: int = 3, seed=None):
        super().__init__()
        self.blur_type = blur_type
        self.kernel_size = kernel_size
        self.std = std
        self.seed = seed
        self.device = device
        self.num_channels = num_channels
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(self.kernel_size // 2),
            nn.Conv2d(
                num_channels, num_channels, self.kernel_size, stride=1, padding=0, bias=False, groups=num_channels # todo here
            ),
        ).to(self.device)

        self.weights_init()

    def forward(self, x):
        dtype = x.dtype
        *batch, C, H, W = x.shape
        x = x.to(torch.float32).reshape(-1, C, H, W)
        y = self.seq(x)
        y = y.reshape(*batch, C, H, W)
        return y.to(dtype)

    def weights_init(self):
        if self.blur_type == "gaussian":
            n = np.zeros((self.kernel_size, self.kernel_size))
            n[self.kernel_size // 2, self.kernel_size // 2] = 1
            k = scipy.ndimage.gaussian_filter(n, sigma=self.std)
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)
        elif self.blur_type == "motion":
            k = Kernel(
                size=(self.kernel_size, self.kernel_size), intensity=self.std, seed=self.seed,
            ).kernelMatrix
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)

    def update_weights(self, k):
        if not torch.is_tensor(k):
            k = torch.from_numpy(k).to(self.device)
        for name, f in self.named_parameters():
            f.data.copy_(k)

    def get_kernel(self):
        return self.k
