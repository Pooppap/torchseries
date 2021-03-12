import torch as _torch
import random as _random

from . import functional as _F


PI = 2 * _torch.acos(_torch.zeros(1))


class Compose(_torch.nn.Module):

    def __init__(self, transforms):
        super().__init__()
        self.transforms = _torch.nn.ModuleList(transforms)

    def forward(self, input):
        for t in self.transforms:
            input = t(input)
        return input


class RandomApply(_torch.nn.Module):

    def __init__(self, transforms, p=0.5):
        super().__init__()
        self.p = p
        self.transforms = _torch.nn.ModuleList(transforms)

    def forward(self, input):
        if self.p < _random.random():
            return input
        
        for t in self.transforms:
            input = t(input)

        return input


class RandomChoiceApply(_torch.nn.Module):

    def __init__(self, transforms, p=0.5):
        super().__init__()
        self.p = p
        self.transforms = _torch.nn.ModuleList(transforms)

    def forward(self, input):
        t = _random.choice(self.transforms)
        if self.p < _random.random():
            return input
        return t(input)


class ToFloat(_torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.float()


class Transpose(_torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return _torch.t(input)


class Scale(_torch.nn.Module):

    def __init__(self, mean=1.0, std=0.1, axis=0):
        super().__init__()
        self.std = std
        self.mean = mean
        self.axis = axis

    def forward(self, input):
        return _F.scale(
            input,
            mean=self.mean,
            std=self.std,
            axis=self.axis
        )


class Jitter(_torch.nn.Module):

    def __init__(self, mean=0.0, std=0.05):
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, input):
        return _F.jitter(
            input,
            mean=self.mean,
            std=self.std
        )


class Rotate(_torch.nn.Module):

    def __init__(self, axis=0, n_sensors=1, n_channels=6, bounds=(-PI/6, PI/6)):
        super().__init__()
        self.axis = axis
        self.n_sensors = n_sensors
        self.n_channels = n_channels
        self.bounds = bounds

    def forward(self, input):
        assert len(input.shape) == 2, f"Expected input to only have 2 dimensions. Instead receive input with dimension of {len(input.shape)}"
        if not self.axis:
            spatial_len, temporal_len = input.shape
        else:
            temporal_len, spatial_len = input.shape
            input = input.t()

        assert spatial_len == (self.n_sensors * self.n_channels), f"Incompatible dimension between input dimension {self.axis} ({spatial_len}) and product of n_sensors ({self.n_sensors}) and n_channels ({self.n_channels}). Input dimension {self.axis} is expected to be {self.n_sensors * self.n_channels}"

        input_reshaped = input.reshape((self.n_sensors * (self.n_channels // 3)), 3, temporal_len)
        input_chunked = input_reshaped.chunk(self.n_sensors)
        input_rotated = _torch.cat([_F.rotate(chunk, self.bounds) for chunk in input_chunked])

        input_return = input_rotated.reshape(-1, temporal_len)
        if self.axis:
            return input_return.t()

        return input_return


class TimeWarp(_torch.nn.Module):

    def __init__(self, axis=0, backend="cubic_spline", **kwargs):
        super().__init__()
        self.axis = axis
        self.backend = backend
        self.kwargs = kwargs

    def forward(self, input):
        assert len(input.shape) == 2, f"Expected input to only have 2 dimensions. Instead receive input with dimension of {len(input.shape)}"
        return _F.time_warp(
            input,
            axis=self.axis,
            backend=self.backend,
            **self.kwargs
        )


class MagnitudeWarp(_torch.nn.Module):

    def __init__(self, axis=0, backend="cubic_spline", **kwargs):
        super().__init__()
        self.axis = axis
        self.backend = backend
        self.kwargs = kwargs

    def forward(self, input):
        assert len(input.shape) == 2, f"Expected input to only have 2 dimensions. Instead receive input with dimension of {len(input.shape)}"
        return _F.magnitude_warp(
            input,
            axis=self.axis,
            backend=self.backend,
            **self.kwargs
        )


class Permute(_torch.nn.Module):

    def __init__(self, axis=0, n_segs=4, **kwargs):
        super().__init__()
        self.axis = axis
        self.n_segs = n_segs
        self.kwargs = kwargs

    def forward(self, input):
        assert len(input.shape) == 2, f"Expected input to only have 2 dimensions. Instead receive input with dimension of {len(input.shape)}"
        return _F.permute(
            input,
            axis=self.axis,
            n_segs=self.n_segs,
            **self.kwargs
        )


class Passthrough(_torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input
