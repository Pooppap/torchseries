import torch
import random

from . import functional as F


class Compose(torch.nn.Module):

    def __init__(self, transforms):
        super().__init__()
        self.transforms = torch.nn.ModuleList(transforms)

    def forward(self, input):
        for t in self.transforms:
            input = t(input)
        return input


class RandomApply(torch.nn.Module):

    def __init__(self, transforms, p=0.5):
        super().__init__()
        self.p = p
        self.transforms = torch.nn.ModuleList(transforms)

    def forward(self, input):
        if self.p < random.random():
            return input
        
        for t in self.transforms:
            input = t(input)

        return input


class RandomChoiceApply(torch.nn.Module):

    def __init__(self, transforms, p=0.5):
        super().__init__()
        self.p = p
        self.transforms = torch.nn.ModuleList(transforms)

    def forward(self, input):
        t = random.choice(self.transforms)
        if self.p < random.random():
            return input
        return t(input)


class ToFloat(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.float()


class Transpose(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.t(input)


class Scale(torch.nn.Module):

    def __init__(self, mean=1.0, std=0.1, axis=0):
        super().__init__()
        self.std = std
        self.mean = mean
        self.axis = axis

    def forward(self, input):
        return F.scale(
            input,
            mean=self.mean,
            std=self.std,
            axis=self.axis
        )


class Jitter(torch.nn.Module):

    def __init__(self, mean=0.0, std=0.05):
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, input):
        return F.jitter(
            input,
            mean=self.mean,
            std=self.std
        )


class Rotate(torch.nn.Module):

    def __init__(self, axis=0, n_sensors=4, n_channels=6):
        super().__init__()
        self.axis = axis
        self.n_sensors = n_sensors
        self.n_channels = n_channels

    def forward(self, input):
        assert len(input.shape) == 2, f"Expected input to only have 2 dimensions. Instead receive input with dimension of {len(input.shape)}"
        if not self.axis:
            spatial_len, temporal_len = input.shape
        else:
            temporal_len, spatial_len = input.shape
            input = input.t()

        assert spatial_len == (self.n_sensors * self.n_channels), f"Incompatible dimension between input dimension {self.axis} ({spatial_len}) and product of n_sensors ({self.n_sensors}) and n_channels ({self.n_channels}). Input dimension {self.axis} is expected to be {self.n_sensors * self.n_channels}"

        input_reshaped = input.reshape((self.n_sensors * 2), 3, temporal_len)
        input_chunked = input_reshaped.chunk(self.n_sensors)
        input_rotated = torch.cat([F.rotate(chunk) for chunk in input_chunked])

        input_return = input_rotated.reshape(-1, temporal_len)
        if self.axis:
            return input_return.t()

        return input_return


class TimeWarp(torch.nn.Module):

    def __init__(self, axis=0, backend="cubic_spline", **kwargs):
        super().__init__()
        self.axis = axis
        self.backend = backend
        self.kwargs = kwargs

    def forward(self, input):
        assert len(input.shape) == 2, f"Expected input to only have 2 dimensions. Instead receive input with dimension of {len(input.shape)}"
        return F.time_warp(
            input,
            axis=self.axis,
            backend=self.backend,
            **self.kwargs
        )


class MagnitudeWarp(torch.nn.Module):

    def __init__(self, axis=0, backend="cubic_spline", **kwargs):
        super().__init__()
        self.axis = axis
        self.backend = backend
        self.kwargs = kwargs

    def forward(self, input):
        assert len(input.shape) == 2, f"Expected input to only have 2 dimensions. Instead receive input with dimension of {len(input.shape)}"
        return F.magnitude_warp(
            input,
            axis=self.axis,
            backend=self.backend,
            **self.kwargs
        )


class Permute(torch.nn.Module):

    def __init__(self, axis=0, n_segs=4, **kwargs):
        super().__init__()
        self.axis = axis
        self.n_segs = n_segs
        self.kwargs = kwargs

    def forward(self, input):
        assert len(input.shape) == 2, f"Expected input to only have 2 dimensions. Instead receive input with dimension of {len(input.shape)}"
        return F.permute(
            input,
            axis=self.axis,
            n_segs=self.n_segs,
            **self.kwargs
        )


class Passthrough(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input
