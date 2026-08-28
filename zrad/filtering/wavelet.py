from math import factorial

import numpy as np
import pywt
from scipy import ndimage as ndi

from .base import BaseFilter


class Simoncelli(BaseFilter):
    """IBSI non-separable Simoncelli band-pass wavelet.

    The wavelet is evaluated directly in the Fourier domain with periodic boundary
    conditions. ``decomposition_level`` selects the B map; level one is the highest-
    frequency band. An optional Riesz multi-index applies the normalized higher-order
    Riesz transform to the B map.

    Parameters
    ----------
    padding_type : {"wrap", "periodic"}
        Periodic boundary handling (``periodic`` is an alias for ``wrap``).
    decomposition_level : int
        One-based scale level of the B map.
    dimensionality : {"2D", "3D"}
        In 2D mode each slice is filtered independently.
    riesz_order : tuple of int, optional
        Riesz multi-index, for example ``(0, 2)`` or ``(0, 2, 0)``. Its length must match
        the filter dimensionality. Omitting it returns the isotropic B map.
    """

    def __init__(self, padding_type, decomposition_level, dimensionality='3D', riesz_order=None):
        if padding_type not in ('wrap', 'periodic'):
            raise ValueError("Simoncelli filtering requires periodic padding ('wrap' or 'periodic').")
        if dimensionality not in ('2D', '3D'):
            raise ValueError("Simoncelli dimensionality must be '2D' or '3D'.")
        if not isinstance(decomposition_level, int) or isinstance(decomposition_level, bool) or decomposition_level < 1:
            raise ValueError('decomposition_level must be a positive integer.')

        dimensions = int(dimensionality[0])
        if riesz_order is not None:
            if len(riesz_order) != dimensions or any(
                    not isinstance(order, (int, np.integer)) or isinstance(order, (bool, np.bool_)) or order < 0
                    for order in riesz_order
            ):
                raise ValueError(f'riesz_order must contain {dimensions} non-negative integers.')
            riesz_order = tuple(int(order) for order in riesz_order)

        super().__init__(
            filtering_method='Simoncelli',
            padding_type='wrap',
            decomposition_level=decomposition_level,
            dimensionality=dimensionality,
            riesz_order=riesz_order,
        )
        self.decomposition_level = decomposition_level
        self.dimensionality = dimensionality
        self.riesz_order = riesz_order

    def _frequency_response(self, shape):
        # 1. IBSI grid generation: Linspace on [-pi, pi] inclusive with endpoint=True
        centered = np.meshgrid(*(np.linspace(-np.pi, np.pi, s, endpoint=True) for s in shape), indexing='ij')

        # 2. Shift coordinates to align with NumPy's unshifted FFT origin
        coordinates = [np.fft.ifftshift(c) for c in centered]

        # 3. Calculate Euclidean radial distance
        radius = np.sqrt(sum(c ** 2 for c in coordinates))

        # 4. IBSI Nyquist cutoff frequency for scale j
        nyquist = np.pi / (2 ** (self.decomposition_level - 1))

        response = np.zeros(shape, dtype=np.float64)

        # 5. Mask support region: [nyquist/4, nyquist]
        support = (radius >= nyquist / 4.0) & (radius <= nyquist)

        # 6. Isotropic Simoncelli B-map continuous cosine profile
        response[support] = np.cos((np.pi / 2.0) * np.log2(2.0 * radius[support] / nyquist))

        # 7. Higher-Order Riesz Multi-Index Transform
        if self.riesz_order is not None and sum(self.riesz_order) > 0:
            total_order = sum(self.riesz_order)
            coefficient = np.sqrt(factorial(total_order) / np.prod([factorial(o) for o in self.riesz_order]))

            numerator = np.ones(shape, dtype=np.float64)
            for c, order in zip(coordinates, self.riesz_order):
                numerator *= (c ** order)

            riesz = np.zeros(shape, dtype=np.complex128)
            nonzero = radius > 0
            riesz[nonzero] = ((-1j) ** total_order) * coefficient * numerator[nonzero] / (
                        radius[nonzero] ** total_order)
            response = response * riesz

        return response

    def _filter_periodic(self, image):
        freq_resp = self._frequency_response(image.shape)
        filtered_fft = np.fft.fftn(image) * freq_resp
        result = np.fft.ifftn(filtered_fft)

        # Total Riesz order parity determines whether spatial response is real or imaginary
        total_order = sum(self.riesz_order) if self.riesz_order is not None else 0

        if total_order % 2 == 0:
            # Even orders (e.g. isotropic or (0,2) where sum=2) yield purely real output
            return np.real(result)
        else:
            # Odd orders (e.g. (1,0) where sum=1) yield purely imaginary output
            return np.imag(result)

    def _apply_array(self, img):
        if self.dimensionality == '3D':
            return self._filter_periodic(img)

        # 2D filtering mode
        if img.ndim == 2:
            return self._filter_periodic(img)

        # 2D slice-by-slice filtering across 3D volumes
        # Automatically detects slice axis (axis 0 for (Z,Y,X) or axis 2 for (Y,X,Z))
        if img.shape[0] < img.shape[2]:
            return np.stack([self._filter_periodic(img[i, :, :]) for i in range(img.shape[0])], axis=0)
        else:
            return np.stack([self._filter_periodic(img[:, :, i]) for i in range(img.shape[2])], axis=2)


class Wavelets2D(BaseFilter):
    """2D separable wavelet filtering evaluated slice-wise.

    Each response map combines low-pass (``L``) and high-pass (``H``) wavelet
    kernels along the two in-plane axes. The result keeps the original image
    grid and is intended for downstream radiomics feature extraction.

    Parameters
    ----------
    wavelet_type : {"db3", "db2", "coif1", "haar"}
        Wavelet family used to obtain low- and high-pass filter kernels.
    padding_type : {"constant", "nearest", "wrap", "reflect"}
        Boundary handling mode used during convolution.
    response_map : {"LL", "HL", "LH", "HH"}
        Low/high-pass kernel combination for the two in-plane axes.
    decomposition_level : {1, 2}
        Wavelet decomposition level.
    rotation_invariance : bool, optional
        If true, average responses over four in-plane rotations.
    """

    def __init__(self, wavelet_type, padding_type, response_map, decomposition_level, rotation_invariance=False):
        super().__init__(
            filtering_method='Wavelets',
            wavelet_type=wavelet_type,
            padding_type=padding_type,
            response_map=response_map,
            decomposition_level=decomposition_level,
            rotation_invariance=rotation_invariance,
            dimensionality='2D',
        )

        self.dimensionality = '2D'

        if padding_type in ['constant', 'nearest', 'wrap', 'reflect']:
            self.padding_type = padding_type
        else:
            raise ValueError(
                f"Wrong padding type '{padding_type}'. "
                "Available padding types are: 'constant', 'nearest', 'wrap', and 'reflect'."
            )

        if wavelet_type in ['db3', 'db2', 'coif1', 'haar']:
            self.wavelet_type = wavelet_type
        else:
            raise ValueError(
                f"Wrong wavelet type '{wavelet_type}'. Available wavelet types are: 'db3', 'db2', 'coif1', 'haar'."
            )

        if decomposition_level in [1, 2]:
            self.decomposition_level = decomposition_level
        else:
            raise ValueError(
                f"Wrong decomposition_level' {decomposition_level}'. "
                "Decomposition level should be integer. Available decomposition levels are: 1 and 2."
            )

        if response_map in ['LL', 'HL', 'LH', 'HH']:
            self.response_map = response_map
        else:
            raise ValueError(
                f"Wrong response_map' {response_map}'. Available response_maps are: 'LL', 'HL', 'LH', 'HH'."
            )

        if isinstance(rotation_invariance, bool):
            self.rotation_invariance = rotation_invariance
        else:
            raise ValueError(f"Rotation Invariance should be True or False but '{type(rotation_invariance)}' detected.")

    def _get_kernel(self, response, decomposition_level=1):
        if response == "L":
            kernel = pywt.Wavelet(name=self.wavelet_type).filter_bank[0]
        elif response == "H":
            kernel = pywt.Wavelet(name=self.wavelet_type).filter_bank[1]
        else:
            kernel = None
        if decomposition_level == 2:
            kernel = [[e, 0] for e in kernel]
            kernel = [item for sublist in kernel for item in sublist]
        return kernel

    def _filter(self, img, x_filter, y_filter):
        filtered_img = ndi.convolve1d(img, x_filter, axis=1, mode=self.padding_type)
        filtered_img = ndi.convolve1d(filtered_img, y_filter, axis=0, mode=self.padding_type)
        return filtered_img

    def _apply_array(self, img):
        if self.decomposition_level == 1:
            x_filter = self._get_kernel(self.response_map[0])
            y_filter = self._get_kernel(self.response_map[1])
            if self.rotation_invariance:
                final_image = np.zeros(img.shape)
                for i in range(img.shape[2]):
                    for k in range(4):
                        final_image[:, :, i] += np.rot90(
                            self._filter(np.rot90(img[:, :, i], k=k, axes=(0, 1)), x_filter, y_filter), k=k, axes=(1, 0)
                        )
                filtered_img = final_image / 4
            else:
                filtered_img = np.zeros(img.shape)
                for i in range(img.shape[2]):
                    filtered_img[:, :, i] = self._filter(img[:, :, i], x_filter, y_filter)
        elif self.decomposition_level == 2:
            l_filter = self._get_kernel("L")
            x_filter = self._get_kernel(self.response_map[0], decomposition_level=2)
            y_filter = self._get_kernel(self.response_map[1], decomposition_level=2)

            final_image = np.zeros(img.shape)
            for i in range(img.shape[2]):
                for k in range(4):
                    img_level0 = np.rot90(img[:, :, i], k=k, axes=(0, 1))
                    img_level1 = self._filter(img_level0, l_filter, l_filter)
                    img_level2 = self._filter(img_level1, x_filter, y_filter)
                    final_image[:, :, i] += np.rot90(img_level2, k=k, axes=(1, 0))
            filtered_img = final_image / 4

        return filtered_img


class Wavelets3D(BaseFilter):
    """3D separable wavelet filtering for volumetric response maps.

    Response maps combine low-pass (``L``) and high-pass (``H``) wavelet
    kernels along all three axes. Rotation-invariant mode averages over axis
    permutations and flips to reduce orientation dependence.

    Parameters
    ----------
    wavelet_type : {"db3", "db2", "coif1", "haar"}
        Wavelet family used to obtain low- and high-pass filter kernels.
    padding_type : {"constant", "nearest", "wrap", "reflect"}
        Boundary handling mode used during convolution.
    response_map : {"LLL", "LLH", "LHL", "HLL", "LHH", "HHL", "HLH", "HHH"}
        Low/high-pass kernel combination for the three axes.
    decomposition_level : {1, 2}
        Wavelet decomposition level.
    rotation_invariance : bool, optional
        If true, average responses over axis permutations and flips.
    """

    def __init__(self, wavelet_type, padding_type, response_map, decomposition_level, rotation_invariance=False):
        super().__init__(
            filtering_method='Wavelets',
            wavelet_type=wavelet_type,
            padding_type=padding_type,
            response_map=response_map,
            decomposition_level=decomposition_level,
            rotation_invariance=rotation_invariance,
            dimensionality='3D',
        )

        self.dimensionality = '3D'

        if padding_type in ['constant', 'nearest', 'wrap', 'reflect']:
            self.padding_type = padding_type
        else:
            raise ValueError(
                f"Wrong padding type '{padding_type}'. "
                "Available padding types are: 'constant', 'nearest', 'wrap', and 'reflect'."
            )

        if wavelet_type in ['db3', 'db2', 'coif1', 'haar']:
            self.wavelet_type = wavelet_type
        else:
            raise ValueError(
                f"Wrong wavelet type '{wavelet_type}'. Available wavelet types are: 'db3', 'db2', 'coif1', 'haar'."
            )

        if decomposition_level in [1, 2]:
            self.decomposition_level = decomposition_level
        else:
            raise ValueError(
                f"Wrong decomposition_level' {decomposition_level}'. "
                "Decomposition level should be integer. Available decomposition levels are: 1 and 2."
            )

        if response_map in ['LLL', 'LLH', 'LHL', 'HLL', 'LHH', 'HHL', 'HLH', 'HHH']:
            self.response_map = response_map
        else:
            raise ValueError(
                f"Wrong response_map' {response_map}'. "
                "Available response_maps are: 'LLL', 'LLH', 'LHL', 'HLL', 'LHH', 'HHL', 'HLH', 'HHH'."
            )

        if isinstance(rotation_invariance, bool):
            self.rotation_invariance = rotation_invariance
        else:
            raise ValueError(f"Rotation Invariance should be True or False but '{type(rotation_invariance)}' detected.")

        self.pooling = None

    def _get_kernel(self, response, decomposition_level=1):
        if response == "L":
            kernel = pywt.Wavelet(name=self.wavelet_type).filter_bank[0]
        elif response == "H":
            kernel = pywt.Wavelet(name=self.wavelet_type).filter_bank[1]
        else:
            kernel = None
        if decomposition_level == 2:
            kernel = [[e, 0] for e in kernel]
            kernel = [item for sublist in kernel for item in sublist]
        return kernel

    def _filter(self, img, x_filter, y_filter, z_filter):
        filtered_img = ndi.convolve1d(img, x_filter, axis=1, mode=self.padding_type)
        filtered_img = ndi.convolve1d(filtered_img, y_filter, axis=0, mode=self.padding_type)
        filtered_img = ndi.convolve1d(filtered_img, z_filter, axis=2, mode=self.padding_type)
        return filtered_img

    def _apply_array(self, img):
        if self.decomposition_level == 1:
            x_filter = self._get_kernel(self.response_map[0])
            y_filter = self._get_kernel(self.response_map[1])
            z_filter = self._get_kernel(self.response_map[2])
            if self.rotation_invariance:
                final_image = np.zeros(img.shape)
                kernels_permutation = [
                    (x_filter, y_filter, z_filter),
                    (z_filter, x_filter, y_filter),
                    (y_filter, z_filter, x_filter),
                ]
                for kernels in kernels_permutation:
                    final_image += self._filter(img, kernels[0], kernels[1], kernels[2])
                    final_image += self._filter(img[::-1, :, :], kernels[0], kernels[1], kernels[2])[::-1, :, :]
                    final_image += self._filter(img[:, ::-1, :], kernels[0], kernels[1], kernels[2])[:, ::-1, :]
                    final_image += self._filter(img[:, :, ::-1], kernels[0], kernels[1], kernels[2])[:, :, ::-1]
                    final_image += self._filter(img[::-1, ::-1, :], kernels[0], kernels[1], kernels[2])[::-1, ::-1, :]
                    final_image += self._filter(img[::-1, :, ::-1], kernels[0], kernels[1], kernels[2])[::-1, :, ::-1]
                    final_image += self._filter(img[:, ::-1, ::-1], kernels[0], kernels[1], kernels[2])[:, ::-1, ::-1]
                    final_image += self._filter(img[::-1, ::-1, ::-1], kernels[0], kernels[1], kernels[2])[
                        ::-1, ::-1, ::-1
                    ]
                filtered_img = final_image / (8 * len(kernels_permutation))
            else:
                filtered_img = self._filter(img, x_filter, y_filter, z_filter)
        else:
            x_filter = self._get_kernel("L")
            y_filter = self._get_kernel("L")
            z_filter = self._get_kernel("L")
            kernels_permutation = [
                (x_filter, y_filter, z_filter),
                (z_filter, x_filter, y_filter),
                (y_filter, z_filter, x_filter),
            ]
            level1_responses = list()
            for kernels in kernels_permutation:
                level1_responses.append(self._filter(img, kernels[0], kernels[1], kernels[2]))
                level1_responses.append(self._filter(img[::-1, :, :], kernels[0], kernels[1], kernels[2])[::-1, :, :])
                level1_responses.append(self._filter(img[:, ::-1, :], kernels[0], kernels[1], kernels[2])[:, ::-1, :])
                level1_responses.append(self._filter(img[:, :, ::-1], kernels[0], kernels[1], kernels[2])[:, :, ::-1])
                level1_responses.append(
                    self._filter(img[::-1, ::-1, :], kernels[0], kernels[1], kernels[2])[::-1, ::-1, :]
                )
                level1_responses.append(
                    self._filter(img[::-1, :, ::-1], kernels[0], kernels[1], kernels[2])[::-1, :, ::-1]
                )
                level1_responses.append(
                    self._filter(img[:, ::-1, ::-1], kernels[0], kernels[1], kernels[2])[:, ::-1, ::-1]
                )
                level1_responses.append(
                    self._filter(img[::-1, ::-1, ::-1], kernels[0], kernels[1], kernels[2])[::-1, ::-1, ::-1]
                )

            x_filter = self._get_kernel(self.response_map[0], decomposition_level=2)
            y_filter = self._get_kernel(self.response_map[1], decomposition_level=2)
            z_filter = self._get_kernel(self.response_map[2], decomposition_level=2)
            final_image = np.zeros(img.shape)
            kernels_permutation = [
                (x_filter, y_filter, z_filter),
                (z_filter, x_filter, y_filter),
                (y_filter, z_filter, x_filter),
            ]
            for kernels in kernels_permutation:
                final_image += self._filter(level1_responses[0], kernels[0], kernels[1], kernels[2])
                final_image += self._filter(level1_responses[1][::-1, :, :], kernels[0], kernels[1], kernels[2])[
                    ::-1, :, :
                ]
                final_image += self._filter(level1_responses[2][:, ::-1, :], kernels[0], kernels[1], kernels[2])[
                    :, ::-1, :
                ]
                final_image += self._filter(level1_responses[3][:, :, ::-1], kernels[0], kernels[1], kernels[2])[
                    :, :, ::-1
                ]
                final_image += self._filter(level1_responses[4][::-1, ::-1, :], kernels[0], kernels[1], kernels[2])[
                    ::-1, ::-1, :
                ]
                final_image += self._filter(level1_responses[5][::-1, :, ::-1], kernels[0], kernels[1], kernels[2])[
                    ::-1, :, ::-1
                ]
                final_image += self._filter(level1_responses[6][:, ::-1, ::-1], kernels[0], kernels[1], kernels[2])[
                    :, ::-1, ::-1
                ]
                final_image += self._filter(level1_responses[7][::-1, ::-1, ::-1], kernels[0], kernels[1], kernels[2])[
                    ::-1, ::-1, ::-1
                ]
            filtered_img = final_image / (8 * len(kernels_permutation))
        return filtered_img
