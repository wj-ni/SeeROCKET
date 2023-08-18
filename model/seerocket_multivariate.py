import random
from itertools import combinations

from numba import njit, prange, vectorize
import numpy as np
from scipy.special import comb


def generate_kernels(kernel_size, num_ones, num_kernels):
    kernels = []
    for ones in combinations(range(kernel_size), num_ones):
        for minus_ones in combinations(range(kernel_size), num_ones):
            if set(ones).isdisjoint(set(minus_ones)):
                kernel = [0] * kernel_size
                for i in ones:
                    kernel[i] = 1
                for i in minus_ones:
                    kernel[i] = -1
                if kernel not in kernels and [-x for x in kernel] not in kernels:
                    kernels.append(kernel)
    if len(kernels) == num_kernels:
        selected_kernels = kernels
    else:
        selected_kernels = random.sample(kernels, num_kernels)
    return np.array(selected_kernels).astype(np.int32)


def get_num_kernels(kernel_size, num_ones):
    return int(comb(kernel_size, num_ones) * comb(kernel_size - num_ones, num_ones) // 2)


@njit("float32[:](float32[:],int32[:],int32,int32,int32)")
def convolve(X, kernel, dilation, initial_stride, stride_increment):
    kernel_size = len(kernel)
    temp_output = []
    current_stride = initial_stride
    i = 0
    while i < len(X):
        result = 0
        for j in range(kernel_size):
            if kernel[j] == 1:
                result += X[i + j * dilation]
            elif kernel[j] == -1:
                result -= X[i + j * dilation]
        temp_output.append(result)
        i += current_stride
        current_stride += stride_increment
        if i + (kernel_size - 1) * dilation >= len(X):
            break
    return np.array(temp_output, dtype=np.float32)


@njit(
    "float32[:](float32[:,:,:],int32[:],int32[:],int32[:],int32[:],float32[:],int32,int32,int32,int32[:,:],int32,int32,int32,int32)",
    fastmath=True, parallel=False, cache=True)
def _fit_biases(X, num_channels_per_combination, channel_indices, dilations, num_features_per_dilation,
                quantiles, num_kernels, initial_stride, stride_increment, kernels, LPPV_size, padding_left,
                padding_right, padding_value):
    num_examples, num_channels, input_length = X.shape

    num_dilations = len(dilations)

    num_features = num_kernels * np.sum(num_features_per_dilation) * LPPV_size

    biases = np.zeros(num_features, dtype=np.float32)

    feature_index_start = 0
    quantiles_index_start = 0

    combination_index = 0
    num_channels_start = 0

    for dilation_index in range(num_dilations):

        dilation = dilations[dilation_index]

        num_features_this_dilation = num_features_per_dilation[dilation_index]

        for kernel in kernels:

            feature_index_end = feature_index_start + num_features_this_dilation * LPPV_size

            quantiles_index_end = quantiles_index_start + num_features_this_dilation

            num_channels_this_combination = num_channels_per_combination[combination_index]

            num_channels_end = num_channels_start + num_channels_this_combination

            channels_this_combination = channel_indices[num_channels_start:num_channels_end]

            _X = X[np.random.randint(num_examples)][channels_this_combination]

            final_output = []
            for row_X in _X:
                row_X = np.concatenate((np.full(padding_left, padding_value, dtype=np.float32), row_X,
                                        np.full(padding_right, padding_value, dtype=np.float32)))
                temp_output = convolve(row_X, kernel, dilation, initial_stride, stride_increment)
                final_output.extend(temp_output)

            if len(final_output) < LPPV_size:
                while len(final_output) < LPPV_size:
                    final_output.append(0)

            length_per_LPPV = len(final_output) // LPPV_size
            for LPPV_index in range(LPPV_size):
                start = LPPV_index * length_per_LPPV
                if LPPV_index < LPPV_size - 1:
                    end = start + length_per_LPPV
                else:
                    end = len(final_output)
                segment = final_output[start:end]
                biases[feature_index_start + LPPV_index * num_features_this_dilation:feature_index_start + (
                        LPPV_index + 1) * num_features_this_dilation] = np.quantile(segment, quantiles[
                                                                                             quantiles_index_start:quantiles_index_end])

            feature_index_start = feature_index_end
            quantiles_index_start = quantiles_index_end
            combination_index += 1
            num_channels_start = num_channels_end

    return biases


def _fit_dilations(input_length, num_features, max_dilations_per_kernel, num_kernels, kernel_size):
    num_features_per_kernel = num_features // num_kernels
    true_max_dilations_per_kernel = min(num_features_per_kernel, max_dilations_per_kernel)
    multiplier = num_features_per_kernel / true_max_dilations_per_kernel

    max_exponent = np.log2((input_length - 1) / (kernel_size - 1))
    dilations, num_features_per_dilation = \
        np.unique(np.logspace(0, max_exponent, true_max_dilations_per_kernel, base=2).astype(np.int32),
                  return_counts=True)
    num_features_per_dilation = (num_features_per_dilation * multiplier).astype(np.int32)

    remainder = num_features_per_kernel - np.sum(num_features_per_dilation)
    i = 0
    while remainder > 0:
        num_features_per_dilation[i] += 1
        remainder -= 1
        i = (i + 1) % len(num_features_per_dilation)

    return dilations, num_features_per_dilation


def _fit_dilations_custom(input_length, dilations, num_features, max_dilations_per_kernel, num_kernels, kernel_size):
    max_dilation_value = (input_length - 1) // (kernel_size - 1)

    dilations = np.array(dilations)
    dilations = dilations[dilations <= max_dilation_value]

    if len(dilations) > max_dilations_per_kernel:
        dilations = dilations[:max_dilations_per_kernel]

    num_features_per_kernel = num_features // num_kernels
    true_max_dilations_per_kernel = min(num_features_per_kernel, len(dilations))

    dilations = dilations[:true_max_dilations_per_kernel]

    num_features_per_dilation = np.full(true_max_dilations_per_kernel,
                                        num_features_per_kernel // true_max_dilations_per_kernel)

    remainder = num_features_per_kernel - np.sum(num_features_per_dilation)

    i = 0
    while remainder > 0:
        num_features_per_dilation[i] += 1
        remainder -= 1
        i = (i + 1) % len(num_features_per_dilation)

    return dilations, num_features_per_dilation


def _quantiles(n):
    return np.array([(_ * ((np.sqrt(5) + 1) / 2)) % 1 for _ in range(1, n + 1)], dtype=np.float32)


def fit(X, num_features=10_000, max_dilations_per_kernel=32, kernel_size=9, num_ones=1, dilations=None,
        max_allow_num_channels=9,
        initial_stride=1, stride_increment=0, LPPV_size=3, padding_left=0, padding_right=0, padding_value=0,
        num_kernels=None):

    _, num_channels, input_length = X.shape

    num_features = num_features // LPPV_size

    input_length = input_length + padding_left + padding_right

    max_num_kernels = get_num_kernels(kernel_size, num_ones)

    if num_kernels is None or num_kernels < 1:
        num_kernels = max_num_kernels

    if num_kernels is not None and num_kernels > max_num_kernels:
        num_kernels = max_num_kernels

    kernels = generate_kernels(kernel_size, num_ones, num_kernels)

    if dilations is None:
        dilations, num_features_per_dilation = _fit_dilations(input_length, num_features, max_dilations_per_kernel,
                                                              num_kernels, kernel_size)
    else:
        dilations, num_features_per_dilation = _fit_dilations_custom(input_length, dilations, num_features,
                                                                     max_dilations_per_kernel, num_kernels, kernel_size)
    num_features_per_kernel = np.sum(num_features_per_dilation)

    quantiles = _quantiles(num_kernels * num_features_per_kernel)

    num_dilations = len(dilations)
    num_combinations = num_kernels * num_dilations

    max_num_channels = min(num_channels, max_allow_num_channels)
    max_exponent = np.log2(max_num_channels + 1)

    num_channels_per_combination = (2 ** np.random.uniform(0, max_exponent, num_combinations)).astype(np.int32)

    channel_indices = np.zeros(num_channels_per_combination.sum(), dtype=np.int32)

    num_channels_start = 0
    for combination_index in range(num_combinations):
        num_channels_this_combination = num_channels_per_combination[combination_index]
        num_channels_end = num_channels_start + num_channels_this_combination
        channel_indices[num_channels_start:num_channels_end] = np.random.choice(num_channels,
                                                                                num_channels_this_combination,
                                                                                replace=False)

        num_channels_start = num_channels_end

    biases = _fit_biases(X, num_channels_per_combination, channel_indices, dilations, num_features_per_dilation,
                         quantiles, num_kernels, initial_stride, stride_increment, kernels, LPPV_size, padding_left,
                         padding_right, padding_value)

    return num_channels_per_combination, channel_indices, dilations, num_features_per_dilation, biases, num_kernels, LPPV_size, \
        kernels, initial_stride, stride_increment, padding_left, padding_right, padding_value


@vectorize("float32(float32,float32)", nopython=True, cache=True)
def _PPV(a, b):
    if a > b:
        return 1
    else:
        return 0


@njit(
    "float32[:,:](float32[:,:,:],Tuple((int32[:],int32[:],int32[:],int32[:],float32[:],int32,int32,int32[:,:],int32,int32,int32,int32,int32)))",
    fastmath=True, parallel=True, cache=True)
def transform(X, parameters):
    num_examples, num_channels, input_length = X.shape

    num_channels_per_combination, channel_indices, dilations, num_features_per_dilation, biases, num_kernels, LPPV_size, \
        kernels, initial_stride, stride_increment, padding_left, padding_right, padding_value = parameters

    num_dilations = len(dilations)

    num_features = num_kernels * np.sum(num_features_per_dilation)

    features = np.zeros((num_examples, LPPV_size * num_features), dtype=np.float32)

    for example_index in prange(num_examples):

        _X = X[example_index]

        feature_index_start = 0

        combination_index = 0
        num_channels_start = 0

        for dilation_index in range(num_dilations):

            dilation = dilations[dilation_index]

            num_features_this_dilation = num_features_per_dilation[dilation_index]

            for kernel in kernels:

                feature_index_end = feature_index_start + num_features_this_dilation * LPPV_size

                num_channels_this_combination = num_channels_per_combination[combination_index]

                num_channels_end = num_channels_start + num_channels_this_combination

                channels_this_combination = channel_indices[num_channels_start:num_channels_end]

                rows_X = _X[channels_this_combination]

                final_output = []
                for row_X in rows_X:
                    row_X = np.concatenate((np.full(padding_left, padding_value, dtype=np.float32), row_X,
                                            np.full(padding_right, padding_value, dtype=np.float32)))
                    temp_output = convolve(row_X, kernel, dilation, initial_stride, stride_increment)
                    final_output.extend(temp_output)

                if len(final_output) < LPPV_size:
                    while len(final_output) < LPPV_size:
                        final_output.append(0)

                segment_length = len(final_output) // LPPV_size
                for LPPV_index in range(LPPV_size):
                    start = LPPV_index * segment_length
                    end = start + segment_length if LPPV_index < LPPV_size - 1 else len(final_output)
                    temp_segment = final_output[start:end]
                    for feature_count in range(num_features_this_dilation):
                        features[
                            example_index, feature_index_start + LPPV_index * num_features_this_dilation + feature_count] = _PPV(
                            np.array(temp_segment, dtype=np.float32),
                            biases[
                                feature_index_start + LPPV_index * num_features_this_dilation + feature_count]).mean()

                feature_index_start = feature_index_end

                combination_index += 1
                num_channels_start = num_channels_end

    return features


def fit_transform(X, num_features=10_000, max_dilations_per_kernel=32, kernel_size=9, num_ones=1, dilations=None,max_allow_num_channels=9,
        initial_stride=1, stride_increment=0, LPPV_size=3, padding_left=0, padding_right=0, padding_value=0, num_kernels=None):
    params = fit(X, num_features=num_features, max_dilations_per_kernel=max_dilations_per_kernel, kernel_size=kernel_size, num_ones=num_ones,
                 dilations=dilations, initial_stride=initial_stride, stride_increment=stride_increment, LPPV_size=LPPV_size, padding_left=padding_left,
                 padding_right=padding_right, padding_value=padding_value, num_kernels=num_kernels, max_allow_num_channels=max_allow_num_channels)
    features = transform(X, params)
    return features
