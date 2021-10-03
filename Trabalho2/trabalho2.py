'''
Processamento de Imagens - 2021.2
Trabalho 2
Guilherme Pereira Corrêa 198397
Outubro, 2021
'''

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import math
import cv2 as cv

# Called if the user inserts a wrong mode name
def wrong_mode(img, extra_arguments=None):
    print('Inserted mode does not exist')
    return img

# Function that takes an fft representation of a image and calculates it's magnitude, so it can be interpreted
def magnitude(img):
    return np.log(np.abs(img)).clip(0,255).astype(np.uint8)

# From a shifted fft representation, gets back the image
def fft_inv(img, extra_arguments=None):
    return np.abs(np.fft.ifft2(np.fft.ifftshift(img)))

# Calculates de FFT of the image and returns its interpretable version
def fft_transform_mag(img, extra_arguments=None):
    return magnitude(np.fft.fftshift(np.fft.fft2(img)))

# Calculates de FFT of the image, shifted to the center
def fft_transform(img, extra_arguments=None):
    return np.fft.fftshift(np.fft.fft2(img))

# Creates a tensor full of zeros but inside the circle in its center, for masking
def create_inner_circle(shape, radius):
    mask = np.zeros(shape)
    cy = mask.shape[0] // 2
    cx = mask.shape[1] // 2
    cv.circle(mask, (cx,cy), radius, (255,255,255), -1)[0]
    return mask

# Apply a low pass filter on the image using FFT representation
def low_pass_filter(img, radius):
    mask = create_inner_circle(img.shape, int(radius))
    fft_shifted = fft_transform(img)
    fft_shifted_masked = np.multiply(fft_shifted, mask) / 255
    img_filtered = fft_inv(fft_shifted_masked)
    return img_filtered

# The outter circle is created, for masking, by taking the complementary of the inner circle
def create_outter_circle(shape, radius):
    mask = np.zeros(shape)
    cy = mask.shape[0] // 2
    cx = mask.shape[1] // 2
    cv.circle(mask, (cx,cy), radius, (255,255,255), -1)[0]
    return 255 - mask

# Apply a high pass filter on the image using FFT representation
def high_pass_filter(img, radius):
    mask = create_outter_circle(img.shape, int(radius))
    fft_shifted = fft_transform(img)
    fft_shifted_masked = np.multiply(fft_shifted, mask) / 255
    img_filtered = fft_inv(fft_shifted_masked)
    return img_filtered

# The idea is to subtract two circles from each other
def create_disk(shape, outter_radius, inner_radius):
    assert outter_radius > inner_radius
    outter_mask = np.zeros(shape)
    inner_mask = np.zeros(shape)
    cy = outter_mask.shape[0] // 2
    cx = outter_mask.shape[1] // 2
    cv.circle(outter_mask, (cx,cy), outter_radius, (255,255,255), -1)[0]
    cv.circle(inner_mask, (cx,cy), inner_radius, (255,255,255), -1)[0]
    return outter_mask - inner_mask

# Apply a band pass filter on the image using FFT representation
def band_pass_filter(img, outter_radius, inner_radius):
    mask = create_disk(img.shape, int(outter_radius), int(inner_radius))
    fft_shifted = fft_transform(img)
    dft_shift_masked = np.multiply(fft_shifted, mask) / 255
    img_filtered = fft_inv(dft_shift_masked)
    return img_filtered

# Creates the inverse of a disk, for masking
def create_inverse_disk(shape, outter_radius, inner_radius):
    assert outter_radius > inner_radius
    aux_disk = create_disk(shape, outter_radius, inner_radius)
    mask = np.full(shape, 255)
    return mask - aux_disk

# Apply a band pass filter on the image using FFT representation
def band_reject_filter(img, outter_radius, inner_radius):
    mask = create_inverse_disk(img.shape, int(outter_radius), int(inner_radius))
    fft_shifted = fft_transform(img)
    dft_shift_masked = np.multiply(fft_shifted, mask) / 255
    img_filtered = fft_inv(dft_shift_masked)
    return img_filtered

# Compress the image using its FFT representation, the compression depends on the parameter keep, that
# tells the percentage of the biggest magnitudes that are going to be held
def compress(img, keep):
    keep = float(keep)
    assert keep > 0 and keep < 1

    fft_img = np.fft.fft2(img)
    # Sort the magnitudes
    fft_sort = np.sort(magnitude(fft_img).reshape(-1))
    # Find the threshold based on the percentage given by the parameter keep
    threshold = fft_sort[int(np.floor((1-keep)*len(fft_sort)))]

    # Use the threshold as a mask
    fft_img[magnitude(fft_img) < threshold] = 0

    # Recover the image
    img_filtered = np.abs(np.fft.ifft2(fft_img))

    return img_filtered

def operate_image(mode, input_img_path, extra_arguments, output_img_path = 'output.png'):
    # Dictionary that selects which function will be called depending on the inserted mode
    operations = {
    'fft-transform': fft_transform_mag,
    'lowpass-filter': low_pass_filter,
    'highpass-filter': high_pass_filter,
    'bandpass-filter': band_pass_filter,
    'bandreject-filter': band_reject_filter,
    'compress': compress
    }
    input_img = np.asarray(Image.open(input_img_path))
    output_img = operations.get(mode, wrong_mode)(input_img, *extra_arguments)

    plt.axis('off')
    plt.imshow(output_img, cmap='gray')
    plt.savefig(output_img_path)
    plt.show()

if __name__ == '__main__':
    arg_count = 2
    input_img_path = sys.argv[1]
    output_img_path = 'output.png'
    if sys.argv[2] == '-o':
        # has output file name
        arg_count += 1
        output_img_path = sys.argv[3]
        arg_count += 1
        mode = sys.argv[4]
        arg_count += 1
        print(f'out é {output_img_path}')
    else:
        mode = sys.argv[2]
        arg_count += 1
    if len(sys.argv) > arg_count:
        # Has a output_path input
        extra_arguments = sys.argv[arg_count:]
        operate_image(mode, input_img_path, extra_arguments, output_img_path)
    else:
        extra_arguments = []
        operate_image(mode, input_img_path, extra_arguments, output_img_path)
