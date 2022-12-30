# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 18:04:48 2022

@author: devad
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
image = cv2.imread('lena.png')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_gray_flat=image_gray.flatten()
image_gray1 = np.zeros_like(image_gray)
k = min(image_gray_flat)
l = max(image_gray_flat)
L=2
for i in range(image_gray.shape[0]):
	for j in range(image_gray.shape[1]):
		image_gray1[i,j] = (image_gray[i,j] - k)
image_gray_flat1 = image_gray1.flatten()
image_gray2 = ((L-1)*image_gray1)/max(image_gray_flat1)
def DFT2(f):
    row_fft = np.zeros_like(f, dtype="complex_")
    col_fft = np.zeros_like(f, dtype="complex_")
    for x in range(f.shape[0]):
        row_fft[x, 0:f.shape[1]] = np.fft.fft(f[x, 0:f.shape[1]])
    for y in range(row_fft.shape[1]):
        col_fft[0: row_fft.shape[0], y] = np.fft.fft(row_fft[0: row_fft.shape[0], y])
    return col_fft

def IDFT2(g):
    col_ifft = np.zeros_like(g, dtype="complex_")
    row_ifft = np.zeros_like(g, dtype="complex_")
    fftShift_b = np.fft.ifftshift(g)
    for y in range(fftShift_b.shape[1]):
        col_ifft[0: fftShift_b.shape[0], y] = np.fft.ifft(fftShift_b[0: fftShift_b.shape[0], y])
    for x in range(col_ifft.shape[0]):
        row_ifft[x, 0: col_ifft.shape[1]] = np.fft.ifft(col_ifft[x, 0: col_ifft.shape[1]])	
    return row_ifft


F = DFT2(image_gray2)
fftShift_a = np.fft.fftshift(F)
magnitude = np.log(1 + np.abs(fftShift_a))
phase = np.angle(fftShift_a)
cv2.imshow('Input Image',image_gray2)
(fig1, ax) = plt.subplots(1, 2, )
ax[0].imshow(magnitude, cmap="gray")
ax[0].set_title("Magnitude Spectrum")
ax[1].imshow(phase, cmap="gray")
ax[1].set_title("Phase Spectrum")
plt.show()

G = IDFT2(fftShift_a);
difference_image = image_gray2 - G.real
cv2.imshow('Final Image',G.real)
cv2.imshow('Difference Image', difference_image)
cv2.waitKey(0)
cv2.destroyAllWindows()