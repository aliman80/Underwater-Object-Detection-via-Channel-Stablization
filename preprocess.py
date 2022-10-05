# بسم الله الرحمن الرحيم و به نستعين

import numpy as np
import math
import cv2
from skimage.color import rgb2lab, lab2rgb

from Augmentations import hist_augment


def img_clip(img):
    return np.clip(img, 0, 255)

def equalize_brightness(src: np.ndarray):
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    v = cv2.equalizeHist(v)

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def equalize_brightness2(src: np.ndarray):
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    v = cv2.equalizeHist(v)
    s = cv2.equalizeHist(s)

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def stabilize_channel(src: np.ndarray):
    img = src.copy()
    means = img.mean((0, 1))
    g_mean = means.mean()
    g_mean /= means
    img[..., 0], img[..., 1], img[..., 2] = img[..., 0] * g_mean[0], img[..., 1] * g_mean[1], img[..., 2] * g_mean[2]
    img = np.clip(img, 0, 255)
    return img


def global_stretching(img_L, height, width):
    length = height * width
    array = (np.copy(img_L)).flatten()
    array.sort()
    I_min = int(array[int(length / 100)])
    I_max = int(array[-int(length / 100)])
    array_Global_histogram_stretching_L = np.empty((height, width))
    for i in range(0, height):
        for j in range(0, width):
            if img_L[i][j] < I_min:
                array_Global_histogram_stretching_L[i][j] = 0
            elif img_L[i][j] > I_max:
                array_Global_histogram_stretching_L[i][j] = 100
            else:
                p_out = int((img_L[i][j] - I_min) * (100 / (I_max - I_min)))
                array_Global_histogram_stretching_L[i][j] = p_out
    return array_Global_histogram_stretching_L


def global_Stretching_ab(a, height, width):
    array_Global_histogram_stretching_L = np.empty((height, width))
    for i in range(0, height):
        for j in range(0, width):
            p_out = a[i][j] * (1.3 ** (1 - math.fabs(a[i][j] / 128)))
            array_Global_histogram_stretching_L[i][j] = p_out
    return array_Global_histogram_stretching_L


def LABStretching(sceneRadiance):
    height = len(sceneRadiance)
    width = len(sceneRadiance[0])
    img_lab = rgb2lab(sceneRadiance)
    L, a, b = cv2.split(img_lab)

    img_L_stretching = global_stretching(L, height, width)
    img_a_stretching = global_Stretching_ab(a, height, width)
    img_b_stretching = global_Stretching_ab(b, height, width)

    labArray = np.empty((height, width, 3))
    labArray[:, :, 0] = img_L_stretching
    labArray[:, :, 1] = img_a_stretching
    labArray[:, :, 2] = img_b_stretching
    img_rgb = lab2rgb(labArray) * 255
    img_rgb = np.clip(img_rgb, 0, 255)
    return np.uint8(img_rgb)


def process1(img):
    # Blurring:
    # img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.medianBlur(img, 3)

    # Sharpening:
    img = cv2.filter2D(img, -1, np.array([[-1, -1, -1],
                                          [-1, 8.5, -1],
                                          [-1, -1, -1]]))
    img = np.clip(img, 0, 255)

    # Histogram Equalization:
    r = cv2.equalizeHist(img[..., 0])
    g = cv2.equalizeHist(img[..., 1])
    b = cv2.equalizeHist(img[..., 2])
    img = np.moveaxis(np.stack((r, g, b)), 0, 2)

    # Cleansing/Enhancing:
    img = LABStretching(stabilize_channel(img))
    return img


def process2(img):
    # Blurring:
    # img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.medianBlur(img, 3)

    # Sharpening:
    img = cv2.filter2D(img, -1, np.array([[-1, -1, -1],
                                          [-1, 8.5, -1],
                                          [-1, -1, -1]]))
    img = np.clip(img, 0, 255)

    # Histogram Equalization:
    r = cv2.equalizeHist(img[..., 0])
    g = cv2.equalizeHist(img[..., 1])
    b = cv2.equalizeHist(img[..., 2])
    img = np.moveaxis(np.stack((r, g, b)), 0, 2)

    # Cleansing/Enhancing:
    img = equalize_brightness(stabilize_channel(img))
    return img


def process3(img):
    # Blurring:
    # img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.medianBlur(img, 3)

    # Sharpening:
    img = cv2.filter2D(img, -1, np.array([[-1, -1, -1],
                                          [-1, 8.5, -1],
                                          [-1, -1, -1]]))
    img = np.clip(img, 0, 255)

    # Cleansing/Enhancing:
    img = equalize_brightness(stabilize_channel(img))
    return img


def process4(img):

    # Cleansing/Enhancing:
    img = equalize_brightness2(stabilize_channel(img))
    img = np.clip(img, 0, 255)
    return img
def stabilize_channel(src: np.ndarray):
    img = src.copy()
    means = img.mean((0, 1))
    g_mean = means.mean()
    g_mean /= means
    img[..., 0], img[..., 1], img[..., 2] = img_clip(img[..., 0] * g_mean[0]), img_clip(img[..., 1] * g_mean[1]), img_clip(img[..., 2] * g_mean[2])
    return img
def broken_mir(src: np.ndarray, rand=False):
    img = np.moveaxis(src.copy(), 2, 0)  # Move columns to last dimension
    c = img.shape[-1]
    if rand:
        mid_p = np.random.choice(c)
    else:
        mid_p = c // 2
    lp = img[..., :mid_p]
    lp = lp[..., ::-1]
    rp = img[..., mid_p:]
    rp = rp[..., ::-1]
    img[..., :mid_p] = lp
    img[..., mid_p:] = rp
    return np.moveaxis(img, 0, 2)


def process5(img):
    #img=global_Stretching_ab
    # Blurring:
    #img = cv2.GaussianBlur(img, (3, 3), 0)
   # img = cv2.medianBlur(img, 3)
    img=broken_mir(img)
    # Sharpening:
    img = cv2.filter2D(img, -1, np.array([[-1, -1, -1],
                                          [-1, 8.5, -1],
                                          [-1, -1, -1]]))
    img = np.clip(img, 0, 255)
   

    #Histogram Equalization:
    r = cv2.equalizeHist(img[..., 0])
    g = cv2.equalizeHist(img[..., 1])
    b = cv2.equalizeHist(img[..., 2])
    img = np.moveaxis(np.stack((r, g, b)), 0, 2)

    # Cleansing/Enhancing:
    img = stabilize_channel(img)
    img = cv2.filter2D(img, -1, np.array([[-1, -1, -1],
                                          [-1, 8.5, -1],
                                          [-1, -1, -1]]))
    img = np.clip(img, 0, 255)
    return img
def process6(img):
    #img=global_Stretching_ab
    # Blurring:
    #img = cv2.GaussianBlur(img, (3, 3), 0)
   # img = cv2.medianBlur(img, 3)
    #img=broken_mir(img)
    # Sharpening:
    #img = cv2.filter2D(img, -1, np.array([[-1, -1, -1],
                                         # [-1, 8.5, -1],
                                         # [-1, -1, -1]]))
   # img = np.clip(img, 0, 255)
   

    #Histogram Equalization:
    #r = cv2.equalizeHist(img[..., 0])
    #g = cv2.equalizeHist(img[..., 1])
    #b = cv2.equalizeHist(img[..., 2])
    #img = np.moveaxis(np.stack((r, g, b)), 0, 2)

    # Cleansing/Enhancing:
    img = stabilize_channel(img)
    
    img = np.clip(img, 0, 255)
    # Sharpening:
    img = cv2.filter2D(img, -1, np.array([[-1, -1, -1],
                                          [-1, 9, -1],
                                          [-1, -1, -1]]))
    img = np.clip(img, 0, 255)
    return img
def process8(img):
    #img=global_Stretching_ab
    # Blurring:
    #img = cv2.GaussianBlur(img, (3, 3), 0)
   # img = cv2.medianBlur(img, 3)
    #img=broken_mir(img)
    # Sharpening:
    #img = cv2.filter2D(img, -1, np.array([[-1, -1, -1],
                                         # [-1, 8.5, -1],
                                         # [-1, -1, -1]]))
   # img = np.clip(img, 0, 255)
   

    #Histogram Equalization:
    #r = cv2.equalizeHist(img[..., 0])
    #g = cv2.equalizeHist(img[..., 1])
    #b = cv2.equalizeHist(img[..., 2])
    #img = np.moveaxis(np.stack((r, g, b)), 0, 2)

    # Cleansing/Enhancing:
    img = cv2.filter2D(img, -1, np.array([[-1, -1, -1],
                                          [-1, 9, -1],
                                          [-1, -1, -1]]))
    img = np.clip(img, 0, 255)
    img = stabilize_channel(img)
    
    img = np.clip(img, 0, 255)
   
    return img
def process9(img):
    #img=global_Stretching_ab
    # Blurring:
    #img = cv2.GaussianBlur(img, (3, 3), 0)
   # img = cv2.medianBlur(img, 3)
    #img=broken_mir(img)
    # Sharpening:
    #img = cv2.filter2D(img, -1, np.array([[-1, -1, -1],
                                         # [-1, 8.5, -1],
                                         # [-1, -1, -1]]))
   # img = np.clip(img, 0, 255)
   

    #Histogram Equalization:
    #r = cv2.equalizeHist(img[..., 0])
    #g = cv2.equalizeHist(img[..., 1])
    #b = cv2.equalizeHist(img[..., 2])
    #img = np.moveaxis(np.stack((r, g, b)), 0, 2)
   
    # Cleansing/Enhancing:
    #img = cv2.filter2D(img, -1, np.array([[-1, -1, -1],
                                          #[-1, 9, -1],
                                          #[-1, -1, -1]]))
    #img = np.clip(img, 0, 255)
    img = stabilize_channel(img)

    
    img = np.clip(img, 0, 255)
    img =hist_augment(img)
   
    return img
def process10(img):
    #img=global_Stretching_ab
    # Blurring:
    #img = cv2.GaussianBlur(img, (3, 3), 0)
   # img = cv2.medianBlur(img, 3)
    #img=broken_mir(img)
     #Sharpening:
    img = cv2.filter2D(img, -1, np.array([[-1, -1, -1],
                                         [-1, 8.5, -1],
                                         [-1, -1, -1]]))
    img = np.clip(img, 0, 255)
   

    #Histogram Equalization:
    #r = cv2.equalizeHist(img[..., 0])
    #g = cv2.equalizeHist(img[..., 1])
    #b = cv2.equalizeHist(img[..., 2])
    #img = np.moveaxis(np.stack((r, g, b)), 0, 2)
   
    # Cleansing/Enhancing:
    #img = cv2.filter2D(img, -1, np.array([[-1, -1, -1],
                                          #[-1, 9, -1],
                                          #[-1, -1, -1]]))
    #img = np.clip(img, 0, 255)
    img = stabilize_channel(img)

    
    img = np.clip(img, 0, 255)
    img =hist_augment(img)
   
    return img
def process12(img):
    #img=global_Stretching_ab
    # Blurring:
    #img = cv2.GaussianBlur(img, (3, 3), 0)
   # img = cv2.medianBlur(img, 3)
    #img=broken_mir(img)
     #Sharpening:
    #img = cv2.filter2D(img, -1, np.array([[-1, -1, -1],
                                        # [-1, 8.5, -1],
                                        # [-1, -1, -1]]))
    #img = np.clip(img, 0, 255)
   

    #Histogram Equalization:
    #r = cv2.equalizeHist(img[..., 0])
    #g = cv2.equalizeHist(img[..., 1])
    #b = cv2.equalizeHist(img[..., 2])
    #img = np.moveaxis(np.stack((r, g, b)), 0, 2)
   
    # Cleansing/Enhancing:
    #img = cv2.filter2D(img, -1, np.array([[-1, -1, -1],
                                          #[-1, 9, -1],
                                          #[-1, -1, -1]]))
    #img = np.clip(img, 0, 255)
   
    img = stabilize_channel(img)

    
    img = np.clip(img, 0, 255)
    img =hist_augment(img)
   
    return img

def process13(img):
    #img=global_Stretching_ab
    # Blurring:
    #img = cv2.GaussianBlur(img, (3, 3), 0)
   # img = cv2.medianBlur(img, 3)
    #img=broken_mir(img)
     #Sharpening:
    img =hist_augment(img)
    img = cv2.filter2D(img, -1, np.array([[-1, -1, -1],
                                         [-1, 9, -1],
                                        [-1, -1, -1]]))
    img = np.clip(img, 0, 255)
   

    #Histogram Equalization:
    #r = cv2.equalizeHist(img[..., 0])
    #g = cv2.equalizeHist(img[..., 1])
    #b = cv2.equalizeHist(img[..., 2])
    #img = np.moveaxis(np.stack((r, g, b)), 0, 2)
   
    # Cleansing/Enhancing:
    #img = cv2.filter2D(img, -1, np.array([[-1, -1, -1],
                                          #[-1, 9, -1],
                                          #[-1, -1, -1]]))
    #img = np.clip(img, 0, 255)
    
    img = stabilize_channel(img)

    
    img = np.clip(img, 0, 255)
    
   
    return img
    

    