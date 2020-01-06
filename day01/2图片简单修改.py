import cv2
import matplotlib.pyplot as plt
import numpy as np

# 显示图片
def show_image(img,figsize=(4,4)):
    plt.figure(figsize=figsize)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # 彩图
    plt.show()  # 显示图片

# 读取图片(read image)
img_origin1 = cv2.imread('lenna.jpg', 1)  # 0：黑白图(二维灰度图)， 1彩图（三维彩图）
# img_origin2 = cv2.imread('lenna.jpg', 0)  # 0：黑白图(二维灰度图)， 1彩图（三维彩图）

# 剪切图片(image crop)
# show_image(img_origin1[100:300][100:200])

# 通道切分（channal split）
# B,G,R = cv2.split(img_origin1)
# cv2.imshow('B',B)
# cv2.imshow('G',G)
# cv2.imshow('R',R)
# cv2.waitKeyEx()

# 像素修改
def image_color(img,b_alp=0, g_alp=0, r_alp=0):
    B,G,R = cv2.split(img)

    # b_lim = 255 - b_alp
    # B[B>b_lim] = 255
    # B[B<b_lim] = (B[B<b_lim] + b_alp).astype(img.dtype)
    # 修改B
    B = B + b_alp
    if b_alp > 0: B[B > 255] = 255
    else: B[B<0] = 0

    G = G + g_alp
    if g_alp > 0: G[G > 255] = 255
    else: G[G<0] = 0

    R = R + r_alp
    if r_alp > 0: R[R > 255] = 255
    else: R[R<0] = 0

    img_merge = cv2.merge((B, G, R))
    show_image(img_merge)
# image_color(img_origin1, b_alp=100)

# Gamma change
def adjust_gamma(img, gamma=1.0):  # 越大越亮 1 保持不变
    invGamma = 1.0/gamma
    table = []
    for i in range(256):
        table.append(((i/255)**invGamma)*255)
    table = np.array(table).astype(img.dtype)
    return cv2.LUT(img, table)
image = adjust_gamma(img_origin1, 10)
# show_image(image)

# 直方图显示数据
def histogram_equalization(img1, img2):
    # img.flatten()降为一维
    plt.subplot(121)  # 一行2列第1个
    plt.title('imge1')
    plt.hist(img1.flatten(), 256, [0,256], color='b')  # 直方图显示

    plt.subplot(122)  # 一行2列第2个
    plt.title('imge2')
    plt.hist(img2.flatten(), 256, [0,256], color='r')
    plt.show()

histogram_equalization(img_origin1, image)


# YUV色彩空间的Y进行直方图来调节图片 y:明亮调节通道
# HSV    H: 调颜色通道
def yuv():
    img_yuv = cv2.cvtColor(img_origin1, cv2.COLOR_BGR2YUV)
    print('1',img_yuv)
    # 直方图均衡
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])  # 第三个控制维度
    print('2:',img_yuv)
    show_image(cv2.cvtColor(img_yuv,cv2.COLOR_YUV2BGR))
# yuv()

# 相似/仿射/投影/变换
