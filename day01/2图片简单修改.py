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
# show_image(img_origin1[100:300, 100:200])

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
    if b_alp > 0: B[B > 255] = 255  # 增加
    else: B[B<0] = 0  # 减少

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
            # x数据， bins条数， range x轴范围，  颜色
    plt.show()

# histogram_equalization(img_origin1, image)


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

def perspect_transform():
    # 投影变换
    pts1 = np.float32([[0,0],[0,500],[500,0],[500,500]]) # 原始点
    pts2 = np.float32([[0,0],[0,400],[300,0],[450,550]])  # 投射点
    M = cv2.getPerspectiveTransform(pts1, pts2)  # 获取投射点
    img_wrap = cv2.warpPerspective(img_origin1, M, (550,550)) # 生成图片
    show_image(img_wrap)
# perspect_transform()

def affine():
    # 仿射  变换后，平移
    rows, cols, ch = img_origin1.shape
    pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
    pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.7, rows * 0.2], [cols * 0.1, rows * 0.9]])
    M = cv2.getAffineTransform(pts1, pts2)
    img_wrap = cv2.warpAffine(img_origin1, M, (cols, rows))
    show_image(img_wrap)
# affine()


img_origin2 = cv2.imread('lenna.jpg', 0)  # 0：黑白图(二维灰度图)， 1彩图（三维彩图）
# show_image(img_origin2)
def dilate():
    # 膨胀  更白 -》255
    image = cv2.dilate(img_origin2, None, iterations=1)
    show_image(image)
# dilate()

def erode():
    # 腐蚀  更黑 -》 0
    image = cv2.erode(img_origin2, None, iterations=1)
    show_image(image)
# erode()

# 旋转
def rot():
    image = cv2.rotate(img_origin1, rotateCode = cv2.ROTATE_90_CLOCKWISE)
    show_image(image)
rot()

