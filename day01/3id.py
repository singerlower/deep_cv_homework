import cv2
import matplotlib.pyplot as plt
import numpy as py


def show_image(img,figsize=(4,4)):
    plt.figure(figsize=figsize)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # 彩图
    plt.show()  # 显示图片

# 读取图片
img_origin1 = cv2.imread('id.jpg', 1)  # 0：黑白图(二维灰度图)
# show_image(img_origin1)  # 图片大小
# print(img_origin1)

# plt.hist(img_origin1[:,:,0].flatten(), 255, [0, 255], color='b')
# plt.hist(img_origin1[:,:,1].flatten(), 255, [0, 255], color='g')
# plt.hist(img_origin1[:,:,2].flatten(), 255, [0, 255], color='r')
# plt.grid()
# plt.show()

# B, G, R = cv2.split(img_origin1)  # 按通道切分
# B[B>100] = 255
# G[G>100] = 255
# R[R>110] = 255
# image = cv2.merge([B, G, R])
# show_image(image)

# 切分， 变换 ， 去底色
def hsv():
    # HSV即Hue(色调),Saturation(饱和度)和Value(亮度)三个channel
    # HSV色彩空间范围为： H：0-180  S: 0-255   V： 0-255
    img_hsv = cv2.cvtColor(img_origin1, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(img_hsv)
    # 数据分布
    # plt.hist(H.flatten(), 180, [0, 255], color='b')
    # plt.hist(S.flatten(), 255, [0, 255], color='g')
    # plt.hist(V.flatten(), 255, [0, 255], color='r')
    # plt.grid()
    # plt.show()
    H[H<90] = 100
    # S[S>10] = 255
    V[V>100] = 255
    image = cv2.merge((H,S,V))
    plt.figure(figsize=(4,4))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_HSV2RGB))  # 彩图
    plt.show()  # 显示图片
hsv()