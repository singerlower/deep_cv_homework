import cv2
import matplotlib.pyplot as plt
import numpy as py


def show_image(img,figsize=(4,4)):
    plt.figure(figsize=figsize)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # 彩图
    plt.show()  # 显示图片

# 读取图片
img_origin1 = cv2.imread('id.jpg', 1)  # 0：黑白图(二维灰度图)
show_image(img_origin1)  # 图片大小
# print(img_origin1)

# plt.hist(img_origin1[:,:,0].flatten(), 255, [0, 255], color='b')
# plt.hist(img_origin1[:,:,1].flatten(), 255, [0, 255], color='g')
# plt.hist(img_origin1[:,:,2].flatten(), 255, [0, 255], color='r')
# plt.grid()
# plt.show()

B, G, R = cv2.split(img_origin1)  # 按通道切分
B[B>100] = 255
G[G>100] = 255
R[R>110] = 255
image = cv2.merge([B, G, R])
show_image(image)
