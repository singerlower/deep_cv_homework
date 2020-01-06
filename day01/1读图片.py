import cv2
import matplotlib.pyplot as plt
import numpy as py

# 读取图片
img_origin1 = cv2.imread('lenna.jpg', 1)  # 0：黑白图(二维灰度图)， 1彩图（三维彩图）
img_origin2 = cv2.imread('lenna.jpg', 0)  # 0：黑白图(二维灰度图)， 1彩图（三维彩图）
print(img_origin1.shape)  # 图片大小

# 1、使用cv2显示图片
# cv2.imshow('lenna',img_origin)  # 显示图片
# key = cv2.waitKey(0)  # 等待操作输入
# if key == 27:  # 27 表示ESC表示退出
#     cv2.destroyAllWindows()  # 关闭窗体

# 2、使用plt显示图片
# plt.imshow(img_origin)  # cv2的颜色通道与plt不同

# 调整图片显示大小
# plt.figure(figsize=(2,2))  # 调整图片大小不写默认[6.4, 4.8]单位需要查

# 显示布局
# plt.subplot(121)  # 一行2列第1个
# plt.imshow(cv2.cvtColor(img_origin1, cv2.COLOR_BGR2RGB)) # 彩图

# plt.subplot(122)  # 一行2列第2个
# plt.imshow(img_origin2, cmap='gray')  # 灰度图
# plt.show()  # 显示图片




