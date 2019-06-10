import numpy as np
import cv2
import matplotlib.pyplot as plt

# 采用投影法对文本图像进行分割
min_pixel = 1 # 字符最小像素
min_width = 5 # 字符最小宽度


# 显示图片
def view_image(img, name):
    cv2.namedWindow(name)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 垂直投影，获取像素在水平方向上的像素分布
# img: 二值化图像,，黑底白字
# return: [[width, begin, end]{,}] 返回字符所在行的起始位置和终止位置
def vertical(img):
    h, w = img.shape
    chaLine = []

    begin, end = 0, 0
    for line in range(h):
        nonzeros = np.sum(img[line, :] > 0)
        # print(nonzeros)
        if nonzeros > min_pixel and begin == 0: # 记录字符起始行
            begin = line
        elif nonzeros > min_pixel and begin != 0: # 跳过字符所在行
            continue
        elif nonzeros <= min_pixel and begin !=0 :
            end = line
            # if (end - begin) > min_width:
            chaLine.append(np.array([end-begin, begin, end]))
            begin, end = 0, 0
    return np.array(chaLine)


# 水平投影：获取像素在垂直方向的分布，用以切分字符
# img: 经过垂直投影的图像，是一行文本
# begin: img在原图像中的起始
# end: img在原图像中的终止
# return: 返回一组切分后的字符
def horizon(img, beginLine, endLine):
    # print("in horizen: ", img.shape)
    h, w = img.shape
    characters = []

    beginCol, endCol = 0, 0
    for col in range(w):
        nonzeros = np.sum(img[:,col] > 0)
        # print(nonzeros)
        if nonzeros > min_pixel and beginCol == 0:
            beginCol = col
        elif nonzeros > min_pixel and beginCol != 0:
            continue
        elif nonzeros <= min_pixel and beginCol != 0 and col - beginCol > min_width:
            endCol = col
            # if (end - begin) > min_width:
            characters.append(np.array([beginLine, beginCol, endLine, endCol]))
            beginCol, endCol = 0, 0
    return np.array(characters)


# 实现字符图片分割
def character_segment(img):
    # 转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯平滑
    blur = cv2.GaussianBlur(gray, (3, 3), 0) # sigma自行计算
    # 采用otsu自适应二值化算法
    ret, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    view_image(threshold, "binary image: otsu")

    seg_line = vertical(threshold)
    if len(seg_line) == 0:
        print("there is no character")
        exit(-1)
    characters = []
    for ind in range(len(seg_line)):
        char_line = threshold[seg_line[ind, 1]:seg_line[ind, 2], :]
        # view_image(char_line, "first line characters")
        characters.append(horizon(char_line, seg_line[ind, 1], seg_line[ind, 2]))
    # for rect in characters:
    # print(len(characters), characters[0])
    # for ind, line in enumerate(characters):
    #     for char, rect in enumerate(line):
    #         ch = threshold[rect[0]:rect[2], rect[1]:rect[3]]
    #         # cv2.imwrite("../seg/line_%d_ch_%d.jpg"%(ind, char), ch)
    #         if (rect[3]-rect[1]) > (rect[2] - rect[0])*1.6:
    #             bin, contours, hire = cv2.findContours(ch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #             if len(contours) > 0:
    #                 view_image(ch, "ch contours")
    #                 cv2.imwrite("../seg/line_%d_ch_%d.jpg" % (ind, char), ch)
    #
    #                 for i, curve in enumerate(contours):
    #                     print(curve.shape)
    #                     tl = [255, 255]
    #                     br = [0, 0]
    #                     for p in curve:
    #                         # print(p.shape, p)
    #                         if p[0, 0] < tl[0]:
    #                             tl[0] = p[0, 0]
    #                         if p[0, 1] < tl[1]:
    #                             tl[1] = p[0,1]
    #                         if p[0,0] > br[0]:
    #                             br[0] = p[0,0]
    #                         if p[0,1] > br[1]:
    #                             br[1] = p[0,1]
    #                     t = ch[tl[0]:br[0], tl[1]:br[1]]
    #                     # print(t)
    #                     if len(t)!=0:
    #                         view_image(t, "seg%d"%i)
    #                         cv2.imwrite("../seg/char_%d_ch_%d.jpg" % (ind, i), t)

    return np.array(characters)


import os
def del_file(path):
    for f in os.listdir(path):
        file = os.path.join(path, f)
        if os.path.isfile(file):
            os.remove(file)
        else:
            del_file(file)


if __name__ == '__main__':
    img = cv2.imread("../pic/test.jpg")
    del_file("../seg")

    characters = character_segment(img)
    if len(characters) == 0:
        print("no characters")
        exit(-1)
    for ind, line in enumerate(characters):
        for char, rect in enumerate(line):
            ch = img[rect[0]:rect[2], rect[1]:rect[3]]
            cv2.imwrite("../seg/line_%d_ch_%d.jpg"%(ind, char), ch)
            # if (rect[3]-rect[1]) > (rect[2] - rect[0])*1.6:
            #     bin, contours, hire = cv2.findContours(ch, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            #     if len(contours) > 0:
            #         cv2.drawContours(ch, contours, -1, (0, 0, 255), 1)
            #         view_image(ch, "ch contours")
