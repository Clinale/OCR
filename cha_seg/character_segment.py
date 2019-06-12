'''
usage: python charachter_segment path_to_img
'''

import numpy as np
import cv2
import os
# import matplotlib.pyplot as plt

# 采用投影法对文本图像进行分割
min_pixel = 2 # 字符占用的最小像素
min_width = 5 # 字符最小宽度

# 表示对图片进行垂直投影
MODE_VER = 0
# 表示对图片进行水平投影
MODE_HOR = 1


def del_file(path):
    for f in os.listdir(path):
        file = os.path.join(path, f)
        if os.path.isfile(file):
            os.remove(file)
        else:
            del_file(file)

# 保存分割后的字符图片
save_path = "./seg"
if not os.path.exists(save_path):
    # os.mkdir(save_path)
    os.mkdir(save_path)
else:
    # print("del file")
    del_file(save_path)


# 显示图片
def view_image(img, name):
    cv2.namedWindow(name)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 返回字符的水平/垂直投影
# img: 二值化图像
# mode: VER：垂直方向投影
#       HOR：水平方向投影
# ret: hor表示图像在水平方向的投影
#      ver表示图像在垂直方向的投影
def projection(img, mode):
    assert ((len(img.shape)) == 2)
    # pro = []
    # 保存水平/垂直方向的投影
    h, w = img.shape
    if mode == MODE_HOR:
        hor = np.zeros((1, w), np.uint32)
        for l in range(h):
            for c in range(w):
                hor[0, c] += (img[l, c] > 0)
        return hor
    elif mode == MODE_VER:
        ver = np.zeros((1,h), np.uint32)
        for l in range(h):
            for c in range(w):
                ver[0, l] += (img[l, c]>0)
        return ver


# 画出投影图
def drawPro(pro, mode):

    if mode == MODE_HOR:
        title = "PRO_HOR"
        h, w = int(np.max(pro))+10, pro.shape[1]
        # print(h, w, type(h), type(w))
        mat = np.zeros((h, w)).astype(np.uint8)

        for c in range(w):
            for ele in range(0, pro[0, c]):
                mat[h-1-ele, c] = 255

    elif mode == MODE_VER:
        title = "PRO_VER"
        h, w = pro.shape[1], int(np.max(pro)+10)

        # print(h, w, type(h), type(w))
        mat = np.zeros((h, w)).astype(np.uint8)

        for l in range(h):
            for ele in range(0, pro[0, l]):
                mat[l, ele] = 255
    view_image(mat, title)
    return mat


# 基于投影实现行切割/列切割
# mode: MODE_VER: 实现行切割
#      MODE_HOR: 实现列切割
# pro: 投影数据
# begin, end: 行/列起始位置，终止位置
# ret: np.array([[beginRow, beginCol, endRow, endCol], {}])
def simplecut(mode, pro, begin, end):
    cut = []

    len = pro.shape[1]
    head, tail = 0, 0
    for l in range(len):
        if pro[0, l] > min_pixel and head == 0:  # 记录起始行
            head = l
        elif pro[0, l] <= min_pixel and head != 0: # and l - head > min_width:  # 记录终止行，且保证字符最小宽度
            tail = l
            if mode == MODE_VER: # 行分割
                cut.append(np.array([head, begin, tail, end]))
            elif mode == MODE_HOR: # 列分割
                cut.append(np.array([begin, head, end, tail]))
            head, tail = 0, 0

    return np.array(cut)


# 基于垂直投影实现行分割
def rowcutting(ver, beginCol, endCol):
    pass


# 基于水平投影实现列切割
# 同rowcutting
def columncutting(hor, beginRow, endRow):
    pass


# 实现字符图片分割，假设文字排列方向是水平的
def character_segment(img):
    # 转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯平滑,消除噪声数据
    blur = cv2.GaussianBlur(gray, (3, 3), 0) # sigma自行计算
    # 采用otsu自适应二值化算法，INV实现黑底白字
    ret, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    view_image(threshold, "threshold")

    # 假设文字方向是从左往右，先进行垂直投影，方便行分割
    ver = projection(threshold, MODE_VER)
    # print(img.shape, ver.shape)
    drawPro(ver, MODE_VER)

    seg_lines = simplecut(MODE_VER, ver, 0, gray.shape[1])

    # seg_line = vertical(threshold)
    if len(seg_lines) == 0:
        print("there is no character")
        exit(-1)

    seg_chars = []
    # print(seg_lines.shape)
    for line in range(len(seg_lines)):
        # print(seg_lines[line, 1], seg_lines[line, 3])
        chars = threshold[seg_lines[line, 0]:seg_lines[line, 2], seg_lines[line, 1]:seg_lines[line, 3]]
        # view_image(chars, "characters in line")

        hor = projection(chars, MODE_HOR) # 基于行字符实现水平投影
        # hor = drawPro(hor, MODE_HOR)
        seg_chars.append(simplecut(MODE_HOR, hor, seg_lines[line, 0], seg_lines[line, 2]))
        # characters.append(horizon(char_line, seg_line[ind, 1], seg_line[ind, 2]))

    return np.array(seg_chars)


# import os
# def del_file(path):
#     for f in os.listdir(path):
#         file = os.path.join(path, f)
#         if os.path.isfile(file):
#             os.remove(file)
#         else:
#             del_file(file)
#

import sys
if __name__ == '__main__':
    # img = cv2.imread("../pic/pic.png")
    if len(sys.argv) != 2:
        print(__doc__)
        exit(-1)
    img = cv2.imread(sys.argv[1])
    if img is None or len(img) == 0:
        print("the file %s does not exist!"%sys.argv[1])
        exit(-1)

    characters = character_segment(img)
    if len(characters) == 0:
        print("no characters")
        exit(-1)
    for ind, line in enumerate(characters):
        for char, rect in enumerate(line):
            ch = img[rect[0]:rect[2], rect[1]:rect[3]]
            # view_image(ch, "char")
            cv2.imwrite(os.path.join(save_path, "line_%d_ch_%d.jpg"%(ind, char)), ch)
