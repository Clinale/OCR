import cv2
import numpy as np
import matplotlib.pyplot as plt


def view_img(img, window):
    plt.imshow(img)
    plt.title(window)
    plt.show()


def view_image(img, window):
    cv2.namedWindow(window)
    cv2.imshow(window, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    lena = cv2.imread("./lena.jpg")
    match = lena[200:400, 200:350, :]
    view_image(lena, "lena")
    view_image(match, "match")

    cv2.imwrite("./match.jpg", match)

    th, tw, ch = match.shape
    result = cv2.matchTemplate(lena, match, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    tl = max_loc
    br = (tl[0]+tw, tl[1]+th)
    cv2.rectangle(lena, tl, br, (0, 0, 255), 3)
    view_image(lena, "math lena")

    cv2.imwrite("./match_lena.jpg", lena)