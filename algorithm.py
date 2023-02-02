import numpy as np
from numpy import ndarray


def plane(img: ndarray) -> ndarray:
    a = [np.array(img), np.array(img), np.array(img), np.array(img), np.array(img), np.array(img), np.array(img),
         np.array(img), np.array(img)]

    for k in range(9):
        [rows, cols] = a[k].shape
        for i in range(rows):
            for j in range(cols):
                if (7 - k) >= 0:
                    a[k][i, j] = (img[i, j] >> (7 - k)) & 1
                else:
                    a[k][i, j] = 1

    aa = np.concatenate((a[0], a[1], a[2]), axis=1)
    bb = np.concatenate((a[3], a[4], a[5]), axis=1)
    cc = np.concatenate((a[6], a[7], a[8]), axis=1)
    res = np.concatenate((aa, bb, cc), axis=0)
    return res


def equalize(img: ndarray) -> ndarray:
    if img.ndim == 3:
        for i in range(img.ndim):
            one_color = img[..., i]
            d = {}
            [rows, cols] = one_color.shape
            for i in range(rows):
                for j in range(cols):
                    if d.get(one_color[i][j]):
                        d[one_color[i][j]] += 1
                    else:
                        d[one_color[i][j]] = 1
            dd = {}
            for i in d:
                dd[i] = d[i] / (rows * cols)
            # print(sorted(dd.keys()))
            ndd = sorted(dd.keys())
            cnt = 0
            ddd = {}
            for i in ndd:
                ddd[i] = cnt + dd[i]
                cnt = ddd[i]
            # for i in ddd:
            #     print(i,ddd[i])
            for i in range(rows):
                for j in range(cols):
                    one_color[i][j] = ddd[one_color[i][j]]
        # print(img)
    else:
        one_color = img
        d = {}
        [rows, cols] = one_color.shape
        for i in range(rows):
            for j in range(cols):
                if d.get(one_color[i][j]):
                    d[one_color[i][j]] += 1
                else:
                    d[one_color[i][j]] = 1
        # for i in d:
        #     print(i, d[i])
        dd = {}
        for i in d:
            dd[i] = d[i] / (rows * cols)
        # for i in dd:
        #     print(i, dd[i])
        cnt = 0
        ddd = {}
        ndd = sorted(dd.keys())
        for i in ndd:
            ddd[i] = cnt + dd[i]
            cnt = ddd[i]
        # for i in ddd:
        #     print(i, ddd[i])
        for i in range(rows):
            for j in range(cols):
                one_color[i][j] = ddd[one_color[i][j]]
        # print(img)
    return img


def denoise(img: ndarray) -> ndarray:
    if img.ndim == 3:
        for ii in range(img.ndim):
            count = 7
            startWindow = 3
            c = int(count / 2) + 1
            original = img[..., ii]
            rows, cols = original.shape
            print(rows, cols)
            newI = np.zeros(original.shape)
            for i in range(c, rows - c):
                for j in range(c, cols - c):
                    k = int(startWindow / 2)
                    temp = original[i - k:i + k + 1, j - k:j + k + 1]
                    # print(i,j,k)
                    # print(i-k,i+k+1,j-k,j+k+1)
                    # print(temp)
                    median = np.median(temp)
                    mi = np.amin(temp)
                    ma = np.amax(temp)
                    if mi < median < ma:
                        if mi < original[i, j] < ma:
                            newI[i, j] = original[i, j]
                        else:
                            newI[i, j] = median
                    else:
                        while not (mi < median < ma or startWindow > count):
                            startWindow = startWindow + 2
                            k = int(startWindow / 2)
                            # print(i,j,k)
                            temp = original[i - k:i + k + 1, j - k:j + k + 1]
                            median = np.median(temp)
                            mi = np.amin(temp)
                            ma = np.amax(temp)
                            # if mi < median < ma or startWindow > count:
                            #     break
                        if mi < median < ma or startWindow > count:
                            if mi < original[i, j] < ma:
                                newI[i, j] = original[i, j]
                            else:
                                newI[i, j] = median
            for i in range(rows):
                for j in range(cols):
                    original[i][j] = newI[i, j]
        # print(img)
    else:
        count = 7
        startWindow = 3
        c = int(count / 2) + 1
        original = img
        rows, cols = original.shape
        print(rows, cols)
        newI = np.zeros(original.shape)
        for i in range(c, rows - c):
            for j in range(c, cols - c):
                k = int(startWindow / 2)
                temp = original[i - k:i + k + 1, j - k:j + k + 1]
                median = np.median(temp)
                mi = np.amin(temp)
                ma = np.amax(temp)
                if mi < median < ma:
                    if mi < original[i, j] < ma:
                        newI[i, j] = original[i, j]
                    else:
                        newI[i, j] = median
                else:
                    while not (mi < median < ma or startWindow > count):
                        startWindow = startWindow + 2
                        k = int(startWindow / 2)
                        temp = original[i - k:i + k + 1, j - k:j + k + 1]
                        median = np.median(temp)
                        mi = np.amin(temp)
                        ma = np.amax(temp)
                    if mi < median < ma or startWindow > count:
                        if mi < original[i, j] < ma:
                            newI[i, j] = original[i, j]
                        else:
                            newI[i, j] = median
        for i in range(rows):
            for j in range(cols):
                original[i][j] = newI[i, j]
    return img


def interpolate(img: ndarray) -> ndarray:
    print(img.ndim)
    if img.ndim == 3:
        rows, cols = img[..., 0].shape
        print(rows, cols)
        new_img = np.zeros((2 * rows, 2 * cols, 3))
        for i in range(img.ndim):
            old = img[..., i]
            new = new_img[..., i]
            for r in range(2 * rows):
                for c in range(2 * cols):
                    new[r][c] = old[int(r / 2)][int(c / 2)]
        return new_img
    else:
        rows, cols = img.shape
        print(rows, cols)
        new_img = np.zeros((2 * rows, 2 * cols, 3))
        old = img
        new = new_img
        for r in range(2 * rows):
            for c in range(2 * cols):
                new[r][c] = old[int(r / 2)][int(c / 2)]
        return new_img


def dft(img: ndarray) -> ndarray:
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    res = 20 * np.log(np.abs(fshift))
    return res


def butterworth(img: ndarray) -> ndarray:
    print(img.ndim)
    rows, cols = img.shape
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    p = fshift
    for r in range(rows):
        for c in range(cols):
            d = pow(pow(r - rows / 2, 2) + pow(c - cols / 2, 2), 0.5)
            d0 = 80
            p[r][c] = p[r][c] * (1 / (1 + pow(d / d0, 2)))
    iff = np.fft.ifft2(fshift)
    res = 20 * np.log(np.abs(fshift))
    print(iff)
    return np.abs(iff)


def canny(img: ndarray) -> ndarray:
    gaussian = np.zeros([5, 5])
    pi = 3.1415926
    e = 2.71828
    sigma1 = sigma2 = 1
    sum = 0
    for i in range(5):
        for j in range(5):
            gaussian[i, j] = pow(e, (-1 / 2 * (np.square(i - 3) / np.square(sigma1)
                                               + (np.square(j - 3) / np.square(sigma2)))) / (
                                         2 * pi * sigma1 * sigma2))
            sum = sum + gaussian[i, j]
    gaussian = gaussian / sum
    # print(gaussian*sum)
    gray = img
    W, H = gray.shape
    new_gray = np.zeros([W - 5, H - 5])
    for i in range(W - 5):
        for j in range(H - 5):
            new_gray[i, j] = np.sum(gray[i:i + 5, j:j + 5] * gaussian)
    W1, H1 = new_gray.shape
    dx = np.zeros([W1 - 1, H1 - 1])
    dy = np.zeros([W1 - 1, H1 - 1])
    d = np.zeros([W1 - 1, H1 - 1])
    # 计算提督
    for i in range(W1 - 1):
        for j in range(H1 - 1):
            dx[i, j] = new_gray[i, j + 1] - new_gray[i, j]
            dy[i, j] = new_gray[i + 1, j] - new_gray[i, j]
            d[i, j] = np.sqrt(np.square(dx[i, j]) + np.square(dy[i, j]))
    W2, H2 = d.shape
    gd = np.copy(d)
    gd[0, :] = gd[W2 - 1, :] = gd[:, 0] = gd[:, H2 - 1] = 0
    for i in range(1, W2 - 1):
        for j in range(1, H2 - 1):
            if d[i, j] == 0:
                gd[i, j] = 0
            else:
                gradX = dx[i, j]
                gradY = dy[i, j]
                cur_grad = d[i, j]
                # 判断方向
                if np.abs(gradY) > np.abs(gradX):
                    weight = np.abs(gradX) / np.abs(gradY)
                    grad2 = d[i - 1, j]
                    grad4 = d[i + 1, j]
                    if gradX * gradY > 0:
                        grad1 = d[i - 1, j - 1]
                        grad3 = d[i + 1, j + 1]
                    else:
                        grad1 = d[i - 1, j + 1]
                        grad3 = d[i + 1, j - 1]
                else:
                    weight = np.abs(gradY) / np.abs(gradX)
                    grad2 = d[i, j - 1]
                    grad4 = d[i, j + 1]
                    if gradX * gradY > 0:
                        grad1 = d[i + 1, j - 1]
                        grad3 = d[i - 1, j + 1]
                    else:
                        grad1 = d[i - 1, j - 1]
                        grad3 = d[i + 1, j + 1]

                nb_grad1 = weight * grad1 + (1 - weight) * grad2
                nb_grad2 = weight * grad3 + (1 - weight) * grad4
                if cur_grad >= nb_grad1 and cur_grad >= nb_grad2:
                    gd[i, j] = cur_grad
                else:
                    gd[i, j] = 0

    W3, H3 = gd.shape
    result = np.zeros([W3, H3])

    lower = 0.2 * np.max(gd)
    higher = 0.3 * np.max(gd)
    for i in range(1, W3 - 1):
        for j in range(1, H3 - 1):
            if gd[i, j] < lower:
                result[i, j] = 0
            elif gd[i, j] > higher:
                result[i, j] = 1
            elif ((gd[i - 1, j - 1:j + 1] < higher).any() or (gd[i + 1, j - 1:j + 1]).any()
                  or (gd[i, [j - 1, j + 1]] < higher).any()):
                result[i, j] = 1
    return result


def morphology(img: ndarray) -> ndarray:
    # 结构元为3*3矩形
    rows, cols = img.shape
    print(rows, cols)
    new = np.zeros(img.shape)
    for r in range(rows):
        for c in range(cols):
            l1 = r - 1 if r - 1 >= 0 else 0
            l2 = r + 2 if r + 2 <= rows else rows
            r1 = c - 1 if c - 1 >= 0 else 0
            r2 = c + 2 if c + 2 <= cols else cols
            new[r][c] = np.amin(img[l1:r + 2, r1:c + 2])
    newn = np.zeros(img.shape)
    print((img == new).all())
    for r in range(rows):
        for c in range(cols):
            l1 = r - 1 if r - 1 >= 0 else 0
            l2 = r + 2 if r + 2 <= rows else rows
            r1 = c - 1 if c - 1 >= 0 else 0
            r2 = c + 2 if c + 2 <= cols else cols
            newn[r][c] = np.amax(new[l1:l2, r1:r2])
    print((new == newn).all())
    return newn

def flip(img: ndarray) -> ndarray:
    if img.ndim==3:
        for i in range(img.ndim):
            one_color=img[...,i]
            temp=np.copy(one_color)
            rows,cols=one_color.shape
            for r in range(rows):
                for c in range(cols):
                    one_color[r][c]=temp[r][cols-c-1]
    else:
        one_color = img
        temp = np.copy(one_color)
        rows, cols = one_color.shape
        for r in range(rows):
            for c in range(cols):
                one_color[r][c] = temp[r][cols - c - 1]
    return img
def fisheye(img: ndarray) -> ndarray:
    if img.ndim==3:
        rows,cols,c = img.shape
        center_x,center_y = rows/2,cols/2
        #radius = min(center_x,center_y)
        #radius = math.sqrt(rows**2+cols**2)/2
        radius = pow(rows**2+cols**2,1/2)/2
        new_img = img.copy()
        for i in range(rows):
            for j in range(cols):
                #dis = math.sqrt((i-center_x)**2+(j-center_y)**2)
                dis = pow((i-center_x)**2+(j-center_y)**2,1/2)
                if dis <= radius:
                    new_i = np.int(np.round(dis/radius*(i-center_x)+center_x))
                    new_j = np.int(np.round(dis/radius*(j-center_y)+center_y))
                    new_img[i,j] = img[new_i,new_j]
                    #print((i,j),'\t',(new_i,new_j))
        return new_img
    else:
        rows,cols = img.shape
        center_x,center_y = rows/2,cols/2
        #radius = min(center_x,center_y)
        #radius = math.sqrt(rows**2+cols**2)/2
        radius = pow(rows**2+cols**2,1/2)/2
        new_img = img.copy()
        for i in range(rows):
            for j in range(cols):
                #dis = math.sqrt((i-center_x)**2+(j-center_y)**2)
                dis = pow((i-center_x)**2+(j-center_y)**2,1/2)
                if dis <= radius:
                    new_i = np.int(np.round(dis/radius*(i-center_x)+center_x))
                    new_j = np.int(np.round(dis/radius*(j-center_y)+center_y))
                    new_img[i,j] = img[new_i,new_j]
                    #print((i,j),'\t',(new_i,new_j))
        return new_img


    # Your Code
