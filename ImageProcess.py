"""
该脚本定义了数字分割相关方法，并在main函数举例说明
"""
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# 定义图片显示
def SHOW(title,img):
    cv2.namedWindow(title, flags=cv2.WINDOW_NORMAL)
    cv2.imshow(title, img)

# 向量显示
def STEM(v):
    plt.stem(v)
    plt.show()

# 定义二值图像去除小连通域
def RemoveSmallCC(bin,small,connectivity=8):
    # (1)获取连通域标签
    ret, labels = cv2.connectedComponents(bin, connectivity=connectivity)
    # (2)去小连通域
    for n in range(ret + 1):# 0~max
        num = 0 # 清零
        for elem in labels.flat:
            if elem == n:
                num += 1
        if num < small:  # 去除小连通域
            bin[np.where(labels == n)] = 0
    return bin

# 二值图像一维投影
def BIN_PROJECT(bin):
    IMG = bin / 255.0 # 浮点型
    PROJ = np.zeros(shape=(IMG.shape[1]))
    for col in range(IMG.shape[1]):
        v = IMG[:, col]# 列向量
        PROJ[col] = np.sum(v)
    return PROJ

# 提取数字部分
def Extract_Num(PROJ):
    num = 0
    COUNT = []
    LOC = []
    for i in range(len(PROJ)):
        if PROJ[i]:# 如果非零则累加
            num+=1
            if i == 0 or PROJ[i-1]==0:# 记录片段起始位置
                start = i
            if i == len(PROJ)-1 or PROJ[i+1]==0 :# 定义片段结束标志，并记录片段
                end = i
                if num > 10:# 提取有效片段
                    COUNT.append(num)
                    LOC.append((start,end))
                num = 0 #清0
    return LOC,COUNT


# COUNT优化
def COUNT_OPTIMIZE(IMG,LOC,COUNT):
    # COUNT归一化
    COUNT = [i/max(COUNT) for i in COUNT]
    COUNT2 = []
    for loc in LOC:
        seg = IMG[:,loc[0]:loc[1]]
        COUNT2.append( np.sum(seg,axis=None))
    # COUNT2 归一化
    COUNT2 = [i / max(COUNT2) for i in COUNT2]
    # 优化COUNT
    return [0.7*i+0.3*j for i,j in zip(COUNT,COUNT2)]

# 分割数字
def Segment4_Num(COUNT,LOC):

    assert len(COUNT)<=4 and len(COUNT)>0
    # 数字部分分析
    if len(COUNT) ==4:#(1,1,1,1)
        return LOC

    if len(COUNT)==3:#(1,1,2)
        idx = np.argmax(np.array(COUNT))# 最大片段下标
        r = LOC[idx]# 最大片段位置
        start = r[0]
        end = r[1]
        m = (r[0]+r[1])//2 # 中间位置
        # 修改LOC[idx]
        LOC[idx] = (start,m)
        LOC.insert(idx+1,(m+1,end))
        return LOC

    if len(COUNT) ==2:#(2,2)or(1,3)
        skew = max(COUNT)/min(COUNT)# 计算偏移程度
        if skew<1.7:# 认为是（2，2）
           start1 = LOC[0][0]
           end1 = LOC[0][1]
           start2 = LOC[1][0]
           end2 = LOC[1][1]
           m1 = (start1+end1)//2
           m2 = (start2+end2)//2
           return [(start1,m1),(m1+1,end1),(start2,m2),(m2+1,end2)]
        else:       # 认为是（1，3）
            idx = np.argmax(np.array(COUNT))# 最大片段下标
            start = LOC[idx][0]
            end = LOC[idx][1]
            m1 = (end-start)//3+start
            m2 = (end-start)//3*2+start
            # 修改LOC[idx]
            LOC[idx] = (start, m1)
            LOC.insert(idx+1,(m1+1,m2))
            LOC.insert(idx+2,(m2+1,end))
            return LOC
    if len(COUNT) ==1:# (4)
        start = LOC[0][0]
        end = LOC[0][1]
        m1 = (end-start)//4+start
        m2 = (end - start) // 4*2 + start
        m3 = (end - start) // 4*3 + start
        return [(start,m1),(m1+1,m2),(m2+1,m3),(m3+1,end)]


#---------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    img_path = r'./images'  # 验证码路径
    img_list = os.listdir(img_path)  # 图片列表

    # 读取图片并保存为灰色
    img_gray = cv2.imread(r'./images/0250.jpg',flags=cv2.IMREAD_GRAYSCALE)
    SHOW('gray',img_gray)

    # 二值化
    _,img_bin = cv2.threshold(img_gray,int(0.9*255),255,cv2.THRESH_BINARY_INV) # 二值化阈值系数选取0.9
    SHOW('bin',img_bin)

    # 去除斑点
    img_bin = RemoveSmallCC(img_bin, 200,connectivity=4)
    SHOW('remove_spot', img_bin)

    # 形态学腐蚀
    size = 2
    kernel = np.ones((size, size), dtype=np.uint8)
    img_erosion = cv2.erode(img_bin, kernel, iterations=1)
    # 小连通域滤除
    img_erosion = RemoveSmallCC(img_erosion,30)
    SHOW('erode_2x2', img_erosion)
    # 形态学膨胀
    img_dila = cv2.dilate(img_erosion, kernel, iterations=1)
    SHOW('dilate_2x2',img_dila)

    # 再次滤除横线
    PROJ = BIN_PROJECT(img_dila)# 投影
    STEM(PROJ)
    flaw_area = np.where(PROJ<5)#瑕疵区域
    img_dila[:,flaw_area]=0 # 去除横线
    IMG = RemoveSmallCC(img_dila,200) # 去除小连通区域
    SHOW('remove_line',IMG)

    # PROJ提取数字部分
    PROJ = BIN_PROJECT(IMG)
    STEM(PROJ)
    LOC,COUNT = Extract_Num(PROJ)

    # 优化COUNT
    COUNT = COUNT_OPTIMIZE(IMG, LOC, COUNT)

    # 分割数字
    LOC = Segment4_Num(COUNT,LOC)

    NUM0 = IMG[:,LOC[0][0]:LOC[0][1]]
    NUM0 = RemoveSmallCC(NUM0,50)
    NUM1 = IMG[:,LOC[1][0]:LOC[1][1]]
    NUM1 = RemoveSmallCC(NUM1,50)
    NUM2 = IMG[:,LOC[2][0]:LOC[2][1]]
    NUM2 = RemoveSmallCC(NUM2,50)
    NUM3 = IMG[:,LOC[3][0]:LOC[3][1]]
    NUM3 = RemoveSmallCC(NUM3,50)
    SHOW('NUM0',NUM0)
    SHOW('NUM1',NUM1)
    SHOW('NUM2',NUM2)
    SHOW('NUM3',NUM3)


    cv2.waitKey(0)



