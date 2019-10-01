from ImageProcess import *
import cv2
import numpy as np
import os
import TFRtools
import shutil

def Segment4_NUMBER(bin):

    # 去除斑点
    img_bin = RemoveSmallCC(bin, 200, connectivity=4)
    # 形态学腐蚀
    size = 2
    kernel = np.ones((size, size), dtype=np.uint8)
    img_erosion = cv2.erode(img_bin, kernel, iterations=1)
    # 小连通域滤除
    img_erosion = RemoveSmallCC(img_erosion, 20)
    # 形态学膨胀
    img_dila = cv2.dilate(img_erosion, kernel, iterations=1)

    # 滤除横线
    PROJ = BIN_PROJECT(img_dila)  # 投影
    flaw_area = np.where(PROJ < 5)  # 瑕疵区域
    img_dila[:, flaw_area] = 0  # 去除横线
    IMG = RemoveSmallCC(img_dila, 200)  # 去除小连通区域

    # PROJ提取数字部分
    PROJ = BIN_PROJECT(IMG)
    LOC, COUNT = Extract_Num(PROJ)

    # 优化COUNT
    COUNT = COUNT_OPTIMIZE(IMG, LOC, COUNT)

    # (4)分割数字部分
    LOC = Segment4_Num(COUNT, LOC)
    # 分割数字
    NUM0 = IMG[:, LOC[0][0]:LOC[0][1]]
    NUM0 = cv2.resize(NUM0,(32,32))
    NUM0 = RemoveSmallCC(NUM0, 70)
    NUM1 = IMG[:, LOC[1][0]:LOC[1][1]]
    NUM1 = cv2.resize(NUM1, (32, 32))
    NUM1 = RemoveSmallCC(NUM1, 70)
    NUM2 = IMG[:, LOC[2][0]:LOC[2][1]]
    NUM2 = cv2.resize(NUM2, (32, 32))
    NUM2 = RemoveSmallCC(NUM2, 70)
    NUM3 = IMG[:, LOC[3][0]:LOC[3][1]]
    NUM3 = cv2.resize(NUM3, (32, 32))
    NUM3 = RemoveSmallCC(NUM3, 70)

    return [NUM0,NUM1,NUM2,NUM3]

# 生成相关目录
def GEN_DIR():
    import os
    if not os.path.isdir('NUMBERS'):
        print('文件夹NUMBERS未创建，现在在当前目录下创建..')
        os.mkdir('NUMBERS')
        for i in range(10):
            os.makedirs('./NUMBERS/%d'%i)
    if not os.path.isdir('TEST'):
        print('文件夹TEST未创建，现在在当前目录下创建..')
        os.mkdir('TEST')

####################### main #################################

if __name__ == '__main__':
    img_path = r'./images'# 验证码路径
    img_list = os.listdir(img_path) # 图片列表
    NUM_PATH = r'./NUMBERS' # 训练样本可视化

    # 创建相关目录
    GEN_DIR()

    # 清理NUMBBERS下全部图片
    for i in range(10):
        num_dir = os.path.join(NUM_PATH,str(i))
        num_list = os.listdir(num_dir)
        for num in num_list:
            os.remove(os.path.join(num_dir,num))
        print('成功清理文件夹./NUMBERS/%d下图片'%i)

    for item in os.listdir(r'./TEST'):
        os.remove(os.path.join(r'./TEST',item))
        print('成功清理文件夹./TEST/下图片%s' % item)

    #--------------------------- 随机选取70%作为训练集，%30作为测试集 ---------------------------------------------

    data_total = len(img_list) # 数据集总数
    train_total = int(0.7*data_total) #训练集总量
    test_total = data_total - train_total # 测试集总量

    #随机采样训练样本
    import random
    train_list = random.sample(img_list,train_total) #训练样本列表
    test_list = [item for item in img_list if item not in train_list] # 测试样本列表

    #*************** (1)训练样本分解并写入TFR *****************************
    traindata = []
    trainlabel = []

    step  = 0
    for img_name in train_list:
        prefix = os.path.splitext(img_name)[0]#图片标签名

        # 制作标签集
        for n in prefix:
            onehot = np.zeros(shape=10,dtype=np.uint8)
            onehot[int(n)] = 1
            trainlabel.append(onehot) # 加入标签集

        # 制作数据集
        ##(1)分割数字
        img_gray = cv2.imread(os.path.join(img_path,img_name), flags=cv2.IMREAD_GRAYSCALE)# 读取图片并保存为灰色
        _, img_bin = cv2.threshold(img_gray, int(0.9 * 255), 255, cv2.THRESH_BINARY_INV)  # 二值化阈值系数选取0.9
        numbers = Segment4_NUMBER(img_bin)# 分割数字

        ##(2)保存数字
        for i in range(4):
           number = numbers[i]
           # 保存为图片
           savename = prefix+'_%d.jpg'%i
           savepath = r'./NUMBERS/%s'%prefix[i]
           cv2.imwrite(os.path.join(savepath,savename),number)
           # 保存traindata
           number = numbers[i]/255 # 归一化
           number = np.reshape(number, [32 * 32])  # 平铺
           traindata.append(number)

        step+=1
        print('成功分割第%d/%d张训练验证码:%s'%(step,len(train_list),prefix))

    ## 训练样本存入TFR
    TFRtools.SaveByTFRecord(traindata,trainlabel,r'./TFR/CODE_TRAIN',5)

    #*************************** (2)测试样本分解写入TFR ******************************************************

    testdata = []
    testlabel = []

    step = 0
    for img_name in test_list:
        #图片标签名
        prefix = os.path.splitext(img_name)[0]

        # 制作标签集
        for n in prefix:
            onehot = np.zeros(shape=10, dtype=np.uint8)
            onehot[int(n)] = 1
            testlabel.append(onehot)  # 加入标签集

        # 制作数据集
        ## （1）分割数字
        img_gray = cv2.imread(os.path.join(img_path, img_name), flags=cv2.IMREAD_GRAYSCALE)  # 读取图片并保存为灰色
        _, img_bin = cv2.threshold(img_gray, int(0.9 * 255), 255, cv2.THRESH_BINARY_INV)  # 二值化阈值系数选取0.9
        numbers = Segment4_NUMBER(img_bin)  # 分割数字

        ##（2）保存数字
        for i in range(4):
            # 保存testdata
            number = numbers[i] / 255  # 归一化
            number = np.reshape(number, [32 * 32])  # 平铺
            testdata.append(number)

        ## (3)复制图片
        src = os.path.join(img_path,img_name)
        dst = os.path.join(r'./TEST',img_name)
        shutil.copy(src,dst)

        step += 1
        print('成功分割第%d/%d张测试验证码:%s' % (step, len(test_list), prefix))

    ## 测试样本存入TFR
    TFRtools.SaveByTFRecord(testdata,testlabel,r'./TFR/CODE_TEST',5)






