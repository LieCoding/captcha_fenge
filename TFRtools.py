# 导入包
import tensorflow as tf
import numpy as np
import os

"""
    介绍：
    tfrecord格式是tensorflow官方推荐的数据格式，把数据、标签进行统一的存储
    tfrecord文件包含了tf.train.Example 协议缓冲区(protocol buffer，协议缓冲区包含了特征 Features)， 能让tensorflow更好的利用内存。

    Author:Ephemerptero
    Version:1.2.0
    Date:2019-3-23
    QQ:605686962
"""

"""
# 定义生成整数型,字符串型和浮点型属性的方法，这是将数据填入到Example协议
# 内存块(protocol buffer)的第一步，以后会调用到这个方法
"""


def Int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def Bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def Float_frature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


"""
# TFRecord格式保存函数:
# data:格式为M x N ，M为数据集个数，N为一个数据项
# label:同data
# filename: 路径名+文件名(无后缀名)
# num_shards: 将该文件拆分为n个TFR保存
"""
def SaveByTFRecord(data, label, prefix, num_shards=5):
    # 变化为数组格式
    data = np.array(data)
    label = np.array(label)

    # 建立序号
    NUM = np.array(range(0, data.shape[0]))

    # 创建文件夹
    dir, file = os.path.split(prefix)
    if not os.path.isdir(dir):
        print('文件夹%s未创建，现在在当前目录下创建..' % (dir))
        os.mkdir(dir)

    # 分割数据集(0~len-1 ,num_shards等分)
    seg_list = np.int32(np.linspace(0, data.shape[0] - 1, num_shards + 1))

    for i in range(num_shards):

        # 生成文件名
        filename = prefix + '-%.4d-%.4d' % (i, num_shards - 1)

        # 写入TFR
        writer = tf.python_io.TFRecordWriter(filename)
        for j in range(seg_list[i], seg_list[i + 1] + 1):
            # 将图像和标签转化成字符串
            image_raw = data[j].tostring()
            label_raw = label[j].tostring()
            # 将图像和标签数据作为一个example
            example = tf.train.Example(features=tf.train.Features(feature={
                'num': Int64_feature(NUM[j]),
                'image': Bytes_feature(image_raw),
                'label': Bytes_feature(label_raw)
            }))
            writer.write(example.SerializeToString())
            print('已经写入至第%d个文件中第%d项，序号项类型：%s，数据项类型：%s ， 标签项类型：%s' % (
            i, NUM[j], NUM[j].dtype, data[j].dtype, label[j].dtype))
        writer.close()

    # 游览文件
    fileSets = os.listdir(dir)
    print('成功存储为TFRecord格式！！，%s文件夹下生成文件如下：\n' % (dir))
    print(fileSets)


"""
 解析读取TFRecord文件:
# sameName: TFRecord文件夹下所有相同文件名修饰，例如：'./TFR/data-0001of0005' --->  './TFR/data-*'
# isShuffle: 读取全部TFR文件顺序是否打乱（True/False）
# datatype: 数据集类型，以MNIST数据集为例，其数据类型为float32
# labeltype: 标签集类型，以MNIST标签集为例，其数据类型为float64

"""
def ReadFromTFRecord(sameName, isShuffle, datatype, labeltype):
    # 生成文件列表
    fileslist = tf.train.match_filenames_once(sameName)

    # 由文件列表生成文件队列
    filename_queue = tf.train.string_input_producer(fileslist, shuffle=isShuffle)

    # 实例化TFRecordReader类，读取每个样本
    reader = tf.TFRecordReader()

    # 序列化入内存
    _, serialization = reader.read(filename_queue)

    # 解析样本
    features = tf.parse_single_example(
        serialization,
        features={
            "image": tf.FixedLenFeature([], tf.string),  # 数据内容
            "label": tf.FixedLenFeature([], tf.string),  # 标签内容
            "num": tf.FixedLenFeature([], tf.int64)  # 序号
            ## 解析其他属性
        })

    # decode_raw()字符信息解码
    data = tf.decode_raw(features["image"], datatype)
    label = tf.decode_raw(features['label'], labeltype)
    num = tf.cast(features["num"], tf.int32)
    # int64类型可以用tf.cast()转换成其他类型

    return [data, label, num]


"""
数据集批处理:
# data:入队数据项data
# label:入队标签项label
# num:入队序号项num
# dataSize:数据项大小 （例如MNIST：784）
# lableSize:标签项大小 （例如MNIST：10）
# isShuffle: 是否打乱顺序
# batchSize: 批次大小 （True/False)
"""
def DataBatch( data, label,num, dataSize, labelSize, isShuffle, batchSize):
    # 设置batch属性
    min_after_dequeue = 3 * batchSize  # 队列中至少保留个数，否则等待
    capacity = 5 * batchSize  # 队列最大容量
    # 设置数据型号
    # num.set_shape(1)
    data.set_shape(dataSize)
    label.set_shape(labelSize)

    # 是否样本打乱
    if isShuffle:
        # 打乱处理
        # tf.data.Dataset.shuffle(min_after_dequeue= min_after_dequeue).batch(batch_size= batchSize)
        [data_batch, label_batch, num_batch] = tf.train.shuffle_batch([data, label, num], batch_size=batchSize,
                                                                      capacity=capacity,
                                                                      min_after_dequeue=min_after_dequeue)
    else:
        # 正常处理
        [data_batch, label_batch, num_batch] = tf.train.batch([data, label, num], batch_size=batchSize,
                                                              capacity=capacity)

    return [data_batch, label_batch, num_batch]


"""

 ########################### 用MINIST数据集测试(Demo) ###################################        
"""
if __name__ == '__main__':
    # Taking MNIST data as example:

    # 导入opencv
    import cv2

    # 导入MNISI数据

    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets(r'.\MNIST', one_hot=True)

    # 读取mnist数据。
    images = mnist.train.images  # 55000x784
    labels = mnist.train.labels  # 55000x10

    # # 设置存储路径，注意：TFR是文件夹，MNIST是文件名，TFR文件为二进制无后缀名
    savepath = r'.\TFR\MNIST'

    # 分成5批次存储
    SaveByTFRecord(images,labels,savepath,5)

    # 读取TFR,不打乱文件顺序，指定数据类型
    [data, label, num] = ReadFromTFRecord(sameName=r'.\TFR\MNIST-*', isShuffle=False, datatype=tf.float32,
                                          labeltype=tf.float64)

    # 批量处理，送入队列数据，指定数据大小，不打乱数据项，设置批次大小10
    [data_batch, label_batch, num_batch] = DataBatch( data, label,num, dataSize=28 * 28, labelSize=10, isShuffle=False,
                                                     batchSize=10)

    with tf.Session() as sess:
        # 初始化变量
        init = (tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)

        # 开启协调器
        coord = tf.train.Coordinator()
        # 启动线程
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # 测试1： 单个输出(运行该循环，请关闭DataBatch)
        # while True:
        #     TD = sess.run([data,label,num])
        #     testnum = TD[2]
        #     testdata = np.reshape(TD[0], [ 28, 28, 1])*255
        #     testlabel = TD[1]
        #     print('读取序号',testnum)
        #     print('真实标签',labels[testnum])
        #     print('读取标签',testlabel)
        #     cv2.imshow('real',images[testnum].reshape([28,28,1]))
        #     cv2.imshow('read',testdata.astype(np.uint8))
        #     cv2.waitKey(0)

        # 测试2：输出一批
        while True:
            TDB = sess.run([data_batch, label_batch, num_batch])
            for N, L in zip(TDB[2], TDB[1]):
                print(N, ' :', L)
            for D in TDB[0]:
                D = np.reshape(D, [28, 28, 1])
                cv2.imshow('TDB', D)
                cv2.waitKey(0)

        # 关闭线程
        coord.request_stop()
        coord.join(threads)




