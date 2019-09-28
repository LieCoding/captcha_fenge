"""
PROJECT:CODE_IDENTIFICATION
Author:Ephemeroptera
Date:2019-4-23
QQ:605686962
"""

import tensorflow as tf
import  numpy as np
import TFRtools
import pickle

# 生成相关目录保存生成信息
def GEN_DIR():
    import os
    if not os.path.isdir('ckpt'):
        print('文件夹ckpt未创建，现在在当前目录下创建..')
        os.mkdir('ckpt')
    if not os.path.isdir('trainLog'):
        print('文件夹ckpt未创建，现在在当前目录下创建..')
        os.mkdir('trainLog')

"""
#********************* CNN架构 *********************************

输入：
    x:[-1,32,32,1] 输入图像
    y_:[-1,10]     标签
CNN:
    CONV1: w(3x3),stride = 1,pad = 'same',fmaps = 32 ([-1,16,16,32])
    POOL1: p(2x2),stride = 2,pad = 'same'
    RELU
    
    CONV2: w(3x3),stride = 1,pad = 'same,fmaps = 64   ([-1,8,8,64])
    POOL2: p(2x2),stride = 2,pad = 'same'
    RELU,Reshape                                      ([-1,8*8*64])
    
    DENSE1: fmaps = 1024                               ([-1,1024])
    RELU
    Dropout
    
    DENSE2: fmaps = 10                                   ([-1,10])
    Softmax
"""

# 定义输入
x = tf.placeholder(tf.float32,[None,32,32,1],'x')
y_ = tf.placeholder(name="y_", shape=[None, 10],dtype=tf.float32)

# 第一层卷积层 32x32 to 16x16 ， fmaps = 32
CONV1 = tf.layers.conv2d(x,32,5,padding='same',activation=tf.nn.relu,
                         kernel_initializer=tf.random_normal_initializer(0,0.1),
                         name='CONV1')
POOL1 = tf.layers.max_pooling2d(CONV1,2,2,padding='same',name='POOL1')

# 第二层卷积层 16x16 to 8x8, fmaps =64
CONV2 = tf.layers.conv2d(POOL1,64,5,padding='same',activation=tf.nn.relu,
                         kernel_initializer=tf.random_normal_initializer(0,0.1),
                         name='CONV2')
POOL2 = tf.layers.max_pooling2d(CONV2,2,2,padding='same',name='POOL2')

# 第三层全连接层 8x8x64 to 1024  , fmaps = 1024
flat = tf.reshape(POOL2,[-1,8*8*64]) # 平铺特征图
DENSE1 = tf.layers.dense(flat,1024,activation=tf.nn.relu,
                         kernel_initializer=tf.random_normal_initializer(0,0.1),
                         name='DENSE1')
# dropout
drop_rate = tf.placeholder(dtype=tf.float32,name='drop_rate')
DP = tf.layers.dropout(DENSE1,rate=drop_rate,name='DROPOUT')

# softmax 1024 to 10 , fmaps = 10
DENSE2 = tf.layers.dense(DP,10,activation=tf.nn.softmax,
                         kernel_initializer=tf.random_normal_initializer(0,0.1),
                         name='DENSE2')
# 定义损失函数
cross_entropy = -tf.reduce_sum(y_ * tf.log(DENSE2)) #计算交叉熵

# 梯度下降
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy) #使用adam优化器来以0.0001的学习率来进行微调

# 准确率测试
correct_prediction = tf.equal(tf.argmax(DENSE2,1), tf.argmax(y_,1)) #判断预测标签和实际标签是否匹配
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

# 保存模型
saver = tf.train.Saver(var_list=[var for var in tf.trainable_variables()])

# 数据集读取
# 读取TFR,不打乱文件顺序，指定数据类型，开启多线程
[data,label,num] = TFRtools.ReadFromTFRecord(sameName= r'.\TFR\CODE_TRAIN-*',isShuffle= False,datatype= tf.float64,labeltype= tf.uint8,)
# 批量处理，送入队列数据，指定数据大小，不打乱数据项，设置批次大小32
[data_batch,label_batch,num_batch] = TFRtools.DataBatch(data,label,num,dataSize= 32*32,labelSize= 10,isShuffle= False,batchSize= 32)
# 修改格式
data_batch = tf.cast(tf.reshape(data_batch,[-1,32,32,1]),tf.float32)
label_batch = tf.cast(label_batch,tf.float32)


ACC = []

# 建立会话
with tf.Session() as sess:

    # 创建相关目录
    GEN_DIR()

    # 初始化变量
    init = (tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)

    # 开启协调器
    coord = tf.train.Coordinator()
    # 启动线程
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for step in range(5001):

        # 获取批数据
        TDB = sess.run([data_batch,label_batch,num_batch])


        # 训练
        sess.run(train_step,feed_dict={x:TDB[0],
                                       y_:TDB[1],
                                       drop_rate:0.5})
        # 测试
        if step%5 == 1:
            train_acc = sess.run(accuracy,feed_dict={x:TDB[0],
                                                     y_:TDB[1],
                                                     drop_rate:0.0})
            ACC.append(train_acc)
            print('迭代次数：%d..准确率：%.3f'%(step,train_acc))

        # 保存模型
        if step % 1000 == 0 and step!=0:
            saver.save(sess, './ckpt/CNN.ckpt', global_step=step)

    # 关闭线程
    coord.request_stop()
    coord.join(threads)

#  保存loss记录
with open('./trainLog/ACC.acc', 'wb') as a:
    losses = np.array(ACC)
    pickle.dump(losses,a)
    print('保存accuracy信息..')

# 显示loss
import matplotlib.pyplot as plt
with open(r'./trainLog/ACC.acc','rb') as a:
    acc = pickle.load(a)
    acc = np.array(acc)
    fig, ax = plt.subplots(figsize=(20, 7))
    plt.plot(acc, label=' ACC')
    plt.title("Training ACC")
    plt.legend()
    plt.show()


