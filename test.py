"""
该脚本用于验证测试集的准确率
"""

import tensorflow as tf
import numpy as np

#*************************** CNN识别 *******************************

# 读取测试集
import TFRtools
[data,label,num] = TFRtools.ReadFromTFRecord(sameName= r'.\TFR\CODE_TEST-*',isShuffle= False,datatype= tf.float64,labeltype= tf.uint8,)
# 单个验证不采用批处理
# [data_batch,label_batch,num_batch] = TFRtools.DataBatch(data,label,num,dataSize= 32*32,labelSize= 10,isShuffle= False,batchSize= 32)

# 数据格式修正
data = tf.cast(tf.reshape(data,[-1,32,32,1]),tf.float32)
label = tf.cast(label,tf.float32)

with tf.Session() as sess:

    # 初始化变量
    init = (tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)

    # 开启协调器
    coord = tf.train.Coordinator()
    # 启动线程
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # 恢复模型
    meta_graph = tf.train.import_meta_graph('./ckpt/CNN.ckpt-5000.meta')  # 加载模型
    meta_graph.restore(sess, tf.train.latest_checkpoint('./ckpt'))  # 加载数据

    # 获取输入输出
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")  # 获取输入占位符x
    drop_rate = graph.get_tensor_by_name("drop_rate:0") # 获取输入占位符drop_rate
    DENSE2 = graph.get_tensor_by_name("DENSE2/Softmax:0")# 获取输出DENSE2

    # 验证码识别
    test_total = 2583 # 测试验证码数量
    NUM_SC = [] # 数字识别情况
    ID_SC = [] # 测试集验证码识别情况
    for N in range(test_total):
        score = 0 # 每组验证码的验证成绩，满分4分
        for idx in range(4):
            # 获取数据集
            TD =sess.run([data,label,num])
            # 识别
            y = sess.run(DENSE2,feed_dict={x:TD[0],drop_rate:0.0})
            # 比较
            comp = np.equal(np.argmax(y),np.argmax(TD[1]))
            NUM_SC.append(comp)
            # 计分
            if comp:
                score +=1
            if idx ==3 :# 完成识别一组验证码
                ID_SC.append(score)
                score = 0 # 识别后清零

            # 打印
            print('第%d张验证码的第%d位识别情况：%s'%(N,idx,comp))

    # 统计识别率
    # （1）数字识别率
    print('数字识别率为：%.3f'%(sum(NUM_SC)/len(NUM_SC)))
    #  (2) 验证码识别率
    SC = 0 # 测试集识别总分
    for sc in ID_SC:
        if sc == 4:
            SC +=1
    print('验证码识别率为：%.3f' % (SC/len(ID_SC)))


    # 关闭线程
    coord.request_stop()
    coord.join(threads)





