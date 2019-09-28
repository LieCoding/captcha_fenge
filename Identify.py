"""
该脚本用于在线识别一张验证码
"""
import tensorflow as tf
from ImageProcess import *
from PreProcess import Segment4_NUMBER

def CNN_Identify(numbers):

    with tf.Session() as sess:

        # 初始化变量
        init = (tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)

        # 恢复模型
        meta_graph = tf.train.import_meta_graph('./ckpt/CNN.ckpt-5000.meta')  # 加载模型
        meta_graph.restore(sess, tf.train.latest_checkpoint('./ckpt'))  # 加载数据

        # 获取输入输出
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")  # 获取输入占位符x
        drop_rate = graph.get_tensor_by_name("drop_rate:0")  # 获取输入占位符drop_rate
        DENSE2 = graph.get_tensor_by_name("DENSE2/Softmax:0")  # 获取输出DENSE2

        id = []
        for number in numbers:
            # 格式修正
            number = np.reshape(number,[-1,32,32,1]).astype(np.float32)
            # 识别
            y = sess.run(DENSE2, feed_dict={x: number, drop_rate: 0.0})
            # 记录识别结果
            id.append(np.argmax(y))

        return id



code_name = r'./TEST/0243.jpg'

##(1)分割数字
img_gray = cv2.imread(code_name, flags=cv2.IMREAD_GRAYSCALE) # 读取灰度图像
SHOW('code',img_gray) # 显示
_, img_bin = cv2.threshold(img_gray, int(0.9 * 255), 255, cv2.THRESH_BINARY_INV) # 二值化（0.9）
numbers = Segment4_NUMBER(img_bin)# 分割数字

##(2)识别
ID = CNN_Identify(numbers)

print('验证码%s的识别结果为%s'%(code_name,ID))
cv2.waitKey(0)

