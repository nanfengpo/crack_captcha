
import tensorflow as tf
import numpy as np
from src.gen_image import text_to_array
from src.config import MAX_CAPTCHA, CHAR_SET_LEN, IMAGE_HEIGHT, IMAGE_WIDTH, MAX_ACCURACY
from src.gen_image import gen_require_captcha_image

x_input = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
y_input = tf.placeholder(tf.float32, [None, CHAR_SET_LEN * MAX_CAPTCHA])
keep_prob = tf.placeholder(tf.float32)


# 把彩色图像转为灰度图像（色彩对识别验证码没有什么用,对于抽取特征也没啥用）
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        return gray
    else:
        return img


def __weight_variable(shape, stddev=0.01):
    initial = tf.random_normal(shape, stddev=stddev)
    return tf.Variable(initial)


def __bias_variable(shape, stddev=0.1):
    initial = tf.random_normal(shape=shape, stddev=stddev)
    return tf.Variable(initial)


def __conv2d(x, w):
    # strides 代表移动的平长
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def __max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 100个一个批次
def gen_next_batch(batch_size=100):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])

    for i in range(batch_size):
        text, image = gen_require_captcha_image()

        # 转成灰度图片，因为颜色对于提取字符形状是没有意义的
        image = convert2gray(image)

        batch_x[i, :] = image.flatten() / 255
        batch_y[i, :] = text_to_array(text)

    return batch_x, batch_y


def create_layer(x_input, keep_prob):
    x_image = tf.reshape(x_input, shape=[-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1])

    # 定义第1个卷积层
    w_c1 = __weight_variable([5, 5, 1, 32], stddev=0.1)  # 3x3 第一层32个卷积核 采用黑白色
    b_c1 = __bias_variable([32], stddev=0.1)
    h_c1 = tf.nn.relu(tf.nn.bias_add(__conv2d(x_image, w_c1), b_c1))  # 定义第一个卷积层
    h_pool1 = __max_pool_2x2(h_c1)  # 定义第一个池化层
    # h_pool1 = tf.nn.dropout(h_pool1, keep_prob)

    # 定义第2个卷积层
    w_c2 = __weight_variable([5, 5, 32, 64], stddev=0.1)
    b_c2 = __bias_variable([64], stddev=0.1)
    h_c2 = tf.nn.relu(tf.nn.bias_add(__conv2d(h_pool1, w_c2), b_c2))
    h_pool2 = __max_pool_2x2(h_c2)
    # h_pool2 = tf.nn.dropout(h_pool2, keep_prob)

    # 定义第3个卷积层
    w_c3 = __weight_variable([5, 5, 64, 64], stddev=0.1)
    b_c3 = __bias_variable([64], stddev=0.1)
    h_c3 = tf.nn.relu(tf.nn.bias_add(__conv2d(h_pool2, w_c3), b_c3))
    h_pool3 = __max_pool_2x2(h_c3)
    # h_pool3 = tf.nn.dropout(h_pool3, keep_prob)

    # 3层池化之后 width 144 / 8 = 18
    # height 64 / 8 = 8

    # 全链接层1
    w_fc1 = __weight_variable([20 * 8 * 64, 1024], stddev=0.1)
    b_fc1 = __bias_variable([1024])
    h_pool3_flat = tf.reshape(h_pool3, [-1, w_fc1.get_shape().as_list()[0]])
    h_fc1 = tf.nn.relu(tf.add(tf.matmul(h_pool3_flat, w_fc1), b_fc1))
    # drop out 内容0
    h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob)

    # 全链接层2
    w_output = __weight_variable([1024, MAX_CAPTCHA * CHAR_SET_LEN], stddev=0.1)
    b_output = __bias_variable([MAX_CAPTCHA * CHAR_SET_LEN])
    y_output = tf.add(tf.matmul(h_fc1_dropout, w_output), b_output)

    return y_output

# 计算loss的典型方法
def create_loss(layer, y_input):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_input, logits=layer))
    return loss

# 计算accuracy的典型方法
def create_accuracy(output, y_input):
    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(y_input, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy


def train():
    # create the layer and loss
    layer_output = create_layer(x_input, keep_prob)
    loss = create_loss(layer_output, y_input)
    accuracy = create_accuracy(layer_output, y_input)
    global_step_tensor=tf.Variable(0,trainable="Flase",name="global_step")
    train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss,global_step=global_step_tensor)
    # save model
    saver = tf.train.Saver()

    with tf.Session() as sess:

        tf.global_variables_initializer().run()
        acc = 0.0
        i = 0

        while acc < MAX_ACCURACY:
            i += 1
            batch_x, batch_y = gen_next_batch(64)
            _, _loss = sess.run([train_step, loss],
                                feed_dict={x_input: batch_x, y_input: batch_y, keep_prob: 0.5})

            # 每20次输出loss
            # tf.train.global_step(sess,global_step_tensor)等于i
            if i % 20 == 0:
                print(tf.train.global_step(sess,global_step_tensor), _loss)

            # 每100 step计算一次准确率并保存模型
            if i % 100 == 0:
                batch_x_test, batch_y_test = gen_next_batch(100)
                acc = sess.run(accuracy, feed_dict={x_input: batch_x_test, y_input: batch_y_test, keep_prob: 1.0})
                print('step is %s' % i, 'and accy is %s' % acc)
                # 保存模型
                saver.save(sess,"model/break.ckpt",global_step=i)
                # 如果准确率大于50%,完成训练
                if acc > MAX_ACCURACY:
                    print('current accuracy > %s  ,stop now' % MAX_ACCURACY)
                    break


if __name__ == '__main__':
    train()
