
from src.gen_model import create_layer
import tensorflow as tf
import src.config as config
import numpy as np
from src.gen_image import convert2gray, gen_random_captcha_image, array_to_text


def crack_captcha(captcha_image):
    x_input = tf.placeholder(tf.float32, [None, config.IMAGE_HEIGHT * config.IMAGE_WIDTH])
    keep_prob = tf.placeholder(tf.float32)  # dropout
    output = create_layer(x_input, keep_prob)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        predict = tf.argmax(tf.reshape(output, [-1, config.MAX_CAPTCHA, config.CHAR_SET_LEN]), 2)
        text_list = sess.run(predict, feed_dict={x_input: [captcha_image], keep_prob: 1})

        text = text_list[0].tolist()
        vector = np.zeros(config.MAX_CAPTCHA * config.CHAR_SET_LEN)
        i = 0
        for n in text:
            vector[i * config.CHAR_SET_LEN + n] = 1
            i += 1

        return array_to_text(vector)


def validate_image():
    text, image = gen_random_captcha_image()
    image = convert2gray(image)

    # map the value to 0 -> 1 ，this will really affect the loss function update ,always remember the value should
    # suit to the loss learning rate
    # refer https://www.youtube.com/watch?v=pU5TG_X7b6E&index=6&list=PLwY2GJhAPWRcZxxVFpNhhfivuW0kX15yG 21:00
    image = image.flatten() / 255
    predict_text = crack_captcha(image)
    print("label is : {} <----> predict is : {}".format(text, predict_text))


if __name__ == '__main__':
    validate_image()
