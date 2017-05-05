from PIL import Image
import cv2
from captcha.image import ImageCaptcha


image = ImageCaptcha(width=160, height=60,font_sizes=[35])

# data = image.generate('123abc') # 生成numpy格式，不保存为本地图片
image.write('123abc', 'out.png')  # 保存为本地图片
# X=cv2.imread("out.png")
# print(X)
# captcha_image = Image.open(X)
# captcha_source = np.array(captcha_image)
# print(captcha_source)

