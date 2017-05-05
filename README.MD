# demo to show how to crack the captcha images 

python==python3.5.2
* first please make sure you already install lib as the requirements list.
* just run the `train.py` script ,i will train and save the model 

# Architecture

# Usage

## 训练
首先运行train.py文件，自动生成图片用于训练，并将模型保存至model文件夹中：
```
   python3 train.py
```

## 生成测试图片
运行test_image_generator.py，用captcha包生成一张用于测试的验证码图片
```
   python3 test_image_generator.py
```

## 破解验证码
运行crack.py文件，破解上一步生成的验证码