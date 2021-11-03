# -*- coding:gb2312 -*-
# -*- coding:UTF-8 -*-
# @Time     :2021 11 2021/11/3 20:03
# @Author   :千乘

import torch
from PIL import Image
import numpy as np
import pyautogui
import cv2

'''
说明：
本类作用：判断是否处于钓上来地状态
predict函数输入图片地址（已有默认分类器模型地址，如需修改自便），输出钓鱼的三种状态：钓上，鱼竿，鱼钩
if_bite函数 一旦运行就进入判断是否处于钓上状态的循环模式，返回True表示钓上.
默认的储存截图的位置为 imgs/whetherbite.png，可以通过cap_path修改
'''


class JudgeBite(object):
    def __init__(self, classifer='classifer_model.pt', cap_path='imgs/whetherbite.png'):
        self.cap_path = cap_path
        self.classifer = classifer

    @staticmethod
    def process_image(image):

        pic = Image.open(image)
        if pic.size[0] < pic.size[1]:
            ratio = float(256) / float(pic.size[0])
        else:
            ratio = float(256) / float(pic.size[1])

        new_size = (int(pic.size[0] * ratio), int(pic.size[1] * ratio))

        pic.thumbnail(new_size)

        pic = pic.crop([pic.size[0] / 2 - 112, pic.size[1] / 2 - 112, pic.size[0] / 2 + 112, pic.size[1] / 2 + 112])

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        np_image = np.array(pic)
        np_image = np_image / 255

        for i in range(2):
            np_image[:, :, i] -= mean[i]
            np_image[:, :, i] /= std[i]

        np_image = np_image.transpose((2, 0, 1))
        np_image = torch.from_numpy(np_image)
        np_image = np_image.float()
        print(np_image.type)
        return np_image

    def predict(self, image_path):
        model = torch.load(self.classifer)
        img = self.process_image(image_path)
        img = img.unsqueeze(0)
        result = model(img).topk(1)

        classes = result[1]

        return classes.numpy()[0][0]

    def if_bite(self):
        flag = 0
        while flag != 1:  # 这个flag就是判断predict出来的东西是否是钓上鱼了
            img = pyautogui.screenshot(region=[2400, 1550, 300, 200])  # x,y,w,h
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            cv2.imwrite(self.cap_path, img)
            flag = self.predict(self.cap_path)
            num += 1
        return True
