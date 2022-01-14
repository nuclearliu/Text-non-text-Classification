#!/usr/bin/env python3
# coding = utf-8

import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
import numpy as np

from ResNet18 import ResNet18


model = ResNet18([2, 2, 2, 2])

checkpoint_save_path = "./TextDis_benchmark/ResNet18.ckpt"
model.load_weights(checkpoint_save_path)

out_path = "./output_images/"
text_image_count = 0
non_text_image_count = 0
img_path = input("\x1b[0mPath: ")
img_path = img_path.replace("\\","")
img_path = img_path.strip()
while img_path not in ["quit", "q"]:
    img_orig = Image.open(img_path)  # 读入图片
    # x_slices = (img_orig.size[0] - 1) // 224 + 1
    # y_slices = (img_orig.size[1] - 1) // 224 + 1
    # img_resized = Image.new('RGB', (x_slices * 224, y_slices * 224), (255, 255, 255))
    # img_resized.paste(img_orig, (0, 0))
    img = img_orig.resize((224, 224))
    img = np.array(img.convert("L"))
    # parts = []
    # for j in range(y_slices):
    #     for i in range(x_slices):
    #         box = (224 * i, 224 * j, 224 * (i + 1), 224 * (j + 1))
    #         part = img_resized.crop(box)
    #         part = np.array(part.convert("L"))
    #         part = part / 255.
    #         parts.append(part)
            # plt.imshow(part)
            # plt.show()
    img = img[tf.newaxis, ...]
    # parts = np.array(parts)
    # parts = np.reshape(parts, (len(parts), 224, 224, 1))
    img = np.reshape(img, (1, 224, 224, 1))
    img = img / 255.
    result = model.predict(img)
    print(result)
    result = tf.argmax(result, axis=1)
    # result = set(result)
    # isText = True
    # if len(result) == 1 and not result[0]:
    #     isText = False
    img_out = Image.new('RGB', (img_orig.size[0], img_orig.size[1] + 35), (20, 136, 173))
    img_out.paste(img_orig, (0, 0))
    font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 20)
    draw = ImageDraw.Draw(img_out)
    if result:
        text_image_count += 1
        print("\x1b[35mtext detected")
        draw.text((10, img_orig.size[1] + 5), "text detected", (255, 255, 255), font=font)
        img_out.save(out_path + "text/" + str(text_image_count) + ".jpg")
    else:
        non_text_image_count += 1
        print("\x1b[35mtext not detected")
        draw.text((10, img_orig.size[1] + 5), "text not detected", (255, 255, 255), font=font)
        img_out.save(out_path + "nonText/" + str(non_text_image_count) + ".jpg")
    # plt.imshow(img_out)
    # plt.show()
    img_path = input("\x1b[0mPath: ")
    img_path = img_path.replace("\\", "")
    img_path = img_path.strip()