import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 嘗試路徑可不可以讀取圖片
main_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# main_path = r"c://Users//lulu3//Desktop//HW1_111061610"
image1 = cv2.cvtColor(cv2.imread(os.path.join(main_path, 'data', 'dog.bmp')), cv2.COLOR_BGR2RGB)
plt.figure("dog")
plt.imshow(image1)
plt.show()