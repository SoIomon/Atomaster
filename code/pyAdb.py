import os
import time

from PIL import Image



def connect_device():
    os.system("adb devices")
    os.system("adb connect 127.0.0.1:62001")


def execute(cmd):
    os.system(cmd)


def screen_cap(img_name='tmp'):
    execute(f"adb exec-out screencap -p > ../img_tmp/{img_name}.png")
    img = Image.open(f'../img_tmp/{img_name}.png')
    a = img.rotate(-90, expand=True)
    # a.show()
    a.save(f'../img_tmp/{img_name}.png')


def click(x, y):
    execute(f"adb shell input tap {x} {y}")
    time.sleep(0.5)

def swipe(x1, y1, x2, y2):
    execute(f"adb shell input swipe {x1} {y1} {x2} {y2}")


if __name__ == '__main__':
    # connect_device()
    screen_cap()
