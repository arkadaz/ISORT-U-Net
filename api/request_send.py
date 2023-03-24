import requests
import os
import base64
from PIL import Image
from io import BytesIO
from multiprocessing.pool import ThreadPool
from concurrent.futures import ThreadPoolExecutor
# http://localhost:8000/predict

current_path = os.getcwd()
image_path = r"{}\\test\\image\\14LD_LGA3X2.5_FJ8001_0110_WafFRAME #8_Col03Row27_0.bmp".format(
    current_path)
image = Image.open(image_path)
buff = BytesIO()
image.save(buff, format="JPEG")
image_base64 = base64.b64encode(buff.getvalue())


def request1(image_path):
    image = Image.open(image_path)
    buff = BytesIO()
    image.save(buff, format="JPEG")
    image_base64 = base64.b64encode(buff.getvalue())

    url = f"http://arkadaz.thddns.net:3991/predict_test1"

    data_send = {
        "image": image_base64.decode("utf-8"),
        "criteria_defect_mole": 5,
        "criteria_defect_pad": 30
    }
    response = requests.post(url, json=data_send)
    return response.text


def request2(image_path):
    image = Image.open(image_path)
    buff = BytesIO()
    image.save(buff, format="JPEG")
    image_base64 = base64.b64encode(buff.getvalue())

    url = f"http://arkadaz.thddns.net:3991/predict_test2"

    data_send = {
        "image": image_base64.decode("utf-8"),
        "criteria_defect_mole": 5,
        "criteria_defect_pad": 30
    }
    response = requests.post(url, json=data_send)
    return response.text


def request3(image_path):
    image = Image.open(image_path)
    buff = BytesIO()
    image.save(buff, format="JPEG")
    image_base64 = base64.b64encode(buff.getvalue())

    url = f"http://arkadaz.thddns.net:3991/predict_test3"

    data_send = {
        "image": image_base64.decode("utf-8"),
        "criteria_defect_mole": 5,
        "criteria_defect_pad": 30
    }
    response = requests.post(url, json=data_send)
    return response.text


def request4(image_path):
    image = Image.open(image_path)
    buff = BytesIO()
    image.save(buff, format="JPEG")
    image_base64 = base64.b64encode(buff.getvalue())

    url = f"http://arkadaz.thddns.net:3991/predict_test4"

    data_send = {
        "image": image_base64.decode("utf-8"),
        "criteria_defect_mole": 5,
        "criteria_defect_pad": 30
    }
    response = requests.post(url, json=data_send)
    return response.text


def request5(image_path):
    image = Image.open(image_path)
    buff = BytesIO()
    image.save(buff, format="JPEG")
    image_base64 = base64.b64encode(buff.getvalue())

    url = f"http://arkadaz.thddns.net:3991/predict_test5"

    data_send = {
        "image": image_base64.decode("utf-8"),
        "criteria_defect_mole": 5,
        "criteria_defect_pad": 30
    }
    response = requests.post(url, json=data_send)
    return response.text


def test():
    for i in range(1, 6):
        url = f"http://arkadaz.thddns.net:3991/predict_test{i}"

        data_send = {
            "image": image_base64.decode("utf-8"),
            "criteria_defect_mole": 5,
            "criteria_defect_pad": 30
        }
        response = requests.post(url, json=data_send)
        print(response.text)
    for i in range(30):
        url = f"http://arkadaz.thddns.net:3991/predict_test5"
        data_send = {
            "image": image_base64.decode("utf-8"),
            "criteria_defect_mole": 5,
            "criteria_defect_pad": 30
        }
        response = requests.post(url, json=data_send)
        print(response.text)


image_path_all = []
for i in range(100):
    image_path_all.append(image_path)
while True:
    with ThreadPool(100) as pool:
        for result in pool.imap_unordered(request1, image_path_all):
            print(result)

    with ThreadPool(100) as pool:
        for result in pool.imap_unordered(request2, image_path_all):
            print(result)
    with ThreadPool(100) as pool:
        for result in pool.imap_unordered(request3, image_path_all):
            print(result)
    with ThreadPool(100) as pool:
        for result in pool.imap_unordered(request4, image_path_all):
            print(result)
    with ThreadPool(100) as pool:
        for result in pool.imap_unordered(request5, image_path_all):
            print(result)


def test1():
    img_file_name = os.listdir(
        r"F:\\Utac Master Dataset\\ISORT\\Fail image bottom side_FJ8001_Isort maxx\\Contam on package (W)\\Over reject")
    path_img = r"F:\\Utac Master Dataset\\ISORT\\Fail image bottom side_FJ8001_Isort maxx\\Contam on package (W)\\Over reject"
    img_path = []
    for img_name in img_file_name:
        img_path.append(r"{}\\{}".format(path_img, img_name))

    def request(image_path):
        image = Image.open(image_path)
        buff = BytesIO()
        image.save(buff, format="JPEG")
        image_base64 = base64.b64encode(buff.getvalue())

        url = "http://localhost:8001/predict_test"

        data_send = {
            "image": image_base64.decode("utf-8"),
            "criteria_defect_mole": 5,
            "criteria_defect_pad": 30
        }
        response = requests.post(url, json=data_send)
        return response.text

    i = 0
    while True:
        with ThreadPool(100) as pool:
            for result in pool.imap_unordered(request, img_path):
                i += 1
                print(result)
                print(i)
