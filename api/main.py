from ISORT_inference import ISORT
import copy
import datetime
from motor.motor_asyncio import AsyncIOMotorClient
import torch
import threading
from concurrent.futures import ThreadPoolExecutor
import asyncio
import time
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import uvicorn
from fastapi import FastAPI


loop = asyncio.get_event_loop()
mongodb_uri = 'mongodb://db:27017/'
db_name = "isort"
client = AsyncIOMotorClient(mongodb_uri)
db = client[db_name]


async def async_insert(data):
    res = await db.predict_log.insert_one(data)
    return res


app = FastAPI()
mutex_lock = asyncio.Lock()
thread_mutex_lock = threading.Lock()
save_image: bool = False
time_now = 0
releast_time = 15


class model():
    def __init__(self):
        self.model = None
        self.last_time = 0

    def get_time(self):
        return self.last_time


maximum_model = 5

isort1 = model()
isort1.model = None
isort1.last_time = 0

isort2 = model()
isort2.model = None
isort2.last_time = 0

isort3 = model()
isort3.model = None
isort3.last_time = 0

isort4 = model()
isort4.model = None
isort4.last_time = 0

isort5 = model()
isort5.model = None
isort5.last_time = 0

all_model_ml = []


def print_all_model():
    for i in range(len(all_model_ml)):
        print(all_model_ml[i].model, flush=True)
        print(all_model_ml[i].last_time, flush=True)
    check_cuda_or_cpu = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"cuda : {check_cuda_or_cpu}")
    print("{} Model allocate memory : {} Gb".format(
        len(all_model_ml), torch.cuda.memory_allocated()/1024**3), flush=True)



def releast_by_time(thread_lock=None):
    global all_model_ml
    with thread_lock:
        for i in range(len(all_model_ml)):
            if (time_now - all_model_ml[i].last_time) > releast_time and all_model_ml[i].model != None:
                all_model_ml[i].model.release_memory()
                all_model_ml[i].model = None
                all_model_ml.pop(i)


def interupt_timer_check_model():
    global time_now, all_model_ml
    with ThreadPoolExecutor() as exe:
        _ = exe.submit(releast_by_time, thread_mutex_lock)
        _ = exe.submit(print_all_model)
    time_now = time.time()
    threading.Timer(5.0, interupt_timer_check_model).start()


interupt_timer_check_model()


class data_input(BaseModel):
    image: bytes = None
    criteria_defect_mole_all: float = 0
    criteria_defect_mole_instant: float = 0
    criteria_defect_pad: float = 0


@app.get("/")
def hello():
    return "Hi server is working"


@app.post("/predict_test1")
async def pred(request: data_input):
    global isort1, all_model_ml
    isort1.last_time = time.time()
    if isort1.model == None:
        if len(all_model_ml) == maximum_model and all_model_ml[0] != None:
            all_model_ml[0].model.release_memory()
            all_model_ml[0].model = None
            all_model_ml.pop(0)
        isort1.model = ISORT(scale_img_devide=2, HALF=False)
        all_model_ml.append(isort1)
        all_model_ml = sorted(all_model_ml, key=model.get_time)

    try:
        isort1.model.decode_base64(request.image)
    except:
        error_convert_image_base64: dict = {
            "status": "Error can not byte to image please recheck your image in json package"}
        return JSONResponse(status_code=400, content=error_convert_image_base64)

    image_base64: bytes = request.image
    criteria_defect_mole_all: float = request.criteria_defect_mole_all
    criteria_defect_mole_instant: float = request.criteria_defect_mole_instant
    criteria_defect_pad: float = request.criteria_defect_pad

    try:
        start_time: float = time.time()
        crop: list[int] = [675, 675+653, 609, 609+785]
        crop: list[int] = []
        # if use model U-Net_256 set [scale_img_devide] to 2
        # if use model U-Net_512 set [scale_img_devide] to 1
        image_ori_with_pred, defect_mole_per_bg, defect_pad_per_pads, defect_mole_instant_per_bgs = await isort1.model.predict(image_base64=image_base64, mode="deploy", crop_at=crop, mutex_lock=mutex_lock)

        defect_mole_per_bg_flag: bool = defect_mole_per_bg > criteria_defect_mole_all
        defect_pad_per_pad_flag: list[bool] = [
            defect_pad_per_pad > criteria_defect_pad for defect_pad_per_pad in defect_pad_per_pads]
        defect_mole_instant_per_bg_flag: list[bool] = [
            defect_mole_instant_per_bg > criteria_defect_mole_instant for defect_mole_instant_per_bg in defect_mole_instant_per_bgs]

        pred: dict = {
            "defect_mole_per_bg": defect_mole_per_bg,
            "defect_pad_per_pad": defect_pad_per_pads,
            "defect_mole_instant_per_bg": defect_mole_instant_per_bgs,
            "defect_mole_per_bg_flag": defect_mole_per_bg_flag,
            "defect_pad_per_pad_flag": defect_pad_per_pad_flag,
            "defect_mole_instant_bg_flag": defect_mole_instant_per_bg_flag,
            "time_log": str(datetime.datetime.utcnow()),
        }
        if save_image:
            path = r"E:\\Work\\Project\\ISORT U-Net\\save_image"
            name = str(defect_mole_per_bg)
            with ThreadPoolExecutor() as exe:
                _ = exe.submit(isort1.model.save_image, path, name +
                               ".png", image_ori_with_pred)
        end_time: float = time.time()
        print(f"Time taken per image: {end_time-start_time}")
        pred_log = copy.deepcopy(pred)
        await async_insert(pred_log)

        return JSONResponse(status_code=200, content=pred)
    except:
        error_predict: dict = {
            "status": "Error at predict function please report to AUTOMATION Team",
            "time_log": str(datetime.datetime.utcnow()),
        }
        error_predict_log = copy.deepcopy(error_predict)
        await async_insert(error_predict_log)
        return JSONResponse(status_code=500, content=error_predict)


@app.post("/predict_test2")
async def pred(request: data_input):
    global isort2, all_model_ml
    isort2.last_time = time.time()
    if isort2.model == None:
        if len(all_model_ml) == maximum_model and all_model_ml[0] != None:
            all_model_ml[0].model.release_memory()
            all_model_ml[0].model = None
            all_model_ml.pop(0)
        isort2.model = ISORT(scale_img_devide=2, HALF=False)
        all_model_ml.append(isort2)
        all_model_ml = sorted(all_model_ml, key=model.get_time)

    try:
        isort2.model.decode_base64(request.image)
    except:
        error_convert_image_base64: dict = {
            "status": "Error can not byte to image please recheck your image in json package"}
        return JSONResponse(status_code=400, content=error_convert_image_base64)

    image_base64: bytes = request.image
    criteria_defect_mole_all: float = request.criteria_defect_mole_all
    criteria_defect_mole_instant: float = request.criteria_defect_mole_instant
    criteria_defect_pad: float = request.criteria_defect_pad

    try:
        start_time: float = time.time()
        crop: list[int] = [675, 675+653, 609, 609+785]
        crop: list[int] = []
        # if use model U-Net_256 set [scale_img_devide] to 2
        # if use model U-Net_512 set [scale_img_devide] to 1
        image_ori_with_pred, defect_mole_per_bg, defect_pad_per_pads, defect_mole_instant_per_bgs = await isort2.model.predict(image_base64=image_base64, mode="deploy", crop_at=crop, mutex_lock=mutex_lock)

        defect_mole_per_bg_flag: bool = defect_mole_per_bg > criteria_defect_mole_all
        defect_pad_per_pad_flag: list[bool] = [
            defect_pad_per_pad > criteria_defect_pad for defect_pad_per_pad in defect_pad_per_pads]
        defect_mole_instant_per_bg_flag: list[bool] = [
            defect_mole_instant_per_bg > criteria_defect_mole_instant for defect_mole_instant_per_bg in defect_mole_instant_per_bgs]

        pred: dict = {
            "defect_mole_per_bg": defect_mole_per_bg,
            "defect_pad_per_pad": defect_pad_per_pads,
            "defect_mole_instant_per_bg": defect_mole_instant_per_bgs,
            "defect_mole_per_bg_flag": defect_mole_per_bg_flag,
            "defect_pad_per_pad_flag": defect_pad_per_pad_flag,
            "defect_mole_instant_bg_flag": defect_mole_instant_per_bg_flag,
        }
        if save_image:
            path = r"E:\\Work\\Project\\ISORT U-Net\\save_image"
            name = str(defect_mole_per_bg)
            with ThreadPoolExecutor() as exe:
                _ = exe.submit(isort2.model.save_image, path, name +
                               ".png", image_ori_with_pred)
        end_time: float = time.time()
        print(f"Time taken per image: {end_time-start_time}")

        return JSONResponse(status_code=200, content=pred)
    except:
        error_predict: dict = {
            "status": "Error at predict function please report to AUTOMATION Team"
        }
        return JSONResponse(status_code=500, content=error_predict)


@app.post("/predict_test3")
async def pred(request: data_input):
    global isort3, all_model_ml
    isort3.last_time = time.time()
    if isort3.model == None:
        if len(all_model_ml) == maximum_model and all_model_ml[0] != None:
            all_model_ml[0].model.release_memory()
            all_model_ml[0].model = None
            all_model_ml.pop(0)
        isort3.model = ISORT(scale_img_devide=2, HALF=False)
        all_model_ml.append(isort3)
        all_model_ml = sorted(all_model_ml, key=model.get_time)

    try:
        isort3.model.decode_base64(request.image)
    except:
        error_convert_image_base64: dict = {
            "status": "Error can not byte to image please recheck your image in json package"}
        return JSONResponse(status_code=400, content=error_convert_image_base64)

    image_base64: bytes = request.image
    criteria_defect_mole_all: float = request.criteria_defect_mole_all
    criteria_defect_mole_instant: float = request.criteria_defect_mole_instant
    criteria_defect_pad: float = request.criteria_defect_pad

    try:
        start_time: float = time.time()
        crop: list[int] = [675, 675+653, 609, 609+785]
        crop: list[int] = []
        # if use model U-Net_256 set [scale_img_devide] to 2
        # if use model U-Net_512 set [scale_img_devide] to 1
        image_ori_with_pred, defect_mole_per_bg, defect_pad_per_pads, defect_mole_instant_per_bgs = await isort3.model.predict(image_base64=image_base64, mode="deploy", crop_at=crop, mutex_lock=mutex_lock)

        defect_mole_per_bg_flag: bool = defect_mole_per_bg > criteria_defect_mole_all
        defect_pad_per_pad_flag: list[bool] = [
            defect_pad_per_pad > criteria_defect_pad for defect_pad_per_pad in defect_pad_per_pads]
        defect_mole_instant_per_bg_flag: list[bool] = [
            defect_mole_instant_per_bg > criteria_defect_mole_instant for defect_mole_instant_per_bg in defect_mole_instant_per_bgs]

        pred: dict = {
            "defect_mole_per_bg": defect_mole_per_bg,
            "defect_pad_per_pad": defect_pad_per_pads,
            "defect_mole_instant_per_bg": defect_mole_instant_per_bgs,
            "defect_mole_per_bg_flag": defect_mole_per_bg_flag,
            "defect_pad_per_pad_flag": defect_pad_per_pad_flag,
            "defect_mole_instant_bg_flag": defect_mole_instant_per_bg_flag,
        }
        if save_image:
            path = r"E:\\Work\\Project\\ISORT U-Net\\save_image"
            name = str(defect_mole_per_bg)
            with ThreadPoolExecutor() as exe:
                _ = exe.submit(isort3.model.save_image, path, name +
                               ".png", image_ori_with_pred)
        end_time: float = time.time()
        print(f"Time taken per image: {end_time-start_time}")

        return JSONResponse(status_code=200, content=pred)
    except:
        error_predict: dict = {
            "status": "Error at predict function please report to AUTOMATION Team"
        }
        return JSONResponse(status_code=500, content=error_predict)


@app.post("/predict_test4")
async def pred(request: data_input):
    global isort4, all_model_ml
    isort4.last_time = time.time()
    if isort4.model == None:
        if len(all_model_ml) == maximum_model and all_model_ml[0] != None:
            all_model_ml[0].model.release_memory()
            all_model_ml[0].model = None
            all_model_ml.pop(0)
        isort4.model = ISORT(scale_img_devide=2, HALF=False)
        all_model_ml.append(isort4)
        all_model_ml = sorted(all_model_ml, key=model.get_time)

    try:
        isort4.model.decode_base64(request.image)
    except:
        error_convert_image_base64: dict = {
            "status": "Error can not byte to image please recheck your image in json package"}
        return JSONResponse(status_code=400, content=error_convert_image_base64)

    image_base64: bytes = request.image
    criteria_defect_mole_all: float = request.criteria_defect_mole_all
    criteria_defect_mole_instant: float = request.criteria_defect_mole_instant
    criteria_defect_pad: float = request.criteria_defect_pad

    try:
        start_time: float = time.time()
        crop: list[int] = [675, 675+653, 609, 609+785]
        crop: list[int] = []
        # if use model U-Net_256 set [scale_img_devide] to 2
        # if use model U-Net_512 set [scale_img_devide] to 1
        image_ori_with_pred, defect_mole_per_bg, defect_pad_per_pads, defect_mole_instant_per_bgs = await isort4.model.predict(image_base64=image_base64, mode="deploy", crop_at=crop, mutex_lock=mutex_lock)

        defect_mole_per_bg_flag: bool = defect_mole_per_bg > criteria_defect_mole_all
        defect_pad_per_pad_flag: list[bool] = [
            defect_pad_per_pad > criteria_defect_pad for defect_pad_per_pad in defect_pad_per_pads]
        defect_mole_instant_per_bg_flag: list[bool] = [
            defect_mole_instant_per_bg > criteria_defect_mole_instant for defect_mole_instant_per_bg in defect_mole_instant_per_bgs]

        pred: dict = {
            "defect_mole_per_bg": defect_mole_per_bg,
            "defect_pad_per_pad": defect_pad_per_pads,
            "defect_mole_instant_per_bg": defect_mole_instant_per_bgs,
            "defect_mole_per_bg_flag": defect_mole_per_bg_flag,
            "defect_pad_per_pad_flag": defect_pad_per_pad_flag,
            "defect_mole_instant_bg_flag": defect_mole_instant_per_bg_flag,
        }
        if save_image:
            path = r"E:\\Work\\Project\\ISORT U-Net\\save_image"
            name = str(defect_mole_per_bg)
            with ThreadPoolExecutor() as exe:
                _ = exe.submit(isort4.model.save_image, path, name +
                               ".png", image_ori_with_pred)
        end_time: float = time.time()
        print(f"Time taken per image: {end_time-start_time}")

        return JSONResponse(status_code=200, content=pred)
    except:
        error_predict: dict = {
            "status": "Error at predict function please report to AUTOMATION Team"
        }
        return JSONResponse(status_code=500, content=error_predict)


@app.post("/predict_test5")
async def pred(request: data_input):
    global isort5, all_model_ml
    isort5.last_time = time.time()
    if isort5.model == None:
        if len(all_model_ml) == maximum_model and all_model_ml[0] != None:
            all_model_ml[0].model.release_memory()
            all_model_ml[0].model = None
            all_model_ml.pop(0)
        isort5.model = ISORT(scale_img_devide=2, HALF=False)
        all_model_ml.append(isort5)
        all_model_ml = sorted(all_model_ml, key=model.get_time)

    try:
        isort5.model.decode_base64(request.image)
    except:
        error_convert_image_base64: dict = {
            "status": "Error can not byte to image please recheck your image in json package"}
        return JSONResponse(status_code=400, content=error_convert_image_base64)

    image_base64: bytes = request.image
    criteria_defect_mole_all: float = request.criteria_defect_mole_all
    criteria_defect_mole_instant: float = request.criteria_defect_mole_instant
    criteria_defect_pad: float = request.criteria_defect_pad

    try:
        start_time: float = time.time()
        crop: list[int] = [675, 675+653, 609, 609+785]
        crop: list[int] = []
        # if use model U-Net_256 set [scale_img_devide] to 2
        # if use model U-Net_512 set [scale_img_devide] to 1
        image_ori_with_pred, defect_mole_per_bg, defect_pad_per_pads, defect_mole_instant_per_bgs = await isort5.model.predict(image_base64=image_base64, mode="deploy", crop_at=crop, mutex_lock=mutex_lock)

        defect_mole_per_bg_flag: bool = defect_mole_per_bg > criteria_defect_mole_all
        defect_pad_per_pad_flag: list[bool] = [
            defect_pad_per_pad > criteria_defect_pad for defect_pad_per_pad in defect_pad_per_pads]
        defect_mole_instant_per_bg_flag: list[bool] = [
            defect_mole_instant_per_bg > criteria_defect_mole_instant for defect_mole_instant_per_bg in defect_mole_instant_per_bgs]

        pred: dict = {
            "defect_mole_per_bg": defect_mole_per_bg,
            "defect_pad_per_pad": defect_pad_per_pads,
            "defect_mole_instant_per_bg": defect_mole_instant_per_bgs,
            "defect_mole_per_bg_flag": defect_mole_per_bg_flag,
            "defect_pad_per_pad_flag": defect_pad_per_pad_flag,
            "defect_mole_instant_bg_flag": defect_mole_instant_per_bg_flag,
        }
        if save_image:
            path = r"E:\\Work\\Project\\ISORT U-Net\\save_image"
            name = str(defect_mole_per_bg)
            with ThreadPoolExecutor() as exe:
                _ = exe.submit(isort5.model.save_image, path, name +
                               ".png", image_ori_with_pred)
        end_time: float = time.time()
        print(f"Time taken per image: {end_time-start_time}")

        return JSONResponse(status_code=200, content=pred)
    except:
        error_predict: dict = {
            "status": "Error at predict function please report to AUTOMATION Team"
        }
        return JSONResponse(status_code=500, content=error_predict)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)
