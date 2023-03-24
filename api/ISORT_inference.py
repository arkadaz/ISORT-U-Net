import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch.nn as nn
from PIL import Image
from io import BytesIO
import base64
from typing import List, Tuple
import matplotlib.pyplot as plt
import os
import asyncio
import time
import cv2


class ISORT:

    def __init__(self, scale_img_devide: int = 2, HALF: bool = False):
        self.HALF: bool = HALF
        self.DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.current_path: str = os.getcwd()
        self.PATH: str = "{}/U-Net_256.pt".format(self.current_path)
        self.scale_img_devide = scale_img_devide
        if self.HALF and self.DEVICE == "cuda":
            self.model_load: torch.jit._script.RecursiveScriptModule = ISORT.load_model(
                path_to_load=self.PATH, DEVICE=self.DEVICE).half()
        else:
            self.model_load: torch.jit._script.RecursiveScriptModule = ISORT.load_model(
                path_to_load=self.PATH, DEVICE=self.DEVICE)

    def release_memory(self):
        del self.model_load

    @staticmethod
    def decode_base64(image_base64: bytes = None) -> Image:
        return Image.open(BytesIO(base64.b64decode(image_base64)))

    @staticmethod
    def transform_image(image_base64: bytes = None, crop_at: List[int] = [], scale_img_devide: int = 2) -> torch.Tensor:

        image_pil: Image = ISORT.decode_base64(image_base64=image_base64)
        image_pil_cut: np.ndarray = np.array(ISORT.crop_image(
            image_full=image_pil, crop_at=crop_at).convert("RGB"))
        IMAGE_HEIGHT: int = 512//scale_img_devide
        IMAGE_WIDTH: int = 512//scale_img_devide
        transform: A.core.composition.Compose = A.Compose([
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.augmentations.transforms.Normalize(
                mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])
        transform_image: torch.Tensor = transform(image=image_pil_cut)[
            "image"].unsqueeze(0)

        return transform_image

    @staticmethod
    def crop_image(image_full: Image = None, crop_at: List[int] = []) -> Image:
        """
        crop_at = [x_min, x_max, y_min, y_max]
        """

        if not crop_at:
            image_array_full: np.ndarray = np.asarray(image_full)
            image_array: np.ndarray = np.array(image_array_full[:, :])
        else:
            image_array_full: np.ndarray = np.asarray(image_full)
            image_array: np.ndarray = np.array(
                image_array_full[crop_at[2]:crop_at[3], crop_at[0]:crop_at[1]])
        image_pil: Image = Image.fromarray(np.uint8(image_array))

        return image_pil

    @staticmethod
    def save_image(path_to_save: str = None, image_name: str = None, image_array: np.ndarray = None):
        assert type(path_to_save) is str, "path_to_save must be string type"
        assert type(image_name) is str, "image_name must be string type"
        assert type(
            image_array) is np.ndarray, "image_array must be np.ndarray type"

        if os.path.isfile(r"{}\\{}".format(path_to_save, image_name)):
            plt.imsave(r"{}\\{}_{}".format(
                path_to_save, time.time(), image_name), image_array)
        else:
            plt.imsave(r"{}\\{}".format(path_to_save, image_name), image_array)

    @staticmethod
    def load_model(path_to_load: str = None, DEVICE="cpu") -> torch.jit._script.RecursiveScriptModule:
        assert type(path_to_load) is str, "path_to_load must be string type"
        assert type(DEVICE) is str, "DEVICE must be string type"

        model_load: torch.jit._script.RecursiveScriptModule = torch.jit.load(
            path_to_load, map_location=DEVICE)
        return model_load.eval()

    async def predict(self,
                      image_base64: bytes = None,
                      crop_at: List[int] = [],
                      mode: str = "deploy",
                      mutex_lock: asyncio.locks.Lock = None) -> Tuple[np.ndarray, float, List[float], List[float]]:
        # if use model 256 set [scale_img_devide] to 2
        # if use model 512 set [scale_img_devide] to 1

        model_load: torch.jit._script.RecursiveScriptModule = self.model_load
        scale_img_devide: int = self.scale_img_devide
        DEVICE: str = self.DEVICE

        assert type(image_base64) is bytes, "image_base64 must be bytes type"
        assert type(crop_at) is list, "crop_at must be list type"
        assert type(
            model_load) is torch.jit._script.RecursiveScriptModule, "model_load nust be torch.jit._script.RecursiveScriptModule type"
        assert type(DEVICE) is str, "DEVICE must be string type"
        assert type(mode) is str, "mode must be string type"
        assert type(
            mutex_lock) is asyncio.locks.Lock, "lock must be asyncio.locks.Lock type"

        async with mutex_lock:
            with torch.no_grad():
                if self.HALF:
                    image_tensor: torch.Tensor = torch.Tensor.half(ISORT.transform_image(
                        image_base64=image_base64, crop_at=crop_at, scale_img_devide=scale_img_devide)).to(DEVICE)
                else:
                    image_tensor: torch.Tensor = ISORT.transform_image(
                        image_base64=image_base64, crop_at=crop_at, scale_img_devide=scale_img_devide).to(DEVICE)
                softmax: torch.nn.modules.activation.Softmax = nn.Softmax(
                    dim=1)
                prediction_torch: torch.Tensor = torch.argmax(
                    softmax(model_load(image_tensor)), axis=1).to(DEVICE)
            prediction: np.ndarray = np.squeeze(
                prediction_torch.detach().cpu().numpy())

        # Ex pad_block = [[x_min,x_max,y_min,y_max],[x_min,x_max,y_min,y_max],[x_min,x_max,y_min,y_max]]
        # pad count in clock wise start at top left pad called pad 1
        pad_block: list[int] = [
            [65, 65+75, 15, 15+100],  # pad 1 top left
            [165, 165+75, 15, 15+100],
            [265, 265+75, 15, 15+100],
            [365, 365+75, 15, 15+100],  # pad 4 top right

            [375, 375+135, 145, 145+65],
            [375, 375+135, 225, 225+65],
            [375, 375+135, 305, 305+65],

            [365, 365+75, 400, 400+100],  # pad 8 buttom right Trapezoid
            [265, 265+75, 400, 400+100],
            [165, 165+75, 400, 400+100],
            [65, 65+75, 400, 400+100],  # pad 11 buttom left

            [15, 15+135, 305, 305+65],
            [15, 15+135, 225, 225+65],
            [15, 15+135, 145, 145+65]
        ]

        # 5 class[background, defect_pad, defect_mole, edge, pad]
        try:
            mask_bg: int = 0
            mask_defect_pad: int = 1
            mask_defect_mole: int = 2
            mask_edge: int = 3
            mask_pad: int = 4

            defect_mole_per_bg: float = 0
            defect_pad_per_pad: float = 0

            bg: int = int(np.sum(prediction == mask_bg))
            defect_pad: int = int(np.sum(prediction == mask_defect_pad))
            defect_mole: int = int(np.sum(prediction == mask_defect_mole))
            pad: int = int(np.sum(prediction == mask_pad))
            edge: int = int(np.sum(prediction == mask_edge))

            defect_mole_per_bg: float = float(
                (defect_mole * 100 / (defect_mole + bg)))

            prediction_cal_pad_defect: np.ndarray = prediction.copy().astype(np.uint8)
            prediction_cal_pad_defect[prediction_cal_pad_defect !=
                                      mask_defect_pad] = 0

            prediction_cal_pad: np.ndarray = prediction.copy().astype(np.uint8)
            prediction_cal_pad[prediction_cal_pad != mask_pad] = 0
            prediction_cal_pad[prediction_cal_pad == mask_pad] = 1

            pad_defect_in_area: list[int] = [int(np.sum(prediction_cal_pad_defect[y_min//scale_img_devide:y_max//scale_img_devide,
                                                 x_min//scale_img_devide:x_max//scale_img_devide]))for x_min, x_max, y_min, y_max in pad_block]
            pad_in_area: list[int] = [int(np.sum(prediction_cal_pad[y_min//scale_img_devide:y_max//scale_img_devide,
                                          x_min//scale_img_devide:x_max//scale_img_devide]))for x_min, x_max, y_min, y_max in pad_block]
            defect_pad_per_pad: list[float] = [float(pad_defect_in_area[i] * 100 / (
                pad_defect_in_area[i] + pad_in_area[i])) for i in range(len(pad_block))]

            prediction_cal_defect_mole_instant: np.ndarray = prediction.copy().astype(np.uint8)
            prediction_cal_defect_mole_instant[prediction_cal_defect_mole_instant !=
                                               mask_defect_mole] = 0
            prediction_cal_defect_mole_instant[prediction_cal_defect_mole_instant ==
                                               mask_defect_mole] = 1

            contours_defect_moles, _ = cv2.findContours(
                prediction_cal_defect_mole_instant, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            defect_mole_instants = [float(cv2.contourArea(
                contours_defect_mole)) for contours_defect_mole in contours_defect_moles]
            defect_mole_instant_per_bg = [(defect_mole_instant * 100/(
                bg + defect_mole_instant)) for defect_mole_instant in defect_mole_instants]

            if defect_pad_per_pad == [100.0]*14:
                raise ValueError
        except:
            raise ValueError("Load wrong model U-Net must has 5 classes")

        image_ori: np.ndarray = np.sum(np.squeeze(
            image_tensor.permute(0, 2, 3, 1).detach().cpu().numpy()), axis=2)
        image_ori_with_pred: np.ndarray = np.concatenate(
            (image_ori, prediction), axis=1)

        if mode == "test":
            for i, temp in enumerate(defect_pad_per_pad):
                print(f"pad {i+1} , defect_pad_per_pad {temp}")

            print(f"bg {bg}")
            print(f"defect_pad {defect_pad}")
            print(f"defect_mole {defect_mole}")
            print(f"pad {pad}")
            print(f"defect_mole_per_bg {defect_mole_per_bg}")
            print(f"edge {edge}")

        return image_ori_with_pred, defect_mole_per_bg, defect_pad_per_pad, defect_mole_instant_per_bg


# ****************************************************************************************************************************************************************************


async def test_cropped_image(lock, isort):
    current_path: str = os.getcwd()
    # step 2 receieve image in base64
    image_path = r"{}\\test\\image\\14LD_LGA3X2.5_FJ8001_0110_WafFRAME #8_Col03Row27_0.bmp".format(
        current_path)
    image = Image.open(image_path)
    buff = BytesIO()
    image.save(buff, format="JPEG")
    image_base64 = base64.b64encode(buff.getvalue())

    # step 3 sent image_base64 and model to fucntion predict
    # 3.1 deploy set mode="deploy" or test set mode="test"
    crop: list[int] = []
    image_ori_with_pred, defect_mole_per_bg, defect_pad_per_pad, defect_mole_instant_per_bg = await isort.predict(image_base64=image_base64, mode="deploy", crop_at=crop, mutex_lock=lock)

    path_to_save = current_path + r"\\test\\result"
    image_name = r"U-Net_256_test_crop.png"
    isort.save_image(path_to_save=path_to_save,
                     image_name=image_name, image_array=image_ori_with_pred)


async def test_raw_image(lock, isort):
    current_path: str = os.getcwd()
    # step 2 receieve image in base64
    image_path = r"{}\\test\\image\\14LD_LGA3X2.5_FJ8001_0033_WafFRAME #2_Col12Row25_0.bmp".format(
        current_path)
    image = Image.open(image_path)
    buff = BytesIO()
    image.save(buff, format="JPEG")
    image_base64 = base64.b64encode(buff.getvalue())
    # step 3 sent image_base64 and model to fucntion predict
    # 3.1 deploy set mode="deploy" or test set mode="test"
    crop: list[int] = [675, 675+653, 609, 609+785]
    image_ori_with_pred, defect_mole_per_bg, defect_pad_per_pad, defect_mole_instant_per_bg = await isort.predict(image_base64=image_base64, mode="deploy", crop_at=crop, mutex_lock=lock)

    path_to_save = current_path + r"\\test\\result"
    image_name = r"U-Net_256_test_raw.png"
    isort.save_image(path_to_save=path_to_save,
                     image_name=image_name, image_array=image_ori_with_pred)


async def test_raw_image_for_report(lock, isort):
    current_path: str = os.getcwd()
    # step 2 receieve image in base64
    image_path = r"E:\Work\Project\ISORT U-Net\test\Report temp\New folder (2)"
    image_name = os.listdir(image_path)
    j = 0
    for i in image_name:
        full_image_path = f"{image_path}\\{i}"
        image = Image.open(full_image_path)
        buff = BytesIO()
        image.save(buff, format="JPEG")
        image_base64 = base64.b64encode(buff.getvalue())
        # step 3 sent image_base64 and model to fucntion predict
        # 3.1 deploy set mode="deploy" or test set mode="test"
        crop: list[int] = [709, 709+603, 669, 669+715]
        image_ori_with_pred, defect_mole_per_bg, defect_pad_per_pad, defect_mole_instant_per_bg = await isort.predict(image_base64=image_base64, mode="deploy", crop_at=crop, mutex_lock=lock)
        j += 1
        path_to_save = r"E:\Work\Project\ISORT U-Net\test\Report temp\New folder (2)"
        image_name = f"{i}_{j}_{defect_mole_per_bg}.png"
        isort.save_image(path_to_save=path_to_save,
                         image_name=image_name, image_array=image_ori_with_pred)


if __name__ == "__main__":
    mutex_lock = asyncio.Lock()
    time_avg = []
    isort = ISORT()

    asyncio.run(test_raw_image_for_report(mutex_lock, isort))
