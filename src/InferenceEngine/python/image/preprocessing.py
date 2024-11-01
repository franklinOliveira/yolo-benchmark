import numpy as np
import cv2

class ImagePreprocessing:
    
    @staticmethod
    def format(images: np.ndarray, input_shape: tuple, litert_model: bool = True) -> np.ndarray:
        for idx, _ in enumerate(images):
            images[idx] = cv2.resize(images[idx], input_shape, interpolation=cv2.INTER_LINEAR)
 
        input = np.stack(images)
        input = input[..., ::-1].transpose((0, 3, 1, 2))
        input = np.ascontiguousarray(input)
        input = input.astype(np.float32)

        if litert_model:
            input = input.transpose((0, 2, 3, 1))
        
        return input / 255


    @staticmethod
    def quantize(input: np.ndarray, scale: float, zero_point: int, type: np.dtype) -> np.ndarray:
        quantized_input = (input / scale + zero_point).astype(type)
        return quantized_input

    @staticmethod
    def __apply_letterbox(image: np.ndarray, new_shape: tuple[int, int]) -> np.ndarray:
        input = image.copy()  
        shape = input.shape[:2]  

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad: tuple[int, int] = (int(round(shape[1] * r)), int(round(shape[0] * r)))  
        dw: float = new_shape[1] - new_unpad[0]  
        dh: float = new_shape[0] - new_unpad[1]  
        dw /= 2  
        dh /= 2 

        if shape[::-1] != new_unpad:  
            input = cv2.resize(input, new_unpad, interpolation=cv2.INTER_LINEAR)  

        top: int = int(round(dh - 0.1))  
        bottom: int = int(round(dh + 0.1))  
        left: int = int(round(dw - 0.1))  
        right: int = int(round(dw + 0.1))  

        input = cv2.copyMakeBorder(input, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        return input  