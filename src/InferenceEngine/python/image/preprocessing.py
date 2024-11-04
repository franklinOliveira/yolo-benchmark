import numpy as np
import cv2
from typing import Tuple

class ImagePreprocessing:
    """
    A class for preprocessing images for deep learning models, including formatting,
    quantization, and applying letterbox resizing.

    Methods
    -------
    format(images: np.ndarray, input_shape: Tuple[int, int], litert_model: bool = True) -> np.ndarray
        Formats the input images for model compatibility by resizing and normalizing.
    quantize(input: np.ndarray, scale: float, zero_point: int, type: np.dtype) -> np.ndarray
        Quantizes the input array based on the specified scale and zero point.
    __apply_letterbox(image: np.ndarray, new_shape: Tuple[int, int]) -> np.ndarray
        Applies letterbox resizing to maintain the aspect ratio of the image.
    """

    @staticmethod
    def format(images: np.ndarray, input_shape: Tuple[int, int], litert_model: bool = True) -> np.ndarray:
        """
        Formats the input images for model compatibility by resizing and normalizing.

        Parameters
        ----------
        images : np.ndarray
            An array of images to format. Each image should be in the format (height, width, channels).
        input_shape : Tuple[int, int]
            The target shape for resizing the images (height, width).
        litert_model : bool, optional
            A flag indicating whether to transpose the input for a lite model (default is True).

        Returns
        -------
        np.ndarray
            The formatted and normalized images, scaled to the range [0, 1].
        """
        for idx in range(len(images)):
            images[idx] = cv2.resize(images[idx], input_shape, interpolation=cv2.INTER_LINEAR)

        input = np.stack(images)
        input = input[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB and rearrange dimensions
        input = np.ascontiguousarray(input)
        input = input.astype(np.float32)

        if litert_model:
            input = input.transpose((0, 2, 3, 1))  # Rearrange dimensions for lite model
        
        return input / 255  # Normalize to [0, 1]

    @staticmethod
    def quantize(input: np.ndarray, scale: float, zero_point: int, type: np.dtype) -> np.ndarray:
        """
        Quantizes the input array based on the specified scale and zero point.

        Parameters
        ----------
        input : np.ndarray
            The input array to quantize.
        scale : float
            The scale factor for quantization.
        zero_point : int
            The zero point for quantization.
        type : np.dtype
            The target data type for the quantized output.

        Returns
        -------
        np.ndarray
            The quantized input array.
        """
        quantized_input = (input / scale + zero_point).astype(type)
        return quantized_input

    @staticmethod
    def __apply_letterbox(image: np.ndarray, new_shape: Tuple[int, int]) -> np.ndarray:
        """
        Applies letterbox resizing to maintain the aspect ratio of the image.

        Parameters
        ----------
        image : np.ndarray
            The image to apply letterbox resizing to.
        new_shape : Tuple[int, int]
            The target shape for the letterbox resizing (height, width).

        Returns
        -------
        np.ndarray
            The resized image with letterboxing applied.
        """
        input = image.copy()  
        shape = input.shape[:2]  # Get the original shape (height, width)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # Calculate the resize ratio
        new_unpad: Tuple[int, int] = (int(round(shape[1] * r)), int(round(shape[0] * r)))  
        dw: float = new_shape[1] - new_unpad[0]  # Width padding
        dh: float = new_shape[0] - new_unpad[1]  # Height padding
        dw /= 2  
        dh /= 2 

        if shape[::-1] != new_unpad:  # If the new shape differs from the unpadded shape
            input = cv2.resize(input, new_unpad, interpolation=cv2.INTER_LINEAR)

        top: int = int(round(dh - 0.1))  
        bottom: int = int(round(dh + 0.1))  
        left: int = int(round(dw - 0.1))  
        right: int = int(round(dw + 0.1))  

        input = cv2.copyMakeBorder(input, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        return input  
