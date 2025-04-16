#!/bin/bash
set -e

DATASET_PATH=/home/pi/yolo-benchmark/data/datasets/coco128/images/train2017/
MODELS_PATH=/home/pi/yolo-benchmark/data/models/

cd /home/pi/yolo-benchmark

for MODEL_ARCHITECTURE in yolo11n yolo11s yolo11m yolo11l yolo11x; do
  echo "RUNNING BENCHMARK FOR '$MODEL_ARCHITECTURE'" && sleep 10

  python3 src/InferenceBenckmark/main.py --images_folder ${DATASET_PATH} --model_path ${MODELS_PATH}/onnx/${MODEL_ARCHITECTURE}_float32.onnx --language python && echo "Waiting 10s to next execution..." && sleep 10
  python3 src/InferenceBenckmark/main.py --images_folder ${DATASET_PATH} --model_path ${MODELS_PATH}/onnx/${MODEL_ARCHITECTURE}_float32.onnx --language python --half_cores && echo "Waiting 10s to next execution..." && sleep 10
  python3 src/InferenceBenckmark/main.py --images_folder ${DATASET_PATH} --model_path ${MODELS_PATH}/onnx/${MODEL_ARCHITECTURE}_float32.onnx --language cpp && echo "Waiting 10s to next execution..." && sleep 10
  python3 src/InferenceBenckmark/main.py --images_folder ${DATASET_PATH} --model_path ${MODELS_PATH}/onnx/${MODEL_ARCHITECTURE}_float32.onnx --language cpp --half_cores && echo "Waiting 10s to next execution..." && sleep 10

  python3 src/InferenceBenckmark/main.py --images_folder ${DATASET_PATH} --model_path ${MODELS_PATH}/onnx/${MODEL_ARCHITECTURE}_float16.onnx --language python && echo "Waiting 10s to next execution..." && sleep 10
  python3 src/InferenceBenckmark/main.py --images_folder ${DATASET_PATH} --model_path ${MODELS_PATH}/onnx/${MODEL_ARCHITECTURE}_float16.onnx --language python --half_cores && echo "Waiting 10s to next execution..." && sleep 10
  python3 src/InferenceBenckmark/main.py --images_folder ${DATASET_PATH} --model_path ${MODELS_PATH}/onnx/${MODEL_ARCHITECTURE}_float16.onnx --language cpp && echo "Waiting 10s to next execution..." && sleep 10
  python3 src/InferenceBenckmark/main.py --images_folder ${DATASET_PATH} --model_path ${MODELS_PATH}/onnx/${MODEL_ARCHITECTURE}_float16.onnx --language cpp --half_cores && echo "Waiting 10s to next execution..." && sleep 10

  python3 src/InferenceBenckmark/main.py --images_folder ${DATASET_PATH} --model_path ${MODELS_PATH}/tflite/${MODEL_ARCHITECTURE}_float32.tflite --language python && echo "Waiting 10s to next execution..." && sleep 10
  python3 src/InferenceBenckmark/main.py --images_folder ${DATASET_PATH} --model_path ${MODELS_PATH}/tflite/${MODEL_ARCHITECTURE}_float32.tflite --language python --half_cores && echo "Waiting 10s to next execution..." && sleep 10
  python3 src/InferenceBenckmark/main.py --images_folder ${DATASET_PATH} --model_path ${MODELS_PATH}/tflite/${MODEL_ARCHITECTURE}_float32.tflite --language cpp && echo "Waiting 10s to next execution..." && sleep 10
  python3 src/InferenceBenckmark/main.py --images_folder ${DATASET_PATH} --model_path ${MODELS_PATH}/tflite/${MODEL_ARCHITECTURE}_float32.tflite --language cpp --half_cores && echo "Waiting 10s to next execution..." && sleep 10

  python3 src/InferenceBenckmark/main.py --images_folder ${DATASET_PATH} --model_path ${MODELS_PATH}/tflite/${MODEL_ARCHITECTURE}_float16.tflite --language python && echo "Waiting 10s to next execution..." && sleep 10
  python3 src/InferenceBenckmark/main.py --images_folder ${DATASET_PATH} --model_path ${MODELS_PATH}/tflite/${MODEL_ARCHITECTURE}_float16.tflite --language python --half_cores && echo "Waiting 10s to next execution..." && sleep 10
  python3 src/InferenceBenckmark/main.py --images_folder ${DATASET_PATH} --model_path ${MODELS_PATH}/tflite/${MODEL_ARCHITECTURE}_float16.tflite --language cpp && echo "Waiting 10s to next execution..." && sleep 10
  python3 src/InferenceBenckmark/main.py --images_folder ${DATASET_PATH} --model_path ${MODELS_PATH}/tflite/${MODEL_ARCHITECTURE}_float16.tflite --language cpp --half_cores && echo "Waiting 10s to next execution..." && sleep 10

  python3 src/InferenceBenckmark/main.py --images_folder ${DATASET_PATH} --model_path ${MODELS_PATH}/tflite/${MODEL_ARCHITECTURE}_int8.tflite --language python && echo "Waiting 10s to next execution..." && sleep 10
  python3 src/InferenceBenckmark/main.py --images_folder ${DATASET_PATH} --model_path ${MODELS_PATH}/tflite/${MODEL_ARCHITECTURE}_int8.tflite --language python --half_cores && echo "Waiting 10s to next execution..." && sleep 10
  python3 src/InferenceBenckmark/main.py --images_folder ${DATASET_PATH} --model_path ${MODELS_PATH}/tflite/${MODEL_ARCHITECTURE}_int8.tflite --language cpp && echo "Waiting 10s to next execution..." && sleep 10
  python3 src/InferenceBenckmark/main.py --images_folder ${DATASET_PATH} --model_path ${MODELS_PATH}/tflite/${MODEL_ARCHITECTURE}_int8.tflite --language cpp --half_cores && echo "Waiting 10s to next execution..." && sleep 10

done