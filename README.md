
```bash
# TensorFlow CPU
pip install -r requirements.txt

# TensorFlow GPU
pip install -r requirements-gpu.txt

Download this Weight "yolov4-obj_best.weights"
Copy and paste weights from into the 'data' folder of this repository and rename it as custom.weights.


cd Desktop\pr1\tensorflow-yolov4-tflite
conda env create -f conda-cpu.yml
conda activate yolov4-cpu
python detect_video.py --weights ./checkpoints/custom-416 --size 416 --model yolov4 --video 0 --output ./detections/results.avi




##Run below for creating the model
##Run below for first time only
cd Desktop\pr1\tensorflow-yolov4-tflite
conda env create -f conda-cpu.yml
conda activate yolov4-cpu
python save_model.py --weights ./data/custom.weights --output ./checkpoints/custom-416 --input_s

##To run the run the model
python detect_video.py --weights ./checkpoints/custom-416 --size 416 --model yolov4 --video 0 --output ./detections/results.avi
