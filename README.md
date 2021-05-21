## Requirements
 TensorFlow CPU
```bash
pip install -r requirements.txt
```
 TensorFlow GPU
```bash
pip install -r requirements-gpu.txt
```
### Weights File
Download the [Weights](https://drive.google.com/file/d/1-4_7BuN3fPMrSX5kJ2MKzM9KuwGwaDq8/view?usp=sharing)

Save the weights into the 'data' folder of this repository and rename it as
```bash
custom.weights
```


### Run below for creating the model
```bash
conda env create -f conda-cpu.yml
conda activate yolov4-cpu
python save_model.py --weights ./data/custom.weights --output ./checkpoints/custom-416 --input_s
```
### To run the run the model
```bash
python detect_video.py --weights ./checkpoints/custom-416 --size 416 --model yolov4 --video 0 --output ./detections/results.avi
```

Special thanks to [The AI Guy](https://github.com/theAIGuysCode) for guidance.
