# Yolov4-DeepSORT with Saliant Human Objects Selection (SHOS)

In order to keep track of human existance within a certain area in the workspace, a high performance human detector combined with multi-object tracking algorithm is needed to complete the task. The human detection and tracking system provided in this repo is implemented with Yolov4 in Tensorflow for detection, DeepSORT for tracking, SHOS for sampling, and PySqlite3 for database. Below shows the overall architecture of the system.
<p align="center"><img src="data/helpers/deepsort.jpg" width="576"\></p>

Repetitive human image samples is being produced after using the detection-tracking system. To ensure that the human image gallery does not include human image with similar view angle, a sampling method called Saliant Human Objects Selection (SHOS) is implemented to filter out unnecessary repetitive human images. Below shows the flow chart on how the SHOS works.
<p align="center"><img src="data/helpers/shos.jpg" width="384"\></p>

## Installation

For Windows OS, open your command prompt. For Ubuntu, open your terminal.
Make sure Anaconda or Pip is installed before running the following code. Anaconda is recommended route for GPU installation as it configures CUDA toolkit version.

### Conda
```bash
# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov4-gpu 
```

### Pip
(TensorFlow 2 packages require a pip version >19.0.)
```bash
# TensorFlow GPU
pip install -r requirements-gpu.txt
```
### Nvidia Driver (For GPU, if you are not using Conda Environment and haven't set up CUDA yet)
Make sure to use CUDA Toolkit version 10.1 as it is the proper version for the TensorFlow version used in this repository.
https://developer.nvidia.com/cuda-10.1-download-archive-update2

## Setting weights file
- Download the pre-trained yolov4.weights file at: https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT
- Copy and paste the yolov4.weights from your downloads folder into the 'data' folder of this repo.

## Convert the weight file to Tensorflow model 
Convert the .weights into corresponding Tensorflow model using the following command line. The converted model will be saved into checkpoint folder. 
```bash
# Convert darknet weights to tensorflow model
python save_model.py --model yolov4 
```

## Run the human detection-tracking system
To run the human detection-tracking system, run the human_tracker.py. Remember to place ALL of the CCTV footages in the default folder at /data/video. Rename the video file according to the camera channel. For example, video captured by camera channel 2 is renamed as "ch2.mp4", channel 3 as "ch3.mp4". These files should be in the ./data/video folder by default.
- Default location of the input video is at /data/video folder.
- Default location of the output file is at /output folder. 
 
```bash
# Run the system with default settings
python human_tracker.py

# Run the system with different input and output video path 
python human_tracker.py --video ./path/to/video/input.mp4 --output ./path/to/video/output.mp4

# Run the system with trajectory
python human_tracker.py --trajectory True

# Run the system with SHOS graph
python human_tracker.py --plot_graph True

```
## Database 
The database file path is at ./database/Image.db. Every time the human_tracker.py is executed, the Image.db file will be deleted and a new Image.db will be created to store new data. Make sure the human_tracker.py execution is not interrupted halftway as there is no checkpoint saved. Will add the checkpoint function in future if needed. 

## Multiprocessing
To set the number of processes to run in parallel, use the following command.
```bash
# Run two process in every batch
python human_tracker.py --parallel_ps 2
```
Keep in mind that the more processes run together, the lower the fps of the data extraction from the detection-tracking system. 

## Overlapping issue
There is a chance that the overlapping issue will cause ID switching problem. It is recommended to use CCTV footages with top-down view angle to reduce human overlapping issue. 

### References  
   Special thanks to theAIGuysCode, hunglc007 and nwojke for providing the methods for human detection and tracking:
  * [Yolov4-Deepsort implementation](https://github.com/theAIGuysCode/yolov4-deepsort)
  * [tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite)
  * [Deep SORT Repository](https://github.com/nwojke/deep_sort)
