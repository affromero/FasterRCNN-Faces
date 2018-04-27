# Face Detection using Faster RCNN
This repo uses a Face Detector caffe-model trained using jointly several databases: IJB-A, FDDB and WIDER. 

## Usage
In order to setup the folders and download the caffemodel, please run first the `setup.sh`

One the setup has been made, the file you are looking for is `get_faces.py`. There are two modules you want to use, for instance:
 
```python
from get_faces import __init__, face_from_file
net=__init__()
face = face_from_file('/path/to/your/img', net)
```

## Requirements
- Caffe
- OpenCV
