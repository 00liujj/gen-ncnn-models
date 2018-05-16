# gen ncnn models

### yolov2 to ncnn/darknet to ncnn
```bash
./darknet2ncnn.py tests/darknet_yolov2.cfg tests/darknet_yolov2.weights
```

### tensorflow to ncnn
```bash
./tensorflow2ncnn.py tests/frozen.pb
```
