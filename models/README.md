# Required model files (download these two, put them in this folder)

No code needed — just download and drop the files in.

1. **deploy.prototxt**
   https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt

2. **res10_300x300_ssd_iter_140000.caffemodel**
   https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel

3. **nn4.small2.v1.t7** (OpenFace embedding model)
   https://github.com/pyannote/pyannote-data/raw/master/openface.nn4.small2.v1.t7
   (mirror also commonly found under cmusatyalab/openface or the "misbah4064/face_recognition_opencv" GitHub repo — search "nn4.small2.v1.t7 download" if a link is stale)

Just save each file into this `models/` folder with the exact filenames above before deploying.
