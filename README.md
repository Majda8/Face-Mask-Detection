# Face Mask Detection

This project uses a **MobileNetV2** based model to detect if a person is wearing a mask or not. It employs transfer learning to fine-tune a mask classifier.

## Files

1. **`detect_mask_video.py`**: Detects masks in real-time from a video or webcam feed.
2. **`mask_detector_model.keras`**: Pre-trained mask detection model.
3. **`train_mask_detector.py`**: Trains the mask detection model using a dataset of images with and without masks.

## Prerequisites

Make sure to have the following installed:

- Python (3.x)
- Required libraries: TensorFlow, Keras, Scikit-learn, Imutils, Matplotlib, Numpy

### Installation of dependencies
You can install the required packages using `pip`:
```bash
pip install tensorflow scikit-learn imutils matplotlib numpy
```

## Usage

To run the mask detection directly from a video or webcam feed, simply execute the following command:

```bash
python detect_mask_video.py
```

The script will start and automatically detect faces with or without masks in the video.

## Notes

- The model is trained using **MobileNetV2** with data augmentation.
- The trained model is saved as `mask_detector_model.keras`.

Feel free to contribute or improve the model as needed!
