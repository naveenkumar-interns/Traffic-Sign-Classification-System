# Traffic Sign Recognition

This project uses a Convolutional Neural Network (CNN) to classify traffic signs into 43 categories. It includes scripts for training a model and testing it with a webcam.

## Requirements

- Python 3.x
- Libraries: `numpy`, `opencv-python`, `matplotlib`, `keras`, `tensorflow`, `scikit-learn`, `pandas`
- Install with:
  ```bash
  pip install numpy opencv-python matplotlib keras tensorflow scikit-learn pandas
  ```
- Webcam for testing
- Dataset in `myData` folder with subfolders for each class (0–42) and a `labels.csv` file mapping class numbers to names

## Project Files

- **train.py**: Trains the CNN model and saves it as `model_trained.p`
- **test.py**: Uses the trained model for real-time traffic sign detection via webcam
- **myData**: Folder for dataset images
- **labels.csv**: Maps class numbers to traffic sign names

## Usage

### Training
1. Place images in `myData` (e.g., `myData/0/`, `myData/1/`) and prepare `labels.csv`
2. Run:
   ```bash
   python train.py
   ```
3. Outputs:
   - Trained model (`model_trained.p`)
   - Plots of sample images, class distribution, and training/validation metrics

### Testing
1. Ensure `model_trained.p` is in the directory
2. Run:
   ```bash
   python test.py
   ```
3. Displays webcam feed with predicted class and confidence (press `q` to exit)

## Model Details

- **Input**: 32x32 grayscale images
- **Architecture**: 4 convolutional layers, 2 max-pooling layers, dropout (0.5), dense layers (500 nodes, 43 classes)
- **Training**: 10 epochs, batch size 50, data augmentation (shift, zoom, rotation)
- **Data Split**: 64% train, 16% validation, 20% test

## Notes
- Ensure images are 32x32 pixels
- Webcam must be at index 0
- Adjust `threshold` (0.75) in `test.py` for prediction confidence
