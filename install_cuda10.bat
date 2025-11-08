@echo off
echo ========================================
echo CUDA 10.0 and cuDNN 7.6 Installation
echo ========================================
echo.

echo Activating conda environment tf115_new...
call C:\ProgramData\miniconda3\Scripts\activate.bat C:\ProgramData\miniconda3\envs\tf115_new

echo.
echo Installing CUDA Toolkit 10.0 and cuDNN 7.6...
conda install cudatoolkit=10.0 cudnn=7.6 -y

echo.
echo Verifying installation...
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU Available:', tf.test.is_gpu_available())"

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Edit train_tacotron2.py line 248 to uncomment GPU settings
echo 2. Run train.bat to start training
echo.
pause
