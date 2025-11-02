@echo off
REM Training script for Tacotron2 with Conda environment
echo Activating conda environment...
call %USERPROFILE%\miniconda3\Scripts\activate.bat %USERPROFILE%\miniconda3\envs\tf115

echo Starting Tacotron2 training...
python train_tacotron2.py

pause
