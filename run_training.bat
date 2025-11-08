@echo off
echo Activating Miniconda environment tf115_new...
call C:\ProgramData\miniconda3\Scripts\activate.bat C:\ProgramData\miniconda3\envs\tf115_new

echo Current Python path:
where python

echo Starting Tacotron2 training with batch_size=2...
python train_tacotron2.py --batch_size=2

pause

