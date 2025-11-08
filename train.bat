@echo off
call C:\ProgramData\miniconda3\Scripts\activate.bat C:\ProgramData\miniconda3\envs\tf115_new
python train_tacotron2.py --batch_size=2
