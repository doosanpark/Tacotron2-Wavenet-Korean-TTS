@echo off
set CUDA_VISIBLE_DEVICES=-1
C:\ProgramData\miniconda3\envs\tf115_new\python.exe train_tacotron2.py --batch_size=4
