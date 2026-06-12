@echo off
echo === NeuroCoder High Performance Mode ===

REM Set this process to High priority
wmic process where name="cmd.exe" CALL setpriority "High" >nul 2>&1

REM Force NVIDIA GPU to stay at max performance
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c >nul 2>&1

REM Run training
C:\Users\YANYZ\anaconda3\python.exe train.py --config small --steps 10000 --data "D:/NeuroCoder/sample_data/source"

REM Restore balanced power when done
powercfg /setactive 381b4222-f694-41f0-9685-ff5bb260df2e >nul 2>&1
pause
