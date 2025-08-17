
@echo off
title Faster-Whisper RU (GPU - NVIDIA)
REM Requires recent NVIDIA driver on Windows. CUDA toolkit is NOT required.
set WHISPER_DEVICE=cuda
set WHISPER_COMPUTE=float16
set WHISPER_LANG=ru
REM Optionally change model: tiny/base/small/medium/large-v3
REM set WHISPER_MODEL=small
python transcribe_faster_ru.py
pause
