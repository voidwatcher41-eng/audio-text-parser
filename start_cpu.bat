
@echo off
title Faster-Whisper RU (CPU)
set WHISPER_DEVICE=cpu
set WHISPER_COMPUTE=int8
set WHISPER_LANG=ru
REM Optionally change model: tiny/base/small/medium/large-v3
REM set WHISPER_MODEL=small
python transcribe_faster_ru.py
pause
