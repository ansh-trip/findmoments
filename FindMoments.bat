@echo off
REM Drag a VOD onto this file (or run: FindMoments.bat "C:\path\vod.mp4" --llm gemini)
REM Keep findmoments.py in the same folder as this .bat.
if "%~1"=="" (
  echo Drag a VOD file onto FindMoments.bat, or run: FindMoments.bat "C:\path\vod.mp4"
  pause
  exit /b 1
)
"%USERPROFILE%\.publikclip\env\Scripts\python.exe" "%~dp0findmoments.py" %*
echo.
pause
