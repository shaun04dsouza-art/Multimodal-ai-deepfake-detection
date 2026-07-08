@echo off
echo Killing processes on port 8000...
for /f "tokens=5" %%a in ('netstat -ano ^| find ":8000"') do (
    echo Terminating PID %%a
    taskkill /PID %%a /F >nul 2>&1
)

echo Starting Backend...
cd /d D:\deepfake_detection_project\backend
call D:\deepfake_detection_project\.venv\Scripts\activate
start "Backend" cmd /k "python main.py"

timeout /t 5 >nul

echo Starting Frontend...
cd /d D:\deepfake_detection_project\frontend
start "Frontend" cmd /k "python -m http.server 3000"

timeout /t 2 >nul
start http://localhost:3000/deepfake_app.html

pause
