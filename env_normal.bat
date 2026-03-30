@echo off
cls
echo Starting engine...

:: Install ReActorZero (common for both modes)
echo ReActorZero installing...
set "url=https://huggingface.co/datasets/LeeAeron/ReActorZero/resolve/main/sd_forge_reactor_zero.zip"
set "download_folder=tmp"
set "extract_folder=extensions-builtin/sd_forge_reactor_zero/"
set "file_name=reactor_version.py"
set "folder_path=.\extensions-builtin\sd_forge_reactor_zero\scripts\"
set "aria2_path=%~dp0extensions-builtin\sd_forge_civitai_browser_plus\aria2\win\aria2.exe"
if not exist "%folder_path%" (
    mkdir "%folder_path%"
)
if not exist "%folder_path%\%file_name%" (
    if not exist "%download_folder%" mkdir "%download_folder%"
    if not exist "%extract_folder%" mkdir "%extract_folder%"
    "%aria2_path%" -d "%download_folder%" -o "archive.zip" "%url%"
    powershell -Command "Expand-Archive -Path '%download_folder%\archive.zip' -DestinationPath '%extract_folder%' -Force"
    del "%download_folder%\archive.zip"
) else (
    echo OK.
)

cls

:: Set cache directories
set "HF_HOME=%cd%_cache\huggingface"
set "XDG_CACHE_HOME=%cd%_cache"
set "HF_DATASETS_CACHE=%cd%_cache\huggingface\datasets"

set PYTHON=
set GIT=
set VENV_DIR=

:: Default arguments for Normal mode (RTX 30xx/40xx and older)
set COMMANDLINE_ARGS=--cuda-stream ^
--cuda-malloc ^
--pin-shared-memory ^
--attention-pytorch ^
--disable-gpu-warning ^
--precision half ^
--fast-fp16

:: Optional: Uncomment to use custom model directories
@REM --ckpt-dir D:/COMFY_UI/ComfyUI/models/checkpoints ^
@REM --lora-dir D:/COMFY_UI/ComfyUI/models/loras ^
@REM --vae-dir D:/COMFY_UI/ComfyUI/models/vae ^
@REM --text-encoder-dir D:/COMFY_UI/ComfyUI/models/text_encoders ^
@REM --embeddings-dir D:/COMFY_UI/ComfyUI/models/embeddings ^
@REM --hypernetwork-dir D:/COMFY_UI/ComfyUI/models/hypernetworks ^
@REM --controlnet-dir D:/COMFY_UI/ComfyUI/models/controlnet

:: Allow user to override GPU device ID and port via environment variables
if defined GPU_DEVICE_ID set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --gpu-device-id %GPU_DEVICE_ID%
if defined PORT set COMMANDLINE_ARGS=%COMMANDLINE_ARGS% --port %PORT%

set EXPORT COMMANDLINE_ARGS=
