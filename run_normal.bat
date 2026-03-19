@echo off
call ../environment.bat
call env_normal.bat
call webui.bat

echo HF_HOME = %HF_HOME%
echo XDG_CACHE_HOME = %XDG_CACHE_HOME%
echo HF_DATASETS_CACHE = %HF_DATASETS_CACHE%
echo COMMANDLINE_ARGS = %COMMANDLINE_ARGS%
