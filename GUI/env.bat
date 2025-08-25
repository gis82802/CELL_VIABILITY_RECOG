@echo off
REM 手動設定環境變數，指定MyGUI環境的python路徑
set PATH=C:\anaconda\envs\MyGUI\Scripts;C:\anaconda\envs\MyGUI;%PATH%
set CONDA_PREFIX=C:\anaconda\envs\MyGUI
set CONDA_DEFAULT_ENV=MyGUI
set CONDA_PROMPT_MODIFIER=(MyGUI)

REM 打印出當前環境確認
echo Successfully switched to MyGUI environment!

REM 啟動命令列
cmd
