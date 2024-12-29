@echo off
REM 定義 Raspberry Pi 的 IP 地址和密碼
set PI1=140.116.72.91
set PI2=140.116.72.89
set PASSWORD=sousokian

REM 設定參數
set RP_COUNT=49            REM 主控端指定參考點數量
set PER_RP_COUNT=300       REM 主控端指定每次封包數
set SAVE_PATH=/home/pi/Desktop/data/
set SAVE_PATH2=/home/pi/Desktop/data/

REM 迴圈執行
for /L %%i in (1,1,%RP_COUNT%) do (
    REM 在每個參考點開始前播放提醒音
    powershell -c (New-Object Media.SoundPlayer "C:\Users\keng-tse\Desktop\csi_tool\exp_script\sound\start1.mp3").PlaySync()

    echo 開始收集第 %%i 個參考點數據...

    REM 並行執行 Raspberry Pi 的數據採集
    start "" plink -ssh -pw %PASSWORD% pi@%PI1% "bash /home/pi/collect_csi.sh %%i %PER_RP_COUNT% %SAVE_PATH%"
    start "" plink -ssh -pw %PASSWORD% pi@%PI2% "bash /home/pi/collect_csi.sh %%i %PER_RP_COUNT% %SAVE_PATH2%"

    REM 使用 tasklist 等待兩個進程完成
    :WAIT_PI1
    tasklist /FI "IMAGENAME eq plink.exe" | findstr /i "plink.exe" > nul
    if errorlevel 1 (
        REM 如果 PI1 的進程結束，則繼續
        echo PI1 完成數據收集
    ) else (
        REM 如果 PI1 的進程仍在運行，則等待1秒後重試
        timeout /t 1 /nobreak > nul
        goto WAIT_PI1
    )

    :WAIT_PI2
    tasklist /FI "IMAGENAME eq plink.exe" | findstr /i "plink.exe" > nul
    if errorlevel 1 (
        REM 如果 PI2 的進程結束，則繼續
        echo PI2 完成數據收集
    ) else (
        REM 如果 PI2 的進程仍在運行，則等待1秒後重試
        timeout /t 1 /nobreak > nul
        goto WAIT_PI2
    )

    REM 播放完成音效並等待 5 秒
    powershell -c (New-Object Media.SoundPlayer "C:\Users\keng-tse\Desktop\csi_tool\exp_script\sound\next.mp3").PlaySync()

    echo 第 %%i 個參考點數據收集完成.

    REM 等待 10 秒，防止快速過渡到下個迴圈
    timeout /t 10 /nobreak > nul
)

echo 全部數據收集完成.
pause
