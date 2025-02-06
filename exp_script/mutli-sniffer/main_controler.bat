@echo off
REM 定義 Raspberry Pi 的 IP 地址和密碼
set PI1=140.116.72.91
set PI2=140.116.72.89
set PASSWORD=sousokian

REM 設定參數
set RP_COUNT=49            REM 主控端指定參考點數量
set PER_RP_COUNT=300       REM 主控端指定每次封包數
set SAVE_PATH=/home/pi/Desktop/mutli/data/
set SAVE_PATH2=/home/pi/Desktop/mutli/data/

REM 迴圈執行
for /L %%i in (1,1,5) do (
    REM 在每個參考點開始前播放提醒音
    wmplayer /play /close "C:\Users\keng-tse\Desktop\csi_tool\exp_script\sound\start1.mp3"
    timeout /t 1 /nobreak > nul
    echo start to collect %%i rp...

    REM 並行執行 Raspberry Pi 的數據採集
    start plink -ssh -batch -v -pw %PASSWORD% pi@%PI1% "sudo bash /home/pi/Desktop/mutli/csi_fingerprint_collection_step2.sh %%i %PER_RP_COUNT% %SAVE_PATH%" 
    timeout /t 1 /nobreak > nul
    start plink -ssh -batch -v -pw %PASSWORD% pi@%PI2% "sudo bash /home/pi/Desktop/mutli/csi_fingerprint_collection_step2.sh %%i %PER_RP_COUNT% %SAVE_PATH2%" 

    REM 使用 tasklist 等待兩個進程完成
    :WAIT_PI1
    tasklist /FI "IMAGENAME eq plink.exe" | findstr /i "plink.exe" > nul
    if errorlevel 1 (
        REM 如果 PI1 的進程結束，則繼續
        echo PI1 finish
    ) else (
        REM 如果 PI1 的進程仍在運行，則等待1秒後重試
        timeout /t 1 /nobreak > nul
        goto WAIT_PI1
    )

    :WAIT_PI2
    tasklist /FI "IMAGENAME eq plink.exe" | findstr /i "plink.exe" > nul
    if errorlevel 1 (
        REM 如果 PI2 的進程結束，則繼續
        echo PI2 finish
    ) else (
        REM 如果 PI2 的進程仍在運行，則等待1秒後重試
        timeout /t 1 /nobreak > nul
        goto WAIT_PI2
    )

    REM 播放完成音效並等待 5 秒
    wmplayer /play /close "C:\Users\keng-tse\Desktop\csi_tool\exp_script\sound\next.mp3"
    echo %%i rp collection is finished.

    REM 等待 10 秒，防止快速過渡到下個迴圈
    timeout /t 10 /nobreak > nul
)

echo All collections have already finished.
pause
