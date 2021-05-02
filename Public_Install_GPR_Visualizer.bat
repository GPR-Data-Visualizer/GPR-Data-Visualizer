REM Echo is on for error visibility. Can be removed in final release by removing "REM" from below line.
REM @echo off 
echo Starting Installation Process...
echo This will take a while...
set folder=%PROGRAMDATA%\Anaconda3
echo Proceding with intall...

set TARGET=%PROGRAMDATA%\gpr_data_vis
set SHORTCUT=%PUBLIC%\Desktop\GPR_Data_Vis.lnk
set PWS=powershell.exe -ExecutionPolicy Bypass -NoLogo -NonInteractive -NoProfile
set BATCH=%~dp0

DEL /S /Q "%SHORTCUT%"
RMDIR /S /Q "%TARGET%"

Xcopy /E /I "%BATCH%" "%TARGET%"

call %folder%\Scripts\activate.bat
call CD /D "%TARGET%"
call conda env create --file datavis.yaml --prefix="%TARGET%\gprenv"
echo CD /D "%TARGET%" >> %TARGET%\dzt_visualizer.bat
echo call %folder%\Scripts\activate.bat >> %TARGET%\dzt_visualizer.bat
echo call conda activate "%TARGET%\gprenv" >> %TARGET%\dzt_visualizer.bat
echo call "%TARGET%\gprenv\python.exe" "%TARGET%\dzt_visualizer.py" >> %TARGET%\dzt_visualizer.bat
echo CreateObject("Wscript.Shell").Run "%TARGET%\dzt_visualizer.bat", 0, True > %TARGET%\run.vbs
%PWS% -Command "$ws = New-Object -ComObject WScript.Shell; $S = $ws.CreateShortcut('%SHORTCUT%'); $S.TargetPath = "%TARGET%\run.vbs"; $S.IconLocation = "%TARGET%\favicon.ico"; $S.Save()"
echo set TARGET=%homedrive%%homepath%\gpr_data_vis >> %TARGET%\Reinstall_Shortcut.bat
echo set SHORTCUT=%PUBLIC%\Desktop\GPR_Data_Vis.lnk >> %TARGET%\Reinstall_Shortcut.bat
echo set PWS=powershell.exe -ExecutionPolicy Bypass -NoLogo -NonInteractive -NoProfile >> %TARGET%\Reinstall_Shortcut.bat
echo %PWS% -Command "$ws = New-Object -ComObject WScript.Shell; $S = $ws.CreateShortcut('%SHORTCUT%'); $S.TargetPath = "%TARGET%\run.vbs"; $S.IconLocation = "%TARGET%\favicon.ico"; $S.Save()" >> %TARGET%\Reinstall_Shortcut.bat
RMDIR /S /Q "%TARGET%\readgssi"
RMDIR /S /Q "%TARGET%\.git"
RMDIR /S /Q "%TARGET%\__pycache__"
DEL /S /Q "%TARGET%\datavis.yaml"
DEL /S /Q "%TARGET%\.gitignore"


echo Installation Complete.
pause
exit

:err_msg
echo Anaconda3 Not Found. Download Anaconda3 and try again.
pause
exit
