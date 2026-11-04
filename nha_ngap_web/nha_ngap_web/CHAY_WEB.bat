@echo off
rem ===========================================================================
rem  KHOI DONG WEB - NHAN DIEN NHA NGAP v0.0.2
rem
rem  Bam dup file nay de mo web. Lan dau tien se hoi tuong lua Windows
rem  (xem ghi chu o cuoi file), cu dong y cho phep.
rem
rem  Chay duoc ca tham so, vi du:  CHAY_WEB.bat --cong 8080
rem
rem  Toan bo file viet ASCII khong dau: console Windows la cp1252.
rem ===========================================================================

setlocal EnableExtensions
cd /d "%~dp0"

rem --- Doi cho cai dat: sua dong nay neu doi o dia ---
set "GOC=%~dp0"
set "VPY=%GOC%venv\Scripts\python.exe"

rem Cho Python ghi tieng Viet ra console duoc
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
chcp 65001 >nul 2>nul

echo ===========================================================================
echo   WEB NHAN DIEN NHA NGAP v0.0.2
echo ===========================================================================
echo.

rem ---------------------------------------------------------------------------
rem 1) Co venv chua?
rem ---------------------------------------------------------------------------
if not exist "%VPY%" (
  echo [LOI] Chua co moi truong Python tai:
  echo       %VPY%
  echo.
  echo       Hay bam dup CAI_DAT.bat truoc de cai dat ^(mat 10-30 phut^).
  echo.
  pause
  exit /b 1
)

rem ---------------------------------------------------------------------------
rem 2) Du thu vien chua?
rem
rem  Dung importlib.util.find_spec chu KHONG import that:
rem  import mmdet se no AssertionError neu chua chay va_moi_truong.py, nen
rem  khong the dung import de kiem tra o day.
rem ---------------------------------------------------------------------------
echo [1/2] Kiem tra thu vien...
"%VPY%" -c "import importlib.util as u, sys; L=['torch','torchvision','mmcv','mmdet','mmengine','fastapi','uvicorn','PIL']; T=[m for m in L if u.find_spec(m) is None]; T += [] if (u.find_spec('multipart') or u.find_spec('python_multipart')) else ['python-multipart']; print('      THIEU: ' + ', '.join(T) if T else '      Du thu vien.'); sys.exit(1 if T else 0)"
if errorlevel 1 (
  echo.
  echo [LOI] Thieu thu vien ^(xem danh sach THIEU o tren^).
  echo       Bam dup CAI_DAT.bat de cai bo sung - no tu bo qua buoc da xong.
  echo.
  pause
  exit /b 1
)

rem ---------------------------------------------------------------------------
rem 3) Chay web
rem
rem  KHONG bat --reload cua uvicorn: reload se nap lai model moi lan luu file.
rem  Model nap MOT LAN luc khoi dong, mat 10-25 giay. Cho do la binh thuong.
rem ---------------------------------------------------------------------------
echo [2/2] Khoi dong web... ^(cho 10-25 giay de nap model^)
echo.
"%VPY%" "%GOC%ung_dung.py" %*
set "MA=1"
if not errorlevel 1 set "MA=0"

echo.
if "%MA%"=="1" (
  echo ===========================================================================
  echo   WEB DA DUNG ^(hoac gap loi^).
  echo ===========================================================================
  echo   Neu phia tren co dong bao loi, doc ky dong do truoc.
  echo   Loi thuong gap:
  echo     - "Address already in use": cong 8000 dang bi chiem.
  echo       Chay lai bang:  CHAY_WEB.bat --cong 8080
  echo     - Loi thieu tep: chay kiem_tra_moi_truong.py de xem tep nao thieu.
) else (
  echo ===========================================================================
  echo   WEB DA DUNG BINH THUONG.
  echo ===========================================================================
)
echo.
pause
exit /b %MA%

rem ===========================================================================
rem  GHI CHU VE TUONG LUA WINDOWS
rem
rem  Lan dau web bind vao 0.0.0.0, Windows Defender se hien hop thoai
rem  "Windows Security Alert" hoi co cho phep khong.
rem    - Cu tich CA "Private networks" lan "Public networks" cho chac.
rem    - Windows loc theo DUONG DAN chu khong theo ten tien trinh, nen
rem      quy tac san co cua "python.exe" he thong KHONG dung cho venv nay.
rem
rem  KHONG THAY HOP THOAI, hoac lo bam Cancel -> may khac trong LAN se bao
rem  khong ket noi duoc. Chay PowerShell bang quyen Administrator roi dan dung
rem  MOT dong nay (chi mot lan duy nhat; viet lien mot dong vi PowerShell
rem  hieu dau ^ khac cmd, dan nhieu dong se sai):
rem
rem    netsh advfirewall firewall add rule name="Nha ngap web (venv python)" dir=in action=allow program="<THU MUC BAN GIAI NEN>\venv\Scripts\python.exe" protocol=TCP profile=any enable=yes
rem
rem  Kiem tra lai xem quy tac da co chua:
rem    netsh advfirewall firewall show rule name="Nha ngap web (venv python)"
rem ===========================================================================
