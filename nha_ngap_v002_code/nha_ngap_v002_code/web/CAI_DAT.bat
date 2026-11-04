@echo off
rem ===========================================================================
rem  CAI DAT MOI TRUONG - NHAN DIEN NHA NGAP v0.0.2
rem
rem  Bam dup file nay MOT LAN. Sau do dung CHAY_WEB.bat de mo web.
rem  Toan bo file viet ASCII khong dau: console Windows la cp1252, in tieng
rem  Viet co dau ra la hong.
rem
rem  Cai NGAY TAI thu muc chua file nay - dung chuyen di noi khac:
rem  venv phai nam canh code thi CHAY_WEB.bat moi tim thay, va thu muc nay
rem  PHAI la duong dan ASCII khong dau, khong dau cach - cmd.exe xu ly
rem  duong dan non-ASCII khong dang tin cay. Chi tiet: HUONG_DAN_CHO_BAN.txt
rem ===========================================================================

setlocal EnableExtensions
cd /d "%~dp0"

rem --- Doi cho cai dat: sua dong nay neu muon doi o dia ---
set "GOC=%~dp0"
set "VPY=%GOC%venv\Scripts\python.exe"

rem Cho Python ghi tieng Viet ra console duoc
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
chcp 65001 >nul 2>nul

echo ===========================================================================
echo   CAI MOI TRUONG - NHAN DIEN NHA NGAP v0.0.2
echo ===========================================================================
echo.
echo Thu muc cai dat: %GOC%
echo.
echo Buoc nay tai khoang 2,5 GB (chu yeu la torch ban CUDA).
echo Lan dau co the mat 10-30 phut tuy mang. Cu de cho.
echo.
pause

rem ---------------------------------------------------------------------------
rem 0) Tim Python 3.13
rem ---------------------------------------------------------------------------
echo.
echo [0/6] Tim Python 3.13...
py -3.13 -c "import sys; print('  Python', sys.version.split()[0], '64-bit' if sys.maxsize > 2**32 else '32-bit')"
if errorlevel 1 goto :loi_python

rem ---------------------------------------------------------------------------
rem 1) Tao venv
rem ---------------------------------------------------------------------------
echo.
if exist "%VPY%" (
  echo [1/6] Da co venv - bo qua buoc tao.
) else (
  echo [1/6] Tao venv tai venv\ ...
  py -3.13 -m venv "%GOC%venv"
  if errorlevel 1 goto :loi_venv
)
"%VPY%" -c "import sys; print('  venv OK:', sys.version.split()[0], '|', sys.executable)"
if errorlevel 1 goto :loi_venv

rem ---------------------------------------------------------------------------
rem 2) Nang cap pip
rem ---------------------------------------------------------------------------
echo.
echo [2/6] Nang cap pip...
"%VPY%" -m pip install --upgrade pip
if errorlevel 1 goto :loi_cai

rem ---------------------------------------------------------------------------
rem 3) torch + torchvision ban CUDA (index chinh thuc cua PyTorch)
rem ---------------------------------------------------------------------------
echo.
echo [3/6] Cai torch 2.13.0+cu132 + torchvision 0.28.0+cu132 ...
echo       (nang nhat, khoang 1,9 GB - cho lau)
"%VPY%" -m pip install torch==2.13.0+cu132 torchvision==0.28.0+cu132 --index-url https://download.pytorch.org/whl/cu132
if errorlevel 1 goto :loi_cai

rem ---------------------------------------------------------------------------
rem 4) mmcv - wheel cong dong MiroPsota (OpenMMLab ngung phat hanh tu 7/2024)
rem
rem  Dung URL TRUC TIEP, KHONG dung:
rem      --extra-index-url https://miropsota.github.io/torch_packages_builder/mmcv/
rem  Da thu that tren may nay: dang --extra-index-url thi pip chi nhin thay ban
rem  tren PyPI (2.2.0 khong hau to) roi bao "No matching distribution found",
rem  du trang index do CO chua wheel nay. Dan thang URL thi cai duoc ngay.
rem
rem  URL that chua chuoi "%2B" (ma hoa cua dau +), ma trong file .bat thi
rem  "%2" bi cmd.exe hieu la tham so dong lenh thu 2 roi rong hoa -> URL sai.
rem  Nen phai viet "%%2B" - cmd tra ve dung "%2B" cho python.
rem  (Da kiem chung rieng: %%2B trong .bat -> python nhan duoc %2B nguyen ven.)
rem
rem  Wheel: 24.070.389 byte, cp313 / win_amd64 / pt2.13.0 / cu132
rem ---------------------------------------------------------------------------
echo.
echo [4/6] Cai mmcv 2.2.0+a8073c7pt2.13.0cu132 ...
"%VPY%" -m pip install "https://github.com/MiroPsota/torch_packages_builder/releases/download/mmcv-2.2.0%%2Ba8073c7/mmcv-2.2.0%%2Ba8073c7pt2.13.0cu132-cp313-cp313-win_amd64.whl"
if errorlevel 1 goto :loi_mmcv

rem ---------------------------------------------------------------------------
rem 5) OpenMMLab
rem ---------------------------------------------------------------------------
echo.
echo [5/6] Cai mmdet 3.3.0 + mmengine 0.10.7 ...
"%VPY%" -m pip install mmdet==3.3.0 mmengine==0.10.7
if errorlevel 1 goto :loi_cai

rem ---------------------------------------------------------------------------
rem 6) Web
rem
rem  python-multipart la BAT BUOC: FastAPI can no de nhan file upload, thieu la
rem  endpoint nhan anh chet ngay.
rem  KHONG cai opencv-python-headless: mmcv keo opencv-python san, hai goi cung
rem  ghi de thu muc cv2/ se lam hong import cv2.
rem ---------------------------------------------------------------------------
echo.
echo [6/6] Cai FastAPI + uvicorn + python-multipart + pillow ...
"%VPY%" -m pip install fastapi "uvicorn[standard]" python-multipart pillow
if errorlevel 1 goto :loi_cai

rem ---------------------------------------------------------------------------
rem Xong - chay luon cong kiem tra moi truong neu co
rem ---------------------------------------------------------------------------
echo.
echo ===========================================================================
echo   CAI XONG.
echo ===========================================================================
echo.

if exist "%GOC%kiem_tra_moi_truong.py" (
  echo Dang chay cong kiem tra moi truong...
  echo.
  "%VPY%" "%GOC%kiem_tra_moi_truong.py"
  echo.
  echo Neu tat ca cac buoc deu XANH thi go CHAY_WEB.bat de mo web.
) else (
  echo Buoc tiep theo: chay kiem_tra_moi_truong.py, roi go CHAY_WEB.bat.
)
echo.
pause
exit /b 0

rem ===========================================================================
rem  Nhanh loi
rem ===========================================================================
:loi_python
echo.
echo [LOI] Khong tim thay Python 3.13.
echo       Cai Python 3.13 (64-bit) tu https://www.python.org/downloads/
echo       khi cai NHO tich "Add python.exe to PATH".
echo.
pause
exit /b 1

:loi_venv
echo.
echo [LOI] Khong tao duoc venv tai: %GOC%venv
echo       Kiem tra quyen ghi o dia va dung luong con trong.
echo.
pause
exit /b 1

:loi_mmcv
echo.
echo [LOI] Cai mmcv that bai.
echo       Buoc nay tai wheel 24 MB truc tiep tu GitHub Releases. Kiem tra:
echo         - Mang con ket noi khong
echo         - Mo duoc dia chi nay trong trinh duyet khong:
echo           https://github.com/MiroPsota/torch_packages_builder/releases
echo       Neu ban phat hanh do da bi go mat, tim wheel moi cung to hop
echo       pt2.13.0 / cu132 / cp313 / win_amd64 roi thay URL o buoc [4/6].
echo.
pause
exit /b 1

:loi_cai
echo.
echo [LOI] Cai goi that bai. Xem thong bao phia tren.
echo       Chay lai file nay duoc - cac buoc da xong se tu bo qua.
echo.
pause
exit /b 1
