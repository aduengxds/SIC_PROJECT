# -*- coding: utf-8 -*-
"""WEB — FastAPI. Chi lo giao dien va vong doi model.

Suy luan nam o `loi_suy_luan.py`, khong biet gi ve web — dung khuon da dung o
v0.0.1 (`scripts/du_doan.py` la loi, `scripts/web_du_doan.py` chi lo giao dien).
Nho vay `kiem_tra_moi_truong.py` kiem chung duoc loi ma khong can dung web.

Chay:
    venv\\Scripts\\python.exe ung_dung.py                 (mo trinh duyet)
    venv\\Scripts\\python.exe ung_dung.py --khong-mo-trinh-duyet

Ba endpoint:
    GET  /               -> giao_dien/index.html
    GET  /api/thong_tin  -> checkpoint kha dung, thu tu lop, ten GPU, mau
    POST /api/du_doan    -> multipart: anh, nguong, model -> JSON co anh base64

Hai quyet dinh quan trong:

  * Model nap MOT LAN luc khoi dong, khong nap moi request — Cascade R-CNN mat
    10-20 giay de nap, moi request ma nap la khong dung duoc.

  * `KHOA` (threading.Lock) + `--workers 1`: model mmdet khong thread-safe va
    chi co mot GPU 4 GB. Hai nguoi bam cung luc thi XEP HANG, khong chay song
    song. Day la lua chon dung, khong phai han che.
"""

import argparse
import base64
import io
import os
import socket
import sys
import threading
import webbrowser
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import duong_dan
from duong_dan import in_utf8

# PHAI va moi truong truoc khi bat cu thu gi cham toi mmdet/mmcv.
from va_moi_truong import va_tat_ca

va_tat_ca()

import torch                                    # noqa: E402
from PIL import Image                           # noqa: E402
import uvicorn                                  # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, UploadFile   # noqa: E402
from fastapi.responses import FileResponse      # noqa: E402

import loi_suy_luan as loi                      # noqa: E402


# ---------------------------------------------------------------------------
# Trang thai chia se
# ---------------------------------------------------------------------------
KHOA = threading.Lock()      # bao ve CA viec nap model LAN suy luan (cung cham GPU)

TT = {
    'det': None,             # detector dang giu trong VRAM
    'khoa_det': None,        # khoa checkpoint cua no
    'cls': None,
    'ten_lop': None,
    'thiet_bi': None,
}

# Duoc main() gan truoc khi uvicorn.run. Tach ra vi lifespan can doc chung.
CAU_HINH = {'cong': 8000, 'mo_trinh_duyet': True}


def ip_lan():
    """IP trong mang LAN. Thu thuat UDP: khong gui goi nao, chi de he dieu hanh
    chon card mang dung — tranh lay nham card ao cua VirtualBox/VMware."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        return s.getsockname()[0]
    except Exception:
        return '127.0.0.1'
    finally:
        s.close()


# ---------------------------------------------------------------------------
# Vong doi model
# ---------------------------------------------------------------------------
def _nap_lai_detector(khoa):
    """Nap detector `khoa`, giai phong ban cu truoc. Goi khi DANG giu KHOA.

    VRAM chi co 4 GB: khong du chua hai ban (278 MB + 558 MB) cung luc, nen
    buoc giai phong la bat buoc chu khong phai toi uu.
    """
    if TT['det'] is not None and TT['khoa_det'] == khoa:
        return False                       # dang dung dung ban can -> khong lam gi
    if TT['det'] is not None:
        TT['det'] = None
        TT['khoa_det'] = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    duong, _nhan = duong_dan.CKPT_DET[khoa]
    TT['det'] = loi.nap_detector(ckpt=duong, thiet_bi=TT['thiet_bi'])
    TT['khoa_det'] = khoa
    return True                            # co nap lai that (request se cham)


@asynccontextmanager
async def lifespan(app):
    print()
    print('Dang nap model (mat 10-25 giay)...')
    print('   nguon RA :', duong_dan.RA, '(%s)' % duong_dan.NGUON_RA)
    TT['thiet_bi'] = loi.chon_thiet_bi()
    print('   thiet bi :', TT['thiet_bi'])

    TT['cls'], TT['ten_lop'] = loi.nap_classifier(thiet_bi=TT['thiet_bi'])
    print('   classifier:', TT['ten_lop'])

    TT['det'] = loi.nap_detector(thiet_bi=TT['thiet_bi'])
    TT['khoa_det'] = duong_dan.CKPT_DET_MAC_DINH
    print('   detector :', os.path.basename(
        duong_dan.CKPT_DET[duong_dan.CKPT_DET_MAC_DINH][0]))

    ip = ip_lan()
    print()
    print('=' * 66)
    print('  WEB DANG CHAY')
    print('  May nay  : http://127.0.0.1:%d' % CAU_HINH['cong'])
    print('  Dien thoai: http://%s:%d   (cung mot wifi)' % (ip, CAU_HINH['cong']))
    print('  Dung: bam Ctrl+C trong cua so nay.')
    print('=' * 66)
    print()

    if CAU_HINH['mo_trinh_duyet']:
        # Cho 1 giay cho uvicorn kip mo cong roi moi goi trinh duyet.
        threading.Timer(1.0, lambda: webbrowser.open(
            'http://127.0.0.1:%d' % CAU_HINH['cong'])).start()

    yield

    TT['det'] = None
    TT['cls'] = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title='Nhận diện nhà ngập v0.0.2', lifespan=lifespan)


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
@app.get('/')
def trang_chu():
    p = os.path.join(duong_dan.GIAO_DIEN, 'index.html')
    if not os.path.exists(p):
        raise HTTPException(500, 'Thiếu tệp giao diện: %s' % p)
    return FileResponse(p, media_type='text/html')


@app.get('/api/thong_tin')
def thong_tin():
    ds = []
    for khoa, (duong, nhan) in duong_dan.CKPT_DET.items():
        ds.append({
            'khoa': khoa,
            'nhan': nhan,
            'co': os.path.exists(duong),
            'mac_dinh': khoa == duong_dan.CKPT_DET_MAC_DINH,
        })
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        'ten': 'Nhận diện nhà ngập v0.0.2',
        'thiet_bi': TT['thiet_bi'],
        'gpu': gpu,
        'ten_lop': TT['ten_lop'] or [],
        # Doi sang ma mau HTML de giao dien to dung mau voi khung tren anh.
        'mau': {k: '#%02x%02x%02x' % tuple(v) for k, v in loi.MAU.items()},
        'checkpoints': ds,
        'khoa_det': TT['khoa_det'],
        'nguong_mac_dinh': loi.THR,
        'san_sang': TT['det'] is not None,
    }


@app.post('/api/du_doan')
def du_doan(anh: UploadFile = File(...),
            nguong: float = Form(loi.THR),
            model: str = Form(duong_dan.CKPT_DET_MAC_DINH)):
    """Khai bao `def` (khong phai `async def`) la CO Y: FastAPI se day vao
    threadpool, nen suy luận nang khong chan event loop — nguoi khac van tai
    duoc trang trong luc cho."""
    if model not in duong_dan.CKPT_DET:
        raise HTTPException(400, 'Khóa checkpoint không hợp lệ: %s' % model)
    nguong = max(0.01, min(0.99, float(nguong)))

    raw = anh.file.read()
    if not raw:
        raise HTTPException(400, 'Không nhận được dữ liệu ảnh.')
    try:
        im = Image.open(io.BytesIO(raw))
        im.load()
    except Exception as e:
        raise HTTPException(400, 'Không đọc được ảnh: %s' % e)

    with KHOA:
        try:
            nap_lai = _nap_lai_detector(model)
        except Exception as e:
            raise HTTPException(500, 'Không nạp được checkpoint %s: %s' % (model, e))
        try:
            kq = loi.nhan_dien(TT['det'], TT['cls'], TT['ten_lop'], im,
                               nguong=nguong, thiet_bi=TT['thiet_bi'])
        except Exception as e:
            raise HTTPException(500, 'Lỗi khi suy luận: %s' % e)

    buf = io.BytesIO()
    kq['anh'].save(buf, format='JPEG', quality=88)
    return {
        'dem': kq['dem'],
        'tong': kq['tong'],
        'boxes': kq['boxes'],
        'ms': kq['ms'],
        'ten_lop': kq['ten_lop'],
        'nguong': nguong,
        'model': model,
        'nap_lai': nap_lai,
        'kich_thuoc': list(im.size),
        'anh_ket_qua': 'data:image/jpeg;base64,'
                       + base64.b64encode(buf.getvalue()).decode('ascii'),
    }


# ---------------------------------------------------------------------------
def main():
    in_utf8()
    ap = argparse.ArgumentParser(description='Web nhận diện nhà ngập v0.0.2')
    ap.add_argument('--cong', type=int, default=8000, help='Cổng (mặc định 8000)')
    ap.add_argument('--may-chu', default='0.0.0.0',
                    help='Địa chỉ bind. 0.0.0.0 = mở cho cả LAN (mặc định)')
    ap.add_argument('--khong-mo-trinh-duyet', action='store_true')
    ap.add_argument('--khong-doi-chieu', action='store_true',
                    help='Bỏ qua kiểm tra đường dẫn lúc khởi động')
    ap.add_argument('--cpu', action='store_true',
                    help='Ép chạy CPU, bỏ qua GPU (chậm hơn nhưng luôn đúng)')
    args = ap.parse_args()

    CAU_HINH['cong'] = args.cong
    CAU_HINH['mo_trinh_duyet'] = not args.khong_mo_trinh_duyet

    # Phải đặt TRƯỚC khi uvicorn chạy lifespan (lifespan mới gọi chon_thiet_bi).
    if args.cpu:
        os.environ['NHA_NGAP_THIET_BI'] = 'cpu'

    if not args.khong_doi_chieu:
        # Cổng chặn chỉ tính theo mục BẮT BUỘC: config detector + checkpoint
        # detector mặc định + checkpoint classifier. Bản gửi cho bạn bè không
        # kèm thư mục test\ và 2 checkpoint detector phụ vẫn chạy bình thường.
        thieu = [n for n, _d, co in duong_dan.kiem_tra_bat_buoc() if not co]
        if thieu:
            print('!! THIẾU ĐƯỜNG DẪN BẮT BUỘC:')
            for n, d, co in duong_dan.kiem_tra_bat_buoc():
                if not co:
                    print('   ', n, '->', d)
            print('   Nguồn đang đọc:', duong_dan.RA,
                  '(%s)' % duong_dan.NGUON_RA)
            print('   Cần đủ 3 món nêu trên. Xem HUONG_DAN_CHO_BAN.txt.')
            return 1
        for n, d, co in duong_dan.danh_sach_tuy_chon():
            if not co:
                print('   [cảnh báo] thiếu (không bắt buộc):', n, '->', d)

    # workers=1 la BAT BUOC: model khong thread-safe va chi co mot GPU.
    # KHONG bat reload: reload se nap lai model moi lan luu file.
    uvicorn.run(app, host=args.may_chu, port=args.cong, workers=1,
                log_level='info')
    return 0


if __name__ == '__main__':
    sys.exit(main())
