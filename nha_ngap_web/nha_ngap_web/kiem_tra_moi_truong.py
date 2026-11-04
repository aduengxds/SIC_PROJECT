# -*- coding: utf-8 -*-
"""CONG KIEM TRA MOI TRUONG — chay doc lap, khong can web.

Chay:  venv\\Scripts\\python.exe kiem_tra_moi_truong.py

5 buoc theo ke hoach. Chi khi ca 5 buoc XANH moi viet tiep giao dien — moi
trang tri giao dien deu vo nghia neu moi truong khong chay duoc.

    0. Cac duong dan can thiet co ton tai
    1. torch + CUDA + ten GPU
    2. import mmdet khong no           (chung minh mieng va 3 an)
    3. from mmcv.ops import nms        (chung minh mieng va 2 an)
    4. Nap duoc CA HAI checkpoint      (chung minh mieng va 1 an)
    5. Suy luan thu 1 anh trong test/  -> ra box + nhan, khong loi

Them co --dem-test de chay PHEP THU QUYET DINH: dem toan bo 203 anh test o
nguong 0.30, doi chieu voi con so 4287 ghi trong
`nha_ngap_v002_out\\04_dem_nha_test.txt`. Phep thu nay CHI dem so model tu
doan, KHONG dung nhan — dung voi rang buoc "cam gan nhan" cua du an.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import duong_dan          # noqa: E402
from duong_dan import in_utf8   # noqa: E402

# Con so ghi trong nha_ngap_v002_out/04_dem_nha_test.txt (checkpoint epoch 4,
# nguong 0.30, toan bo 203 anh test). Day la thuoc do duy nhat dang tin.
MOC_4287 = {'Flooded': 890, 'Not-Flooded': 1604, 'Partially-Flooded': 1793}
MOC_TONG = 4287
MOC_ANH = 203

DUOI_ANH = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')


def _ds_anh(thu_muc):
    return sorted(f for f in os.listdir(thu_muc)
                  if f.lower().endswith(DUOI_ANH))


def buoc_0_duong_dan():
    print('--- Buoc 0: duong dan ---')
    print('   nguon RA   : %s  (%s)' % (duong_dan.RA, duong_dan.NGUON_RA))
    thieu = []
    print('   BAT BUOC (thieu mot trong ba la web khong chay duoc):')
    for nhan, duong, co in duong_dan.kiem_tra_bat_buoc():
        print('      [%s] %-28s %s' % ('x' if co else ' ', nhan, duong))
        if not co:
            thieu.append(nhan)
    print('   TUY CHON (thieu thi chi canh bao):')
    for nhan, duong, co in duong_dan.danh_sach_tuy_chon():
        print('      [%s] %-28s %s' % ('x' if co else ' ', nhan, duong))
    if thieu:
        print('   THIEU (bat buoc):', ', '.join(thieu))
        if duong_dan.NGUON_RA != 'gui_kem':
            print('   Goi y: dat 3 mon bat buoc vao thu muc "checkpoints" nam')
            print('          ngay canh file nay (giu dung cau truc thu muc con).')
    return not thieu


def buoc_1_torch():
    print('--- Buoc 1: torch + CUDA ---')
    import torch
    print('   torch      :', torch.__version__)
    print('   torchvision:', end=' ')
    import torchvision
    print(torchvision.__version__)
    co_cuda = torch.cuda.is_available()
    print('   cuda kha dung:', co_cuda)
    if not co_cuda:
        print('   [CANH BAO] khong co GPU -> se chay CPU, cham nhung van dung.')
        return True
    print('   GPU        :', torch.cuda.get_device_name(0))
    print('   VRAM       : %.0f MiB' % (torch.cuda.get_device_properties(0).total_memory / 2 ** 20))
    print('   CUDA build :', torch.version.cuda)
    print('   kien truc  : sm_%d%d' % torch.cuda.get_device_capability(0))
    print('   wheel ho tro: %s' % ', '.join(torch.cuda.get_arch_list()))
    # Chay thu THAT chu khong doi chieu danh sach kien truc: RTX 4050/4060/4070
    # (Ada sm_89) khong co trong wheel nhung VAN chay duoc. Xem
    # loi_suy_luan._cuda_chay_duoc de biet day du ly do.
    import loi_suy_luan as loi
    if loi._cuda_chay_duoc():
        print('   [OK] GPU chay duoc ca kernel torch lan op mmcv (nms).')
    else:
        print('   [CANH BAO] GPU co nhung KHONG chay duoc -> web se dung CPU.')
        print('              Ket qua van dung, chi cham hon.')
    return True


def buoc_2_mmdet():
    print('--- Buoc 2: import mmdet ---')
    import mmdet
    print('   mmdet    :', mmdet.__version__)
    import mmengine
    print('   mmengine :', mmengine.__version__)
    import mmcv
    print('   mmcv     :', mmcv.__version__)
    return True


def buoc_3_mmcv_ops():
    print('--- Buoc 3: from mmcv.ops import nms ---')
    from mmcv.ops import nms
    print('   nms      :', nms)
    return True


def buoc_4_nap_checkpoint():
    print('--- Buoc 4: nap ca hai checkpoint ---')
    print('   dang nap classifier (nhanh, ~111 MB)...')
    import loi_suy_luan as loi
    t0 = time.perf_counter()
    _cls, ten_lop = loi.nap_classifier()
    print('   classifier OK sau %.1f giay | lop: %s'
          % (time.perf_counter() - t0, ten_lop))
    if len(ten_lop) != 3:
        print('   [LOI] so lop phai la 3, dang la %d' % len(ten_lop))
        return False

    print('   dang nap detector ban tot nhat (~278 MB, mat 10-20 giay)...')
    t0 = time.perf_counter()
    det = loi.nap_detector()
    print('   detector OK sau %.1f giay' % (time.perf_counter() - t0))
    print('   config   :', duong_dan.CONFIG_DET)
    print('   checkpoint:', os.path.basename(
        duong_dan.CKPT_DET[duong_dan.CKPT_DET_MAC_DINH][0]))
    # Giu lai cho buoc 5 qua mot thuoc tinh module de khong phai nap lai.
    buoc_4_nap_checkpoint.det = det
    buoc_4_nap_checkpoint.cls = _cls
    buoc_4_nap_checkpoint.ten_lop = ten_lop
    return True


def buoc_5_suy_luan_mot_anh():
    print('--- Buoc 5: suy luan thu 1 anh test ---')
    import loi_suy_luan as loi
    from PIL import Image

    ds = _ds_anh(duong_dan.ANH_TEST)
    if not ds:
        print('   [LOI] khong co anh nao trong', duong_dan.ANH_TEST)
        return False
    ten = ds[0]
    p = os.path.join(duong_dan.ANH_TEST, ten)
    print('   anh      :', ten)
    im = Image.open(p)
    print('   kich thuoc:', im.size)

    kq = loi.nhan_dien(buoc_4_nap_checkpoint.det,
                       buoc_4_nap_checkpoint.cls,
                       buoc_4_nap_checkpoint.ten_lop,
                       im, nguong=loi.THR)
    print('   tong box :', kq['tong'])
    for k, v in kq['dem'].items():
        print('      %-20s %d' % (k, v))
    print('   thoi gian: %.0f ms' % kq['ms'])
    if kq['boxes']:
        b0 = kq['boxes'][0]
        print('   box dau  : (%.0f, %.0f, %.0f, %.0f) %s %.2f'
              % (b0['x1'], b0['y1'], b0['x2'], b0['y2'], b0['nhan'], b0['diem']))
    return True


def dem_toan_bo_test(det, cls, ten_lop, nguong=0.30):
    """PHEP THU 4287 — dem toan bo anh test, doi chieu voi moc da ghi."""
    import loi_suy_luan as loi
    from PIL import Image

    ds = _ds_anh(duong_dan.ANH_TEST)
    print('   so anh   :', len(ds), '(moc: %d)' % MOC_ANH)
    if len(ds) != MOC_ANH:
        print('   [CANH BAO] so anh KHAC moc — ket qua se khong so sanh duoc.')

    dem = {k: 0 for k in ten_lop}
    tong = 0
    t0 = time.perf_counter()
    hong = 0
    for i, ten in enumerate(ds, 1):
        try:
            im = Image.open(os.path.join(duong_dan.ANH_TEST, ten))
            kq = loi.nhan_dien(det, cls, ten_lop, im, nguong=nguong)
            tong += kq['tong']
            for k, v in kq['dem'].items():
                dem[k] += v
        except Exception as e:
            hong += 1
            print('   [LOI] %s -> %s: %s' % (ten, type(e).__name__, str(e)[:80]))
        if i % 25 == 0 or i == len(ds):
            giay = time.perf_counter() - t0
            print('   %3d/%d  tong=%d  (%.0f giay, con ~%.0f giay)'
                  % (i, len(ds), tong, giay, giay / i * (len(ds) - i)))
    print('   thoi gian: %.0f giay' % (time.perf_counter() - t0))
    if hong:
        print('   so anh loi:', hong)

    print()
    print('   %-22s %8s %8s %8s' % ('lop', 'moc', 'dem duoc', 'lech'))
    for k in ten_lop:
        m = MOC_4287.get(k, 0)
        print('   %-22s %8d %8d %+8d' % (k, m, dem.get(k, 0), dem.get(k, 0) - m))
    print('   %-22s %8d %8d %+8d' % ('TONG', MOC_TONG, tong, tong - MOC_TONG))
    print()

    if hong:
        print('   KET LUAN: CO %d ANH LOI -> chua ket luan duoc, phai chay lai.' % hong)
        return False
    if tong == MOC_TONG and all(dem.get(k, 0) == v for k, v in MOC_4287.items()):
        print('   KET LUAN: KHOP HOAN TOAN voi moc 4287.')
        print('   => Ban chuyen sang web trung thanh tuyet doi voi notebook.')
        return True
    print('   KET LUAN: LECH MOC.')
    print('   => Neu chay tren MAY KHAC (GPU / OpenCV khac): lech vai box la')
    print('      BINH THUONG, khong phai loi code. Do la sai so lam tron cua')
    print('      phan cung: co 207 box chi cach nguong %.2f dung 1/100 diem,'
          % nguong)
    print('      chi can mot box lat la tong lech 1. Xem muc 4 cua file')
    print('      KET_LUAN_KIEM_CHUNG.txt di kem de co so lieu day du.')
    print('   => Neu chay tren DUNG may da sinh ra moc: co sai sot that o')
    print('      khau crop / resize / normalize / thu tu lop.')
    print('      Phai tim cho ra TRUOC KHI BAN GIAO, khong doan.')
    return False


def main():
    in_utf8()
    ap = argparse.ArgumentParser(description='Kiem tra moi truong web nha ngap v0.0.2')
    ap.add_argument('--dem-test', action='store_true',
                    help='Chay phep thu 4287: dem toan bo 203 anh test (2-3 phut)')
    ap.add_argument('--nguong', type=float, default=0.30)
    ap.add_argument('--cpu', action='store_true',
                    help='Ep chay CPU, bo qua GPU (cham hon nhung luon dung)')
    args = ap.parse_args()

    # Phai dat TRUOC buoc 4 (buoc 4 moi goi chon_thiet_bi).
    if args.cpu:
        os.environ['NHA_NGAP_THIET_BI'] = 'cpu'

    print('=' * 70)
    print('KIEM TRA MOI TRUONG — NHAN DIEN NHA NGAP v0.0.2')
    print('=' * 70)

    # Ba mieng va PHAI chay truoc khi import mmdet / mmcv.ops.
    from va_moi_truong import va_tat_ca
    va_tat_ca()

    buoc = [('0. duong dan', buoc_0_duong_dan),
            ('1. torch + CUDA', buoc_1_torch),
            ('2. import mmdet', buoc_2_mmdet),
            ('3. mmcv.ops.nms', buoc_3_mmcv_ops),
            ('4. nap checkpoint', buoc_4_nap_checkpoint),
            ('5. suy luan 1 anh', buoc_5_suy_luan_mot_anh)]

    ket = []
    for ten, ham in buoc:
        try:
            ok = ham()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print('   [NO] %s: %s' % (type(e).__name__, str(e)[:120]))
            ok = False
        ket.append((ten, ok))
        print('   => %s' % ('XANH' if ok else 'DO'))
        print()
        if not ok:
            print('=' * 70)
            print('DUNG LAI o buoc', ten, '- cac buoc sau khong con y nghia.')
            print('=' * 70)
            return 1

    print('=' * 70)
    print('CA 5 BUOC XANH.')
    print('=' * 70)

    if args.dem_test:
        print()
        print('=' * 70)
        print('PHEP THU 4287 — dem toan bo 203 anh test o nguong %.2f' % args.nguong)
        print('=' * 70)
        ok = dem_toan_bo_test(buoc_4_nap_checkpoint.det,
                              buoc_4_nap_checkpoint.cls,
                              buoc_4_nap_checkpoint.ten_lop,
                              nguong=args.nguong)
        return 0 if ok else 1
    else:
        print()
        print('Buoc tiep theo: chay lai voi co --dem-test de lam phep thu 4287.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
