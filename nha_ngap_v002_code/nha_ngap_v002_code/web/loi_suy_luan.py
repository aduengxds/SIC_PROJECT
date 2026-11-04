# -*- coding: utf-8 -*-
"""LOI SUY LUAN — nap 2 model va chay mot anh. Khong biet gi ve web.

Tai hien Y HET o 12 cua `Colab_NhaNgap_v002_Aug.ipynb`: moi con so cua do an
(trong do co phep thu hoi quy 4287) deu sinh ra tu duong code do. Doi bat cu
thu gi o day — le crop, cach resize, thu tu lop, cong thuc lam tron — la doi
ket qua va phep thu 4287 mat y nghia.

Cach dung:
    from va_moi_truong import va_tat_ca
    va_tat_ca()                                  # BAT BUOC, goi truoc tat ca
    import loi_suy_luan as loi
    det = loi.nap_detector()                     # ~10-20 giay
    cls, ten_lop = loi.nap_classifier()
    kq = loi.nhan_dien(det, cls, ten_lop, anh_pil, nguong=0.30)
"""

import collections
import os
import tempfile
import time

import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageOps
from torchvision import transforms
from torchvision.models import convnext_tiny

import duong_dan

# ---------------------------------------------------------------------------
# Tham so: GIU NGUYEN nhu o 8 / o 9 / o 12 cua notebook — doi la ket qua sai.
# ---------------------------------------------------------------------------
THR = 0.30               # nguong tin cay detector (mac dinh cua thanh truot)
SIZE = 224               # kich thuoc anh dua vao classifier
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Mau khung ve len anh, theo thu tu RGB cua PIL (khong phai BGR cua OpenCV).
MAU = {
    'Flooded': (255, 40, 40),            # do
    'Partially-Flooded': (255, 165, 0),  # cam
    'Not-Flooded': (40, 220, 40),        # xanh la
}

# Crop nha them 15% moi ben — phai GIONG HET luc train (o 8).
LE_CROP = 0.15

# So crop moi lan dua vao classifier. O 12 dua CA batch mot luc; o day chia
# nho cho an VRAM 4 GB khi anh co nhieu nha. KET QUA KHONG DOI vi model dang o
# .eval() va ConvNeXt dung LayerNorm (chuan hoa theo tung mau, khong theo batch)
# — cat nho batch cho ra argmax y het.
LO_CROP = 64

_TF2 = transforms.Compose([transforms.ToTensor(),
                           transforms.Normalize(MEAN, STD)])


# ---------------------------------------------------------------------------
# Chon thiet bi — co GPU CHUA CHAC da dung duoc GPU.
#
# Wheel mmcv o day khong kem PTX (chi co cubin cua sm_75/80/86/90/100/120), nen
# GPU doi qua cu (Pascal GTX 10xx tro xuong) se bao "no kernel image is
# available" ngay lan goi nms dau tien — web chet giua chung, khong kip canh
# bao. Vi vay phai CHAY THU THAT roi moi quyet dinh.
#
# KHONG duoc thay bang phep "compute capability co nam trong danh sach kien
# truc khong": RTX 4050/4060/4070 la Ada sm_89, KHONG co trong fatbin nhung
# VAN chay duoc nho tuong thich nhi phan cung major (cubin sm_86 chay tren
# thiet bi sm_89). Doi chieu danh sach se day oan nhom GPU laptop pho bien
# nhat xuong CPU. Chay thu that thi dung cho moi doi GPU, ke ca doi sau nay.
#
# Ham duoc goi o nhieu noi (nap_detector / nap_classifier / phan_loai_crop /
# nhan_dien) nen ket qua phai duoc NHO LAI, chi chay thu MOT lan cho ca tien
# trinh.
# ---------------------------------------------------------------------------
_KT_CUDA = None


def _cuda_chay_duoc_that():
    """Chay that mot kernel torch va mot op mmcv tren GPU. True / False."""
    ten = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    try:
        # 1) Kernel cua chinh torch
        _ = float((torch.zeros(4, device='cuda:0') + 1.0).sum())
        # 2) Op cua mmcv — day moi la cho hay no: torch va mmcv co the lech
        #    nhau, nhung nms la op dau tien pipeline that su dung.
        from mmcv.ops import nms
        hop = torch.tensor([[0.0, 0.0, 10.0, 10.0],
                            [1.0, 1.0, 11.0, 11.0],
                            [50.0, 50.0, 60.0, 60.0]], device='cuda:0')
        diem = torch.tensor([0.9, 0.8, 0.7], device='cuda:0')
        nms(hop, diem, 0.5)
        torch.cuda.synchronize()
    except Exception as e:
        print()
        print('=' * 70)
        print('  [CANH BAO] Co GPU nhung GPU KHONG CHAY DUOC -> chuyen sang CPU')
        print('  GPU       : %s (sm_%d%d)' % (ten, cap[0], cap[1]))
        print('  Ly do     : %s' % str(e).strip()[:160])
        print('  Wheel chi kem nhan cho: %s' % ', '.join(
            torch.cuda.get_arch_list()))
        print('  => Ket qua VAN DUNG, chi CHAM HON. Cu de chay tiep.')
        print('  => Lan sau muon tat han canh bao: CHAY_WEB.bat --cpu')
        print('=' * 70)
        print()
        return False
    print('   GPU chay thu OK: %s (sm_%d%d)' % (ten, cap[0], cap[1]))
    return True


def _cuda_chay_duoc():
    """Nhu tren nhung co nho ket qua — chi chay thu mot lan."""
    global _KT_CUDA
    if _KT_CUDA is None:
        _KT_CUDA = _cuda_chay_duoc_that()
    return _KT_CUDA


def chon_thiet_bi():
    """'cuda:0' neu GPU THAT SU chay duoc, khong thi 'cpu'.

    Thu tu uu tien:
      1. Bien moi truong NHA_NGAP_THIET_BI (do co --cpu cua ung_dung.py va
         kiem_tra_moi_truong.py dat) — nguoi dung da noi thi khong cai lai.
      2. Khong co GPU -> 'cpu'.
      3. Co GPU -> chay thu that, xem _cuda_chay_duoc().

    THU TU BAT BUOC: _cuda_chay_duoc() import mmcv.ops, nen phai goi SAU
    va_moi_truong.va_tat_ca(). Moi duong vao hien tai deu da thoa.
    """
    ep = os.environ.get('NHA_NGAP_THIET_BI', '').strip()
    if ep:
        return ep
    if not torch.cuda.is_available():
        return 'cpu'
    return 'cuda:0' if _cuda_chay_duoc() else 'cpu'


def nap_detector(ckpt=None, config=None, thiet_bi=None):
    """Nap Cascade R-CNN. `ckpt` mac dinh la ban tot nhat (epoch 4).

    Luu y ve `init_cfg`: config khai
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained')
    nhung `init_detector` xoa dong do truoc khi dung model — NHUNG chi khi
    KHONG truyen `cfg_options` (trong mmdet/apis/inference.py do la nhanh
    `elif`). O day khong truyen cfg_options nen khong bi tai resnet50 tu mang.
    Neu sau nay phai truyen cfg_options (vi du ha `scale` khi OOM) thi PHAI
    truyen kem 'model.backbone.init_cfg': None, khong la mmdet di tai mang.
    """
    from mmdet.apis import init_detector
    ckpt = ckpt or duong_dan.CKPT_DET[duong_dan.CKPT_DET_MAC_DINH][0]
    config = config or duong_dan.CONFIG_DET
    thiet_bi = thiet_bi or chon_thiet_bi()
    return init_detector(config, ckpt, device=thiet_bi)


def nap_classifier(ckpt=None, thiet_bi=None):
    """Nap ConvNeXt-Tiny 3 lop. Tra ve (model, ten_lop).

    Thu tu lop doc THANG tu checkpoint (o 8 co luu kem 'ten_lop') — khong doan.
    Sau nay doi model thi khong phai sua code.
    """
    ckpt = ckpt or duong_dan.CKPT_CLS
    thiet_bi = thiet_bi or chon_thiet_bi()
    ck = torch.load(ckpt, map_location=thiet_bi)
    ten_lop = ck.get('ten_lop') or ['Flooded', 'Not-Flooded', 'Partially-Flooded']
    model = convnext_tiny()
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(ten_lop))
    model.load_state_dict(ck['model'])
    model = model.to(thiet_bi).eval()
    return model, list(ten_lop)


def phan_loai_crop(model_cls, ten_lop, im, boxes, thiet_bi=None):
    """`boxes` = pred_instances.bboxes cua mmdet = (x1, y1, x2, y2).

    Crop phai GIONG HET luc train: nha 0.15 moi ben, lam tron +0.5.
    (`int(... + 0.5)` la chu y cua tac gia — dung "lam gon" thanh round.)
    """
    if len(boxes) == 0:
        return []
    thiet_bi = thiet_bi or chon_thiet_bi()
    W, H = im.size
    batch = []
    for x1, y1, x2, y2 in boxes:
        bw, bh = x2 - x1, y2 - y1
        x0, y0 = max(0, int(x1 - bw * LE_CROP)), max(0, int(y1 - bh * LE_CROP))
        x3 = min(W, int(x2 + bw * LE_CROP + 0.5))
        y3 = min(H, int(y2 + bh * LE_CROP + 0.5))
        if x3 - x0 < 4 or y3 - y0 < 4:
            # Crop qua nho (box sat mep anh) -> lay ca anh, y nhu o 12.
            x0, y0, x3, y3 = 0, 0, W, H
        batch.append(_TF2(im.crop((x0, y0, x3, y3))
                            .resize((SIZE, SIZE), Image.Resampling.BILINEAR)))

    nhan = []
    with torch.no_grad():
        for i in range(0, len(batch), LO_CROP):
            khuc = torch.stack(batch[i:i + LO_CROP]).to(thiet_bi)
            out = model_cls(khuc).argmax(1).cpu().numpy()
            nhan.extend(ten_lop[j] for j in out)
    return nhan


def nhan_dien(model_det, model_cls, ten_lop, im, nguong=THR, thiet_bi=None):
    """Chay ca hai model tren mot anh PIL. Tra ve dict ket qua.

    Khoa tra ve:
        anh      — PIL.Image da ve khung
        dem      — {ten_lop: so_luong}
        tong     — tong so box giu lai
        boxes    — [{x1,y1,x2,y2,nhan,diem}]
        ms       — thoi gian suy luận (mili giay)
        ten_lop  — thu tu lop
    """
    from mmdet.apis import inference_detector
    thiet_bi = thiet_bi or chon_thiet_bi()

    t0 = time.perf_counter()

    im = im.convert('RGB')
    # Anh dien thoai thuong kem the EXIF "Orientation"; PIL KHONG tu xoay theo
    # the do, con trinh duyet thi CO — khong xu ly la anh gui len bi quay ngang.
    # Anh dataset (Roboflow) khong co the nay nen buoc nay la no-op voi phep
    # thu 4287.
    try:
        im = ImageOps.exif_transpose(im)
    except Exception:
        pass

    # Ghi ra file tam roi dua DUONG DAN cho mmdet — y het cach o 9 / o 12 da
    # chay dung. Dua mang numpy thang vao co the lech kenh mau BGR/RGB va
    # chua tung duoc kiem chung.
    fd, p = tempfile.mkstemp(suffix='.png')
    os.close(fd)
    try:
        im.save(p)
        r = inference_detector(model_det, p)
    finally:
        try:
            os.remove(p)
        except OSError:
            pass

    b = r.pred_instances.bboxes.cpu().numpy()
    sc = r.pred_instances.scores.cpu().numpy()
    giu = sc >= nguong
    b, sc = b[giu], sc[giu]

    nhan = phan_loai_crop(model_cls, ten_lop, im, b, thiet_bi)
    dem = collections.Counter(nhan)

    dr = ImageDraw.Draw(im)
    for (x1, y1, x2, y2), n in zip(b, nhan):
        dr.rectangle([float(x1), float(y1), float(x2), float(y2)],
                     outline=MAU.get(n, (255, 255, 255)), width=2)

    ms = (time.perf_counter() - t0) * 1000.0
    return {
        'anh': im,
        'dem': {k: dem[k] for k in ten_lop},
        'tong': int(len(b)),
        'boxes': [{'x1': float(x1), 'y1': float(y1),
                   'x2': float(x2), 'y2': float(y2),
                   'nhan': n, 'diem': float(s)}
                  for (x1, y1, x2, y2), n, s in zip(b, nhan, sc)],
        'ms': ms,
        'ten_lop': ten_lop,
    }
