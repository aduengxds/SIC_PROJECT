# -*- coding: utf-8 -*-
"""HANG SO DUONG DAN — noi DUY NHAT chua duong dan co dau tieng Viet.

Vi sao tach rieng mot file: hai rang buoc da biet cua may nay va vao nhau —
  1. Tham so dong lenh co dau bi hong khi truyen qua Git Bash.
  2. File .bat cua cmd.exe xu ly duong dan non-ASCII khong dang tin cay.
Nen: moi duong dan co dau duoc viet cung o DAY, con .bat va cac script khac
chi goi ten hang so. Doi cho luu du an thi chi sua file nay.

BAN GUI CHO BAN BE
------------------
Ban gui di KHONG can sua dong nao trong file nay. Chi can canh file nay co
thu muc `checkpoints\\` du ba mon:

    checkpoints\\work_det\\cau_hinh_detector.py
    checkpoints\\work_det\\work_aug\\best_coco_bbox_mAP_epoch_4.pth
    checkpoints\\work_cls\\convnext_tiny_3lop.pth

thi `RA` tu tro vao do (xem `_chon_ra()`). Khong co thi no roi ve duong dan
Desktop cua may tac gia — chi dung tren may tac gia ma thoi.
"""

import os
import sys


# --------------------------------------------------------------------------
# Chong loi in tieng Viet co dau ra console cp1252 cua Windows.
# Goi ham nay o dau moi script chay doc lap.
# --------------------------------------------------------------------------
def in_utf8():
    """Cho stdout/stderr ghi duoc UTF-8. Console cp1252 se hien o vuong thay
    vi no UnicodeEncodeError giua chung. Trong file .bat cung dat san
    PYTHONIOENCODING=utf-8 + PYTHONUTF8=1, ham nay chi la luoi an toan.

    `line_buffering=True` de khi chuyen huong ra file (`... > log.txt`) van
    thay tien do chay tung dong. Mac dinh Python dem khoi 8 KB khi stdout
    khong phai console -> ngoi cho ca phut ma file van 0 byte.
    """
    for luong in (sys.stdout, sys.stderr):
        try:
            luong.reconfigure(encoding="utf-8", errors="replace",
                              line_buffering=True)
        except (AttributeError, ValueError):
            pass


# --------------------------------------------------------------------------
# Goc web — PHAI khai TRUOC RA, vi _chon_ra() ben duoi can no.
# --------------------------------------------------------------------------
GOC_WEB = os.path.dirname(os.path.abspath(__file__))
GIAO_DIEN = os.path.join(GOC_WEB, "giao_dien")


# --------------------------------------------------------------------------
# Du an (thu muc co DAU va DAU CACH — khong bao gio truyen qua argv)
# --------------------------------------------------------------------------
DU_AN = r"C:\Users\DUY HOANG\Desktop\nhân diện nhà ngập v0.0.2"

# --------------------------------------------------------------------------
# Chon goc ket qua train (RA)
#
# Co HAI noi co the chua ket qua, xet theo thu tu:
#   1. `checkpoints\` nam ngay canh code — bo cuc cua BAN GUI CHO BAN BE.
#      Nho giu dung cau truc thu muc con (work_det\..., work_cls\...) nen
#      cac hang so ben duoi khong phai sua gi.
#   2. `nha_ngap_v002_out\` trong thu muc du an — bo cuc tren may tac gia.
#
# Dieu kien chon la CO DU BA MON, khong phai "thu muc co ton tai": mot ban
# copy thieu file se bi phat hien ngay thay vi bao loi tro sai cho.
# --------------------------------------------------------------------------
_RA_GUI_KEM = os.path.join(GOC_WEB, "checkpoints")
_RA_DESKTOP = os.path.join(DU_AN, "nha_ngap_v002_out")

# Ba mon toi thieu de web chay duoc: config detector + checkpoint detector
# mac dinh (best_ep4) + checkpoint classifier.
_MON_WORK_DET = os.path.join("work_det")
_MON_CKPT_DET = os.path.join("work_det", "work_aug",
                             "best_coco_bbox_mAP_epoch_4.pth")
_MON_CKPT_CLS = os.path.join("work_cls", "convnext_tiny_3lop.pth")


def _du_ba_mon(ra):
    """True neu `ra` co du config detector + checkpoint detector mac dinh +
    checkpoint classifier."""
    return (os.path.exists(os.path.join(ra, _MON_WORK_DET,
                                        "cau_hinh_detector.py"))
            and os.path.exists(os.path.join(ra, _MON_CKPT_DET))
            and os.path.exists(os.path.join(ra, _MON_CKPT_CLS)))


def _chon_ra():
    """Tra ve (duong dan RA, nguon).

    `nguon` la mot trong:
      'gui_kem'   — lay tu checkpoints\\ canh code (ban gui cho ban be)
      'desktop'   — lay tu thu muc du an (may tac gia)
      'thieu_file'— khong noi nao du ba mon, tam lay noi co thu muc de thong
                    bao loi tro dung cho nguoi dang ngoi
      'khong_co'  — khong co ca hai, bao loi muon hon
    """
    for ra, nguon in ((_RA_GUI_KEM, "gui_kem"), (_RA_DESKTOP, "desktop")):
        if _du_ba_mon(ra):
            return ra, nguon
    # Buoc du phong: uu tien noi co thu muc de thong bao loi tro ve dung may
    # nguoi dung, chu khong tro ve Desktop cua tac gia (vo nghia voi ho).
    for ra, nguon in ((_RA_GUI_KEM, "thieu_file"), (_RA_DESKTOP, "thieu_file")):
        if os.path.isdir(ra):
            return ra, nguon
    return _RA_DESKTOP, "khong_co"


RA, NGUON_RA = _chon_ra()

# --------------------------------------------------------------------------
# Detector (Cascade R-CNN, 1 lop 'house')
# --------------------------------------------------------------------------
# Dung DUNG ban config ma o 12 cua notebook da dung: WORK = DRIVE_OUT/work_det.
# Config nay da co metainfo.classes = ('house',) va num_classes = 1 o ca ba
# tang cascade — khac han config goc COCO 80 lop.
CONFIG_DET = os.path.join(RA, "work_det", "cau_hinh_detector.py")

# Ba checkpoint, nhan hien thi de dua len giao dien.
# Khoa (key) la ma ASCII de di qua URL/JSON cho an toan.
CKPT_DET = {
    "best_ep4": (
        os.path.join(RA, "work_det", "work_aug",
                     "best_coco_bbox_mAP_epoch_4.pth"),
        "Bản tốt nhất (mAP 0.3130, epoch 4)",
    ),
    "aug_ep20": (
        os.path.join(RA, "work_det", "work_aug", "epoch_20.pth"),
        "Bản aug đủ 20 epoch",
    ),
    "khong_aug_ep20": (
        os.path.join(RA, "work_det_khong_aug", "epoch_20.pth"),
        "Bản cũ không aug (mAP 0.2770)",
    ),
}
CKPT_DET_MAC_DINH = "best_ep4"

# --------------------------------------------------------------------------
# Classifier (ConvNeXt-Tiny, 3 lop ngap / khong ngap / ngap mot phan)
# --------------------------------------------------------------------------
CKPT_CLS = os.path.join(RA, "work_cls", "convnext_tiny_3lop.pth")

# --------------------------------------------------------------------------
# Tap test — 203 anh + _annotations.coco.json.
# Dung cho phep thu hoi quy: dem toan bo phai ra 4287 o nguong 0.30.
# Phep thu nay CHI dem so model tu doan, KHONG dung nhan -> dung voi rang buoc
# "cam gan nhan" cua du an.
#
# Ban gui cho ban be co kem `test\` canh code thi uu tien lay o do.
# --------------------------------------------------------------------------
_TEST_GUI_KEM = os.path.join(GOC_WEB, "test")
ANH_TEST = _TEST_GUI_KEM if os.path.isdir(_TEST_GUI_KEM) else os.path.join(DU_AN, "test")


# --------------------------------------------------------------------------
# Kiem tra duong dan
#
# Giu nguyen hinh dang 3-phan-tu (nhan, duong dan, co_ton_tai) cua
# kiem_tra_duong_dan(): co ba noi unpack no, doi hinh dang la vo het.
# --------------------------------------------------------------------------
def _tat_ca():
    """Danh sach day du: (nhan, duong dan, bat_buoc)."""
    can = [("Thu muc du an", DU_AN, False),
           ("Thu muc ket qua", RA, False),
           ("Config detector", CONFIG_DET, True),
           ("Checkpoint classifier", CKPT_CLS, True),
           ("Thu muc anh test", ANH_TEST, False)]
    for khoa, (duong, _nhan) in CKPT_DET.items():
        can.append(("Checkpoint det [%s]" % khoa, duong,
                    khoa == CKPT_DET_MAC_DINH))
    return can


def kiem_tra_duong_dan():
    """Tra ve danh sach (nhan, duong dan, co_ton_tai). Dung cho buoc 1 cua
    kiem_tra_moi_truong.py va cho /api/thong_tin."""
    return [(nhan, duong, os.path.exists(duong))
            for nhan, duong, _bb in _tat_ca()]


def kiem_tra_bat_buoc():
    """Chi cac muc BAT BUOC — thieu mot trong ba thi web khong chay duoc:
    config detector + checkpoint detector mac dinh + checkpoint classifier.

    Cac muc con lai (thu muc du an, thu muc anh test, 2 checkpoint detector
    phu) la TUY CHON: ban gui cho ban be khong co chung van chay binh thuong.
    """
    return [(nhan, duong, os.path.exists(duong))
            for nhan, duong, bb in _tat_ca() if bb]


def danh_sach_tuy_chon():
    """Cac muc KHONG bat buoc — thieu thi chi canh bao, khong chan."""
    return [(nhan, duong, os.path.exists(duong))
            for nhan, duong, bb in _tat_ca() if not bb]
