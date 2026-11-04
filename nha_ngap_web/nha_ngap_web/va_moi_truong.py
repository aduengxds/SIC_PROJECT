# -*- coding: utf-8 -*-
"""BA MIENG VA BAT BUOC truoc khi nap model.

Chep NGUYEN VAN tu o 2 cua `Colab_NhaNgap_v002_Aug.ipynb` (da chay that tren
Colab, sinh ra moi con so cua do an). Khong "don dep" hay viet lai — sua mot
chi tiet nho la mat tinh tuong duong voi ban da kiem chung.

Vi sao can tung mieng:

  1. va_torch_load()  — torch >= 2.6 doi mac dinh torch.load(weights_only=True),
     nhung checkpoint mmengine luu kem object HistoryBuffer -> UnpicklingError.
     DA TAI HIEN THAT tren may nay: `best_coco_bbox_mAP_epoch_4.pth` nem
     UnpicklingError, trong khi `convnext_tiny_3lop.pth` nap binh thuong.
     => BAT BUOC, khong va la init_detector chet.

  2. va_mmengine()    — mmengine 0.10.7 dang ky lop Adafactor tu HAI nguon
     (vong dir(torch.optim) va `from transformers import Adafactor`) -> KeyError
     khi `from mmcv.ops import nms`. Tren Colab co san transformers nen luon no.
     O venv nay KHONG cai transformers (metadata mmengine 0.10.7 xac nhan no chi
     nam trong extras [all]/[tests]) nen nhieu kha nang la no-op — nhung va la
     vo hai va re, cu de phong.

  3. va_mmdet()       — mmdet/__init__.py ghim `mmcv_maximum_version = '2.2.0'`
     -> import mmdet no AssertionError. => BAT BUOC.

THU TU GOI: va_mmdet() -> va_mmengine() -> va_torch_load() -> roi moi
init_detector. Dung `va_tat_ca()` cho gon.

Ca ba deu IDEMPOTENT: goi lai bao nhieu lan cung duoc, tu kiem tra da va chua.
"""

import os
import re
import torch


def va_mmengine():
    """Chan loi trung ten Adafactor trong mmengine.

    Thay moi loi goi `OPTIMIZERS.register_module(` bang `_va_reg(` (mot ham chan
    tra trung truoc khi dang ky) va chen helper do vao DAU moi file.

    Ba dieu kien bat buoc, thieu cai nao cung hong (da kiem chung that):
      (a) Doc ten theo DUNG thu tu mmengine dung: kwarg `name` -> tham so vi tri
          a[0] -> module.__name__. Ban chi so module.__name__ KHONG chan duoc
          `name='TorchAdafactor'` (khoa registry la TorchAdafactor, khong phai
          Adafactor) -> loi quay lai y nguyen du da va.
      (b) Chen o DAU file, dung bam vao `^def`: zero_optimizer.py chi co
          decorator tren `class`, khong co dong ^def nao. Chen ngay truoc
          `class` thi chui vao giua decorator -> SyntaxError.
      (c) Phai va LAI duoc: lan chay sau gap file da co _va_reg ma bo qua thi
          helper hong cu o lai. Phai re.sub xoa helper cu roi chen ban moi.
    """
    import importlib.util
    spec = importlib.util.find_spec('mmengine')
    if spec is None:
        return 'mmengine: CHUA CAI'
    root = os.path.join(os.path.dirname(spec.origin), 'optim')
    helper = (
        "def _va_reg(*a, **kw):\n"
        "    _ten = kw.get('name', a[0] if len(a) > 0 else None)\n"
        "    _force = kw.get('force', a[1] if len(a) > 1 else False)\n"
        "    _mod = kw.get('module', a[2] if len(a) > 2 else None)\n"
        "    if _mod is None:\n"
        "        def _d(_cls):\n"
        "            _va_reg(_ten, _force, _cls)\n"
        "            return _cls\n"
        "        return _d\n"
        "    if _ten is None:\n"
        "        _ten = getattr(_mod, '__name__', None)\n"
        "    try:\n"
        "        if not _force and _ten is not None and _ten in OPTIMIZERS.module_dict:\n"
        "            return None\n"
        "    except Exception:\n"
        "        pass\n"
        "    return OPTIMIZERS.register_module(_ten, _force, _mod)\n"
        "\n"
        "\n"
    )
    n, loi = 0, []
    # os.walk chu KHONG dung glob: glob bo qua moi thu muc bat dau bang dau cham
    # (vi du mmdet/.mim), con os.walk thi vao duoc.
    for dirpath, _dn, files in os.walk(root):
        for fn in files:
            if not fn.endswith('.py'):
                continue
            p = os.path.join(dirpath, fn)
            s = open(p, encoding='utf-8').read()
            if 'OPTIMIZERS.register_module(' not in s and 'def _va_reg(' not in s:
                continue
            # Bo helper cu (neu phien truoc da va bang ban hong) de thay ban moi.
            s = re.sub(r'^def _va_reg\(.*?^    return OPTIMIZERS\.register_module\([^\n]*\n',
                       '', s, flags=re.S | re.M)
            if 'def _va_reg(' in s:
                loi.append(fn)
                continue
            s = s.replace('OPTIMIZERS.register_module(', '_va_reg(')
            # Chen helper o DAU file (sau dong __future__ neu co).
            moc = None
            for mm2 in re.finditer(r'^from __future__ import .*$', s, re.M):
                moc = mm2
            vitri = (moc.end() + 1) if moc else 0
            s2 = s[:vitri] + helper + s[vitri:]
            try:
                compile(s2, p, 'exec')
            except SyntaxError:
                loi.append(fn)
                continue
            open(p, 'w', encoding='utf-8').write(s2)
            n += 1
    tb = 'mmengine: da va %d file (chan loi Adafactor trung ten)' % n
    if loi:
        tb += ' | BO QUA (loi cu phap): ' + ', '.join(loi)
    return tb


def va_mmdet():
    """Noi tran mmcv cho phep tu 2.2.0 len 2.3.0.

    mmdet/__init__.py dong 9: mmcv_maximum_version = '2.2.0'. Wheel mmcv dang
    dung la 2.2.0+a8073c7... — nhung ban co local version van bi coi la "2.2.0"
    nen assert co the qua; giu mieng va nay vi notebook da dung va no vo hai.

    DUNG `import mmdet` o day: import la no AssertionError ngay. Phai dung
    importlib.util.find_spec de lay duong dan.
    """
    import importlib.util
    spec = importlib.util.find_spec('mmdet')
    if spec is None:
        return 'mmdet: CHUA CAI'
    p = os.path.join(os.path.dirname(spec.origin), '__init__.py')
    s = open(p, encoding='utf-8').read()
    if "mmcv_maximum_version = '2.3.0'" in s:
        return 'mmdet: da va truoc do'
    s2 = re.sub(r"mmcv_maximum_version\s*=\s*'2\.2\.0'",
                "mmcv_maximum_version = '2.3.0'", s)
    if s2 == s:
        return 'mmdet: KHONG thay dong can va (bao lai de kiem tra)'
    open(p, 'w', encoding='utf-8').write(s2)
    return 'mmdet: da noi mmcv_maximum_version -> 2.3.0'


def va_torch_load():
    """Mo lai weights_only=False cho torch.load.

    torch >= 2.6 doi mac dinh thanh weights_only=True. Checkpoint mmengine luu
    kem object HistoryBuffer -> bi chan -> UnpicklingError khi init_detector /
    test / resume. Moi checkpoint o day deu do chinh minh train ra (hoac
    torchvision chinh thuc) nen mo lai dung nhu truoc torch 2.6.
    """
    if getattr(torch.load, '_da_va', False):
        return 'torch.load: da vá truoc do'
    goc = torch.load

    def load_mo(*a, **kw):
        kw.setdefault('weights_only', False)
        return goc(*a, **kw)

    load_mo._da_va = True
    torch.load = load_mo
    return 'torch.load: da mo weights_only=False (torch %s)' % torch.__version__


def va_tat_ca():
    """Goi ca ba mieng va theo dung thu tu, in ket qua ra man hinh."""
    for ham in (va_mmdet, va_mmengine, va_torch_load):
        print('  [vá]', ham())


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from duong_dan import in_utf8
    in_utf8()
    print('=== VA MOI TRUONG ===')
    va_tat_ca()
    print('=== XONG ===')
