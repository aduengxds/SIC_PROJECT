# Détection et comptage des maisons inondées à partir d'images aériennes

[Tiếng Việt](README.md) · [English](README.en.md) · [中文](README.zh.md) · **Français** · [한국어](README.ko.md) · [日本語](README.ja.md)

## Équipe

| Membre | GitHub |
| --- | --- |
| Phùng Duy Hoàng | [@21xDHoang](https://github.com/21xDHoang) |
| Bùi Nam Khánh | [@NK-Lync](https://github.com/NK-Lync) |
| Tào Hoàng Minh Vũ | [@aduengxds](https://github.com/aduengxds) |

Pipeline en deux étapes : un **détecteur** localise chaque maison dans l'image aérienne, un **classifieur** lit l'état de la zone découpée, puis les maisons sont comptées par classe.

- Détecteur : **Cascade R-CNN + ResNet-50-FPN**, une seule classe `house`
- Classifieur : **ConvNeXt-Tiny**, trois classes `Flooded` / `Not-Flooded` / `Partially-Flooded`
- Application web : **FastAPI**, entièrement locale, sur une machine dotée d'un GPU

| Élément | Métrique | Valeur |
| --- | --- | --- |
| Détecteur (`best_ep4`) | AP50 | 0,631 |
| Classifieur | exactitude de bout en bout | 0,6682 |
| Comptage au seuil 0,30 | 4 287 boîtes / 3 441 boîtes annotées | +24,6 % |

> Le seuil de comptage par défaut (0,30) privilégie le rappel : le total est donc **surestimé de 24,6 %**, tandis que l'erreur par image reste élevée (MAE 10,95 ; p90 18 ; max 49). Le pipeline convient à une estimation grossière et **ne permet pas de compter précisément une image donnée** — voir *Limites connues*.

## Le problème

| | |
| --- | --- |
| Entrée | une image aérienne 640×640 |
| Sortie | une boîte par maison, chacune portant l'un des trois états, et un tableau de comptage par classe |

Principales difficultés : maisons très petites, eau trouble, frontière floue entre les annotations `Flooded` et `Partially-Flooded`, images parfois vides ou extrêmement denses.

## Données

Source Roboflow, images 640×640, trois classes `Flooded`, `Not-Flooded`, `Partially-Flooded`. Total **2 778 images · 44 967 boîtes** ; la répartition est reprise telle quelle de Roboflow, et l'ensemble de test n'est utilisé qu'une seule fois, pour le rapport.

| Attribut | Valeur |
| --- | --- |
| Jeu de données | [`merged_dataset` – v1](https://universe.roboflow.com/housedetectionclassification-jugwz/merged_dataset-dk0ch) |
| Licence | **CC BY 4.0** — attribution obligatoire en cas de réutilisation |
| Format des annotations | COCO |
| Prétraitement Roboflow | Auto-orientation (EXIF supprimé), redimensionnement 640×640 par étirement ; aucune augmentation |
| Date d'export | 07/11/2024 |

| Sous-ensemble | Images | Boîtes | Remarques |
| --- | --- | --- | --- |
| Train | 2 116 | 33 145 | boîtes < 32 px : 10,7 % ; < 64 px : 48,5 % |
| Valid | 459 | 8 381 | sert au choix du modèle et du seuil |
| Test | 203 | 3 441 | 12 images ne contiennent aucune maison |

Statistiques complètes : `nha_ngap_v002_code/nha_ngap_v002_code/huan_luyen/01_thong_ke_du_lieu.txt`.

## Architecture

Les deux modèles apprennent deux tâches distinctes : le détecteur répond seulement « où est une maison », le classifieur seulement « dans quel état est-elle ». À l'inférence, les découpes proviennent des boîtes du détecteur — les annotations de référence ne sont pas utilisées.

**Détecteur** — Cascade R-CNN, backbone ResNet-50-FPN, 3 étages avec IoU 0,5 / 0,6 / 0,7, une classe `house` + fond (2 logits), ancres de 32 à 512 px (stride 4–64 × scale 8), pipeline de redimensionnement `keep_ratio` en 1333×800, sans `RandomResize`. Configuration : `huan_luyen/cau_hinh_detector.py`.

**Classifieur** — ConvNeXt-Tiny (ImageNet-1K V1), 27,82 M paramètres, sortie 768 → 3 classes, entrée 224 px, marge de découpe de 15 % à l'entraînement comme à l'inférence, 12 époques, AdamW 3e-4, label smoothing 0,05.

Pipeline par défaut : image → détecteur → score ≥ 0,30, au plus 100 boîtes → découpe (+15 %) → ConvNeXt-Tiny → comptage par classe.

## Réentraînement

Deux notebooks s'exécutent sur Colab (un T4 suffit) ; ouvrez-en un et lancez *Run all* de haut en bas :

- `nha_ngap_v002_code/nha_ngap_v002_code/huan_luyen/Colab_NhaNgap_v002_Aug.ipynb` — version avec augmentation, qui produit le `best_ep4` utilisé
- `nha_ngap_v002_code/nha_ngap_v002_code/huan_luyen/Colab_NhaNgap_v002.ipynb` — version sans augmentation ; ses chiffres de l'époque 20 se trouvent dans `02_ket_qua_detector.txt`

| Cellules | Contenu |
| --- | --- |
| 1–3 | vérification de l'environnement, installation de `mmcv` / `mmengine` / `mmdet`, contrôle que les opérations CUDA fonctionnent réellement |
| 4–5 | récupération des données, statistiques, conversion en deux formats : une classe `house` pour le détecteur et les trois classes d'origine pour le classifieur |
| 6–7 | entraînement du détecteur Cascade R-CNN, tracé des courbes d'apprentissage, évaluation du mAP sur le test |
| 8 | entraînement du classifieur ConvNeXt-Tiny sur des découpes issues des vraies boîtes |
| 9–10 | exécution détection → classification → comptage sur 203 images, synthèse et écriture des 4 fichiers de résultats |
| 11 | démonstration web Gradio |

Les points de contrôle sont sauvegardés par `save_best` selon le mAP de validation : `best_ep4` est donc **l'époque 4**, pas la dernière — à ne pas confondre avec `epoch_20` de la version sans augmentation.

## Résultats

![Résultats de détection sur une image aérienne — boîtes rouges : inondée, orange : partiellement inondée, vertes : non inondée](export/mau_ket_qua.jpg)

*Image exportée par l'application web elle-même (`export/mau_ket_qua.jpg`). Les couleurs suivent `loi_suy_luan.py` : rouge = `Flooded`, orange = `Partially-Flooded`, vert = `Not-Flooded`.*

### Détecteur

Deux points de contrôle ont des résultats, **ne mélangez pas leurs chiffres** :

| Point de contrôle | AP | AP50 | AP75 | AP_s | Source |
| --- | --- | --- | --- | --- | --- |
| `best_ep4` (défaut de l'application web) | 0,313 | 0,631 | 0,284 | 0,169 | chiffres présentés dans le rapport |
| epoch_20 (sans augmentation) | 0,277 | 0,584 | 0,233 | 0,149 | `02_ket_qua_detector.txt` |

Un AP_s faible est une limite réelle : la plus petite ancre du RPN fait 32 px (stride 4 × scale 8) et l'entraînement n'utilise qu'un seul pipeline de redimensionnement ; les maisons de quelques dizaines de pixels n'ont donc pas d'ancre bien adaptée.

### Classifieur

Sur les 8 381 découpes de validation : **exactitude 0,7195**. Sur les découpes de test (annotations de référence) : 0,6713. Avec les boîtes du détecteur : **0,6682 de bout en bout** — à peine moins, donc la découpe du détecteur n'est pas le goulot d'étranglement.

| Réel \ Prédit | Flooded | Not-Flooded | Partially-Flooded | Rappel |
| --- | --- | --- | --- | --- |
| Flooded | 1 514 | 115 | 606 | 67,7 % |
| Not-Flooded | 55 | 2 696 | 608 | 80,3 % |
| Partially-Flooded | 440 | 527 | 1 820 | 65,3 % |

Les erreurs se concentrent sur la paire `Flooded` ↔ `Partially-Flooded` (606 + 440). Le classifieur **n'a pas de classe de fond** : les boîtes parasites du détecteur sont donc quand même rangées dans l'une des trois classes.

### Comptage

| Point de contrôle | Boîtes @ 0,30 | Annotations | Écart | MAE / image avec maisons |
| --- | --- | --- | --- | --- |
| `best_ep4` | 4 287 | 3 441 | +846 (+24,6 %) | 10,95 (p90 18, max 49) |
| epoch_20 | 3 414 | 3 441 | −27 (−0,8 %) | 9,47 |

Le total de epoch_20 correspond presque aux annotations, mais c'est une **compensation d'erreurs** : certaines images sont surestimées, d'autres sous-estimées, et la MAE par image reste proche de 10. Il ne faut donc pas lire le total isolément.

## Définition des métriques

| Métrique | Définition / portée |
| --- | --- |
| AP | COCO bbox, moyenne sur IoU 0,50 : 0,05 : 0,95 |
| AP50 / AP75 | IoU 0,50 / 0,75 |
| AP_s / AP_m / AP_l | boîtes d'aire < 32² / 32²–96² / > 96² px² |
| Exactitude (classifieur) | corrects / total, mesurée sur les 8 381 découpes de validation |
| De bout en bout | classifieur appliqué aux boîtes du détecteur appariées aux annotations à IoU 0,5 |
| MAE | moyenne de \|comptage − annotation\| sur les images contenant des maisons |

L'AP est mesuré à `score_thr` 0,05 alors que le comptage utilise un seuil de 0,30 — deux échelles différentes : ne comparez pas directement l'AP et l'exactitude.

## Vérification

Relancer les 203 images avec le code de l'application web elle-même (`best_ep4`, seuil 0,30) donne 4 288 boîtes, soit **+1 boîte sur 4 287 = 0,023 %**. La classe `Flooded` correspond exactement (890/890) ; l'écart provient des différences d'OpenCV et d'architecture GPU entre Colab (Tesla T4) et la machine web (RTX 2050). La cible 4 287 est codée en dur sous le nom `MOC_4287` dans `kiem_tra_moi_truong.py`, afin que ce contrôle reste reproductible.

Détails : `nha_ngap_web/nha_ngap_web/KET_LUAN_KIEM_CHUNG.txt`.

## Documents du dépôt

| Fichier | Contenu |
| --- | --- |
| [`HUONG_DAN_CHO_BAN.txt`](nha_ngap_web/nha_ngap_web/HUONG_DAN_CHO_BAN.txt) | Guide complet d'installation et d'exécution, avec un tableau des erreurs fréquentes |
| [`KET_LUAN_KIEM_CHUNG.txt`](nha_ngap_web/nha_ngap_web/KET_LUAN_KIEM_CHUNG.txt) | Compte rendu du test des 4 287 boîtes et explication de l'écart de +1 |
| [`01_thong_ke_du_lieu.txt`](nha_ngap_v002_code/nha_ngap_v002_code/huan_luyen/01_thong_ke_du_lieu.txt) | Statistiques des 2 778 images · 44 967 boîtes par sous-ensemble |
| [`02_ket_qua_detector.txt`](nha_ngap_v002_code/nha_ngap_v002_code/huan_luyen/02_ket_qua_detector.txt) | AP par métrique du point de contrôle epoch_20 |
| [`03_ket_qua_classifier.txt`](nha_ngap_v002_code/nha_ngap_v002_code/huan_luyen/03_ket_qua_classifier.txt) | Matrice de confusion et exactitude du classifieur |
| [`04_dem_nha_test.txt`](nha_ngap_v002_code/nha_ngap_v002_code/huan_luyen/04_dem_nha_test.txt) | Tableau de comptage par classe du point de contrôle epoch_20 |

## Arborescence

```
.
├── export/                              # fichiers exportés par l'application : png/, jpg/, csv/
├── nha_ngap_v002_code/nha_ngap_v002_code/
│   ├── huan_luyen/                      # notebooks Colab, config du détecteur, 4 fichiers de résultats
│   └── web/                             # ancienne copie web, conservée pour comparaison
└── nha_ngap_web/nha_ngap_web/           # l'application principale
    ├── ung_dung.py                      # FastAPI : interface et cycle de vie du modèle
    ├── loi_suy_luan.py                  # inférence pure, indépendante du web
    ├── kiem_tra_moi_truong.py           # contrôle d'environnement en 5 étapes, option --dem-test
    ├── giao_dien/index.html
    ├── checkpoints/                     # cau_hinh_detector.py (fourni) ; 2 fichiers .pth (à ajouter)
    └── test/                            # 203 images de test
```

## Installation et exécution

### Prérequis

| Élément | Valeur |
| --- | --- |
| Système | Windows 10/11 64 bits |
| Python | **3.13 64 bits**, depuis python.org — pas la version Microsoft Store ; 3.12 et 3.14 ne permettent pas d'installer `mmcv` |
| GPU | NVIDIA ; vérifié sur une RTX 2050 4 Go |
| Pilote | R580 ou plus récent (CUDA 13.2) |
| Disque | ~6–7 Go ; le premier installateur télécharge ~2,5 Go |
| Chemin | ASCII, sans accents ni espaces |

### Points de contrôle

Le dépôt **ne contient pas** les fichiers `.pth` (`.gitignore` exclut `**/*.pth`). Les trois fichiers suivants sont obligatoires dans `nha_ngap_web/nha_ngap_web/checkpoints/` :

```
work_det/cau_hinh_detector.py                     déjà présent dans le dépôt
work_det/work_aug/best_coco_bbox_mAP_epoch_4.pth  à ajouter
work_cls/convnext_tiny_3lop.pth                   à ajouter
```

Il suffit qu'un des trois manque pour que l'application web ne démarre pas. Les deux fichiers `.pth` proviennent de la machine d'entraînement ou du Drive de l'équipe (`nha_ngap_v002_out/`).

### Étapes

```bat
cd nha_ngap_web\nha_ngap_web

:: crée le venv et télécharge ~2,5 Go, 10 à 30 minutes
CAI_DAT.bat

:: les 5 étapes doivent toutes réussir
venv\Scripts\python.exe kiem_tra_moi_truong.py

:: ouvre http://127.0.0.1:8000
CHAY_WEB.bat
```

Si le port 8000 est occupé, utilisez `CHAY_WEB.bat --cong 8080`.

Pour reproduire le chiffre du rapport (2–3 minutes, environ 4 287 boîtes attendues) :

```bat
venv\Scripts\python.exe kiem_tra_moi_truong.py --dem-test
```

L'application expose quatre points d'entrée : `GET /`, `GET /api/thong_tin`, `POST /api/du_doan`, `POST /api/export` (qui enregistre PNG/JPG/CSV dans `export/`). Le modèle est chargé une seule fois au démarrage ; comme `mmdet` n'est pas thread-safe et que le GPU ne dispose que de 4 Go, les requêtes sont mises en file plutôt que traitées en parallèle.

## Limites connues

- **Ensemble de test petit et mono-source** — 203 images, aucun test externe.
- **Annotations ambiguës** — `Flooded` et `Partially-Flooded` représentent l'essentiel de l'erreur du classifieur.
- **Répartition toujours entachée de recouvrements** — 7 groupes MD5, 49 noms de base, 3 groupes vidéo ; une déduplication et une répartition par source sont nécessaires.
- **Un total juste ne signifie pas des comptages justes** — MAE 10,95 ; p90 18 ; max 49.
- **Absence de classe de fond** — les boîtes parasites sont rangées dans l'une des trois classes, ce qui gonfle le comptage.
- **Seuil non optimisé** — les scores ne sont pas calibrés ; augmenter le seuil n'échange que précision et rappel, sans corriger la cause.
- **Petites maisons** — AP_s faible, dû à l'ancre minimale et à un entraînement à une seule échelle.
- **Pas prêt pour la production** — il faut une classe de fond, une calibration et une répartition propre avant tout comptage réel.

## Axes d'amélioration

| Priorité | Travail | Pourquoi |
| --- | --- | --- |
| P0 | Ajouter une classe de fond / de rejet au classifieur | les boîtes parasites sont aujourd'hui étiquetées comme maisons — principale source de surestimation |
| P0 | Calibrer les scores sur l'ensemble de validation | un seuil à 0,30 n'a actuellement aucune signification probabiliste |
| P1 | Choisir le seuil selon la MAE de comptage plutôt que le F1 | l'objectif est le comptage, pas la détection |
| P1 | Ancres / tuilage / échelle aléatoire pour les petites maisons | AP_s n'atteint que 0,169 |
| P2 | Dédupliquer et répartir par source | un test plus propre rend les chiffres pleinement fiables |
| P2 | Porter `max_per_img` à 150–200 | 2 images sur 203 restent plafonnées à 100 boîtes |

## Licence

- **Code source** : MIT — voir [`LICENSE`](LICENSE).
- **Données** : `merged_dataset` v1 de Roboflow, licence CC BY 4.0 — attribution obligatoire en cas de réutilisation.

---

*Traduit de [README.md](README.md). En cas d'écart de chiffres, la version vietnamienne fait foi.*
