# Invoice OCR service

## Utiliser PP-StructureV3 depuis Python

La configuration CPU et tous les modèles sont définis dans `config/ocr.json`.
Le service charge une seule instance de `PPStructureV3` par processus. Cette
instance contient le layout, l'OCR et la reconnaissance des tableaux :

```python
from ocr import get_ocr_service

ocr = get_ocr_service()
page_results = ocr.predict("FA-2026-003.pdf")
```

`page_results` contient un résultat natif PaddleOCR par page. Ces résultats
seront convertis vers `schemas/ocr_document.schema.json` par l'adaptateur OCR.

Pour éviter les pics de RAM sur la machine CPU de 16 Go, les appels à
`predict()` sont exécutés un par un dans le processus. Le futur worker devra
également être lancé avec une concurrence OCR de 1.

## Utiliser la commande OCR

Depuis la racine du projet :

```powershell
.\venv\Scripts\python.exe -m ocr .\data\input\FA-2026-003.pdf
```

La commande lit `config/ocr.json`, exécute PP-StructureV3 une seule fois par
page et sauvegarde dans `data/ocr` le JSON structuré complet, le JSON du layout
et son image annotée. Utiliser `--config chemin\ocr.json` pour sélectionner une
autre configuration.

Le pipeline utilise explicitement :

- `PP-DocLayout_plus-L` pour le layout ;
- `PP-OCRv5_server_det` et `latin_PP-OCRv5_mobile_rec` pour l'OCR ;
- `PP-LCNet_x1_0_table_cls` pour router les tableaux ;
- `SLANeXt_wired` et `SLANet_plus` pour leur structure ;
- les deux modèles `RT-DETR-L_*_table_cell_det` pour les cellules.

Sur une machine GPU, utiliser `--device gpu` sans modifier la configuration
locale CPU. Le fichier `requirements-kaggle.txt` exclut volontairement le wheel
CPU `paddlepaddle`, qui doit être remplacé par `paddlepaddle-gpu` dans Kaggle.

## Auditer les détections de layout

Les modèles sont chargés une seule fois, puis tous les documents du dossier sont
traités séquentiellement. Lancer d'abord le profil de référence :

```powershell
python -m ocr .\data\layout_audit\input --audit-profile default
```

Puis lancer le profil avec un seuil plus bas et sans suppression des boîtes
imbriquées :

```powershell
python -m ocr .\data\layout_audit\input --audit-profile low_threshold
```

Les résultats sont séparés dans :

```text
data/layout_audit/output/default/
data/layout_audit/output/low_threshold/
```

Comparer les images `*_layout_det_res.png`, puis enregistrer les erreurs dans
`data/layout_audit/audit.csv`. Les profils ne dupliquent pas `ocr.json` : le
profil `default` en hérite et `low_threshold` ne contient que ses surcharges.

## Préparer les images pour l'annotation du layout

Depuis la racine du projet, la commande suivante parcourt les factures, rend
chaque page PDF en PNG à 200 DPI et convertit les images existantes en PNG RGB.
Chaque page est ensuite auditée et normalisée de manière conservatrice avec
OpenCV avant la détection de layout :

```powershell
python -m prepare_layout_data
```

Les valeurs par défaut sont équivalentes à :

```powershell
python -m prepare_layout_data --input .\data\layout_audit\input --output .\data\layout_training --dpi 200
```

Le dossier `data/layout_training/images` contient une image finale par page et
`data/layout_training/manifest.csv` conserve le lien entre chaque image, son
document source et son numéro de page. Le manifeste contient également les
mesures de netteté, luminosité, contraste, orientation, perspective et
inclinaison, les corrections appliquées et le statut qualité. La rotation à
angle droit utilise `PP-LCNet_x1_0_doc_ori`. OpenCV applique ensuite, avec un
seuil de confiance conservateur, la rectification et le recadrage d'un contour
de document dominant, le redressement de faible amplitude et l'amélioration
des contrastes très faibles. Les cas ambigus sont marqués `review`. Par
sécurité, la commande refuse de remplacer un dataset de sortie déjà rempli.

Pour rendre ou convertir les pages sans lancer l'audit OpenCV :

```powershell
python -m prepare_layout_data --no-preprocess
```

Si le rendu des pages ou le prétraitement est interrompu, la commande suivante
reconstruit au besoin le manifeste, saute les pages déjà checkpointées et ne
reconvertit pas les PDF :

```powershell
python -m prepare_layout_data --resume-preprocessing
```

Pour recalculer aussi les pages déjà normalisées (par exemple après une
amélioration de la détection d'orientation) :

```powershell
python -m prepare_layout_data --resume-preprocessing --force-reprocess
```

Le même manifeste contient aussi le SHA-256, le ratio de contenu, les groupes
de doublons exacts et les colonnes d'audit. Une page multipage valide n'est
jamais supprimée automatiquement : les anomalies ont `audit_status=review` et
restent incluses jusqu'à ce que `include_in_training` soit changé de `yes` à
`no` après vérification humaine.

Pour enrichir un manifeste déjà créé sans reconvertir les PDF ni recopier les
images :

```powershell
python -m prepare_layout_data --refresh-manifest
```

## Exécuter OCR, layout et tableaux en un seul passage

Après `prepare_layout_data`, lancer le pipeline complet sur les PNG normalisés.
Une seule instance de PP-StructureV3 est chargée et chaque fichier correspond à
une page :

```powershell
python -m ocr .\data\layout_training\images `
  --audit-profile low_threshold `
  --output .\data\layout_training\preannotations\low_threshold `
  --resume `
  --build-coco
```

Pour chaque page, cet unique appel produit `*.structure.json`, `*.layout.json`
et `*.layout.png`. Une fois toutes les pages terminées, la même commande
construit `data/layout_training/annotations/preannotations.coco.json` sans
relancer l'inférence.

`--resume` valide les deux JSON et l'image layout avant de sauter une page. Une
sortie absente ou corrompue est automatiquement recalculée.

`python -m layout_detect ... --build-coco` reste disponible pour reconstruire
uniquement le COCO à partir de JSON layout existants. Son ancien service
d'inférence autonome a été supprimé.

La commande vérifie le manifeste, les images, les dimensions, les classes, les
coordonnées et l'unicité des IDs, puis écrit
`data/layout_training/annotations/preannotations.coco.json`.

## Convertir les exports Paddle vers le document OCR canonique

L'adaptateur conserve chaque ligne OCR et sa geometrie, associe les lignes aux
regions de layout fiables et convertit les tables vers
`schemas/ocr_document.schema.json`. Les tables couvrant plus de 75 % d'une page
sont ignorees comme structure suspecte, sans supprimer leur texte OCR :

```powershell
python -m ocr_adapter .\data\layout_training\preannotations `
  --output .\data\ocr_canonical
```

Les pages nommees `document__p001.structure.json`, `document__p002...` sont
regroupees dans un seul `document.ocr.json`. Chaque sortie est validee contre
le schema canonique avant son ecriture.

Le contrat de sortie métier de la première phase est défini dans
`schemas/invoice_summary.schema.json`. Il couvre l'en-tête de facture, le
vendeur, l'acheteur, les montants, la ventilation TVA et leurs preuves OCR,
sans inclure les lignes de facture.

## Extraire le résumé métier avec Qwen

Le pipeline sémantique valide le JSON OCR, construit un contexte compact avec
les coordonnées normalisées, appelle Qwen, normalise les dates et montants,
valide les preuves OCR et applique les contrôles comptables. Tester d'abord la
préparation du contexte, sans modèle et sans GPU :

```powershell
python -m invoice_extract `
  .\data\ocr_canonical\1-1-invoice-35.ocr.json `
  --output .\data\invoice_contexts `
  --context-only
```

Pour une machine NVIDIA avec 8 Go de VRAM, utiliser un environnement séparé
du pipeline PaddleOCR. Installer d'abord une version CUDA de PyTorch adaptée au
pilote de la machine, puis :

```powershell
py -3.11 -m venv .venv-extract
.\.venv-extract\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-llm.txt
python -m pip install -e .
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

La configuration par défaut charge `Qwen/Qwen3-4B` en 4 bits via
bitsandbytes, sans clé API. Le premier lancement télécharge le modèle. Lancer
ensuite une seule facture :

```powershell
python -m invoice_extract `
  .\data\ocr_canonical\1-1-invoice-35.ocr.json `
  --output .\data\invoice_results
```

En cas de mémoire GPU insuffisante, réduire `context.max_characters`,
`model.max_input_tokens` et `model.max_new_tokens` dans
`config/extraction.json`. Une fois le test unitaire validé, un dossier complet
peut être traité avec `--resume` :

```powershell
python -m invoice_extract .\data\ocr_canonical `
  --output .\data\invoice_results `
  --resume
```

## Tests

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```
