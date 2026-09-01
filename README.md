# Invoice OCR service

## Utiliser PP-StructureV3 depuis Python

La configuration CPU et les modèles sont définis dans `config/ocr.json`. Le
service charge les modèles une seule fois par processus :

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

La commande lit `config/ocr.json`, exécute PP-StructureV3 et sauvegarde dans
`data/ocr` le JSON natif ainsi que l'image annotée de détection du layout
(`*_layout_det_res.png`). Utiliser `--config chemin\ocr.json` pour sélectionner
une autre configuration.

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
chaque page PDF en PNG à 200 DPI et copie les JPG/PNG existants sans les
modifier :

```powershell
python -m prepare_layout_data
```

Les valeurs par défaut sont équivalentes à :

```powershell
python -m prepare_layout_data --input .\data\layout_audit\input --output .\data\layout_training --dpi 200
```

Le dossier `data/layout_training/images` contient une image par page et
`data/layout_training/manifest.csv` conserve le lien entre chaque image, son
document source et son numéro de page. Par sécurité, la commande refuse de
remplacer un dataset de sortie déjà rempli.

## Générer les pré-annotations de layout

Cette commande charge uniquement `PP-DocLayout_plus-L` sur CPU. Elle ne lance
ni l'OCR, ni la reconnaissance des tableaux, ni les autres modules de
PP-StructureV3. Tester d'abord une image avec le profil permissif :

```powershell
python -m layout_detect ".\data\layout_training\images\fa-2026-003__p001.png" --profile low_threshold
```

Après validation visuelle, traiter toutes les images séquentiellement avec une
seule instance du modèle :

```powershell
python -m layout_detect ".\data\layout_training\images" --profile low_threshold
```

Les JSON contenant les boîtes/classes/scores et les images annotées sont
enregistrés dans `data/layout_training/preannotations/low_threshold`.

## Tests

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```
