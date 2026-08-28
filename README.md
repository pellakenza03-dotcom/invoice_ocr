# Invoice OCR service

## Utiliser PP-StructureV3 depuis Python

La configuration CPU et les modèles sont définis dans `config/ocr.json`. Le
service charge les modèles une seule fois par processus :

```python
from src.ocr import get_ocr_service

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
.\venv\Scripts\python.exe -m src.ocr .\data\input\FA-2026-003.pdf
```

La commande lit `config/ocr.json`, exécute PP-StructureV3 et sauvegarde uniquement
les résultats JSON natifs dans `data/ocr`. Utiliser
`--config chemin\ocr.json` pour sélectionner une autre configuration.

## Tests

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```
