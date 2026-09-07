# Guide d'annotation du layout des factures

## 1. Objectif

Ce guide définit les règles d'annotation des pages utilisées pour le
fine-tuning de `PP-DocLayout_plus-L`.

Le modèle de layout doit construire une carte visuelle simple de la facture :

```text
doc_title
header
footer
text
table
image
```

Il ne doit pas extraire les champs métier. ICE, IF, RC, patente, dates,
devises, montants et lignes de facture seront traités ensuite par l'OCR, les
règles métier et le LLM.

`config/layout_labels.json` doit contenir exactement les six catégories de ce
guide avant la conversion des pré-annotations en COCO.

## 2. Principes généraux

1. Annoter chaque page séparément, y compris pour une facture multipage.
2. Attribuer une seule classe à chaque région.
3. Éviter les boîtes imbriquées décrivant le même contenu.
4. Dessiner des boîtes serrées sans couper les caractères ou les bordures.
5. Regrouper les lignes qui forment un même bloc cohérent.
6. Ne pas annoter les fonds, lignes décoratives ou espaces vides seuls.
7. Ne jamais accepter une pré-annotation uniquement à cause de son score.
8. Appliquer la même décision à toutes les mises en page similaires.

## 3. Catégories

| ID | Classe | Utilisation |
|---:|---|---|
| 1 | `doc_title` | Titre principal de la facture |
| 2 | `header` | Bloc textuel structurel en haut de page |
| 3 | `footer` | Bloc textuel structurel en bas de page |
| 4 | `text` | Tout autre bloc textuel |
| 5 | `table` | Région structurée en lignes et colonnes |
| 6 | `image` | Logo, cachet, illustration ou graphique |

### 3.1 `doc_title`

Utiliser pour le titre principal identifiant le document :

- `FACTURE` ;
- `FACTURE N° FA-002-26` ;
- `FACTURE PROFORMA` ;
- `AVOIR`.

Si le type et le numéro sont visuellement réunis, créer une seule boîte. Si
plusieurs blocs sont éloignés, annoter comme `doc_title` celui qui joue
clairement le rôle de titre principal ; l'autre devient `header` ou `text`.

Un titre situé en haut de page reste `doc_title`, pas `header`.

### 3.2 `header`

Utiliser uniquement pour des blocs textuels structurels placés en haut :

- identité et coordonnées de l'émetteur ;
- numéro, date et échéance ;
- références documentaires ;
- informations répétées au début des pages.

Ne pas créer une grande boîte englobant simultanément un logo, un titre et
plusieurs blocs indépendants :

```text
logo                    -> image
FACTURE N° FA-002-26    -> doc_title
date et échéance        -> header
coordonnées société     -> header
```

Un bloc générique placé en haut sans fonction d'en-tête claire peut rester
`text`.

### 3.3 `footer`

Utiliser pour des blocs textuels structurels visuellement placés en bas :

- mentions légales ;
- informations d'entreprise répétées ;
- coordonnées bancaires répétées ;
- numéro de page ;
- conditions constituant réellement le pied de page.

Une mention légale déplacée en haut d'une page supplémentaire à cause de la
pagination n'est pas un `footer` visuel. L'annoter `text`, sauf si elle forme
réellement un en-tête répété sur cette page.

### 3.4 `text`

Utiliser pour tout bloc textuel sans rôle plus spécifique :

- adresse du client dans le corps ;
- conditions de paiement ;
- notes et références ;
- titres de section comme `Émetteur`, `Client` ou `Règlement` ;
- titres ou légendes de tableaux et d'images ;
- phrase d'arrêté de facture ;
- ICE, IF, RC, patente ou RIB placés dans le corps ;
- texte isolé sans structure tabulaire.

Regrouper les lignes proches qui forment un même paragraphe. Séparer les blocs
appartenant à des zones ou fonctions visuellement distinctes.

Ne pas annoter comme `text` un contenu déjà couvert par une boîte `table`.

### 3.5 `table`

Utiliser pour une région organisée en lignes et colonnes, avec ou sans
bordures :

- lignes de produits ou prestations ;
- tableau de TVA ;
- récapitulatif des totaux ;
- échéancier ou historique des paiements ;
- bloc quantité, prix unitaire et montant aligné en colonnes.

La boîte `table` contient :

- les en-têtes de colonnes ;
- les cellules et leurs textes ;
- les bordures utiles ;
- toutes les lignes appartenant au même tableau.

Ne pas créer de boîtes `text` à l'intérieur du tableau.

Un récapitulatif sans bordures peut être `table` s'il présente plusieurs lignes
et des colonnes stables, par exemple des libellés à gauche et des montants
alignés à droite. Une seule paire isolée `Total TTC : valeur` reste `text`.

Le titre placé au-dessus d'un tableau reste une boîte `text` séparée. Les noms
de colonnes restent à l'intérieur de `table`.

### 3.6 `image`

Utiliser pour :

- logo ;
- cachet ou tampon ;
- signature graphique ;
- photo ou illustration ;
- graphique.

Un logo placé en haut reste `image`, pas `header`. Une image contenant du texte
stylisé, comme un logo, reste `image`.

Ne pas annoter comme image une ligne, un rectangle de fond ou une décoration
sans information.

## 4. Priorité entre les classes

Cette priorité est une règle humaine de correction. Elle n'est pas appliquée
automatiquement par le modèle pré-entraîné :

```text
table > doc_title > image > header/footer > text
```

Exemples :

- texte dans un tableau -> seulement `table` ;
- titre de facture dans l'en-tête -> `doc_title` ;
- logo dans l'en-tête -> `image` ;
- texte administratif en haut -> `header` ;
- texte administratif en bas -> `footer` ;
- titre de section ou légende -> `text`.

Le modèle peut prédire un logo comme `header`. Cette pré-annotation doit être
corrigée en `image` conformément au guide.

## 5. Géométrie des boîtes

Une bonne boîte :

- contient toute la région utile ;
- ne coupe aucun caractère ;
- inclut les bordures nécessaires à la reconnaissance d'un tableau ;
- contient peu d'espace vide extérieur ;
- ne déborde pas sur une région indépendante.

Pour un bloc de plusieurs lignes, annoter le rectangle global. Ne pas créer une
boîte par mot ou par ligne OCR.

Les boîtes peuvent se toucher, mais les chevauchements doivent rester
exceptionnels. Les annotations parent-enfant sont interdites pour un même
contenu.

## 6. Factures multipages

Chaque page reçoit ses propres annotations, mais les pages restent associées
par `source_document` et `page` dans `manifest.csv`.

1. Annoter un en-tête répété sur chaque page avec `header`.
2. Annoter un pied de page répété sur chaque page avec `footer`.
3. Annoter séparément la partie visible d'un tableau sur chaque page.
4. Ne jamais dessiner une boîte traversant deux pages.
5. Inclure dans `table` les en-têtes de colonnes répétés sur la page suivante.
6. Conserver une page peu remplie si elle appartient réellement à la facture.
7. Garder toutes les pages d'une facture dans le même split.

Les pages identiques signalées par `duplicate_group` ne sont pas supprimées.
`include_in_training` détermine si elles participent au fine-tuning.

## 7. Cas spécifiques aux factures marocaines

La nature métier du contenu ne crée pas une nouvelle classe de layout :

| Contenu | Classe selon la région |
|---|---|
| ICE, IF, RC, patente, CNSS | `header`, `footer` ou `text` |
| MAD, DH, Dhs | `text` ou contenu d'un `table` |
| Cachet de l'entreprise | `image` |
| RIB et coordonnées bancaires | `footer`, `text` ou `table` |
| Texte arabe ou bilingue | Même règle que le français |

Ne pas créer de classes `ice`, `if`, `rc`, `mad` ou `arabic`. Ces informations
seront reconnues par l'OCR puis extraites par les règles métier ou le LLM.

## 8. Correction des pré-annotations

Pour chaque page :

1. Supprimer les faux positifs.
2. Ajouter les régions manquantes.
3. Corriger les classes erronées.
4. Ajuster les boîtes trop grandes ou trop petites.
5. Supprimer les boîtes internes aux tableaux.
6. Vérifier les logos, titres, headers et footers.
7. Vérifier les éléments proches des bords de page.
8. Relire la page avant de la marquer comme terminée.

Le score de confiance sert seulement à prioriser la vérification. Il ne
représente pas la vérité terrain.

## 9. Contrôle avant export COCO

Vérifier que :

- toutes les annotations utilisent l'une des six catégories ;
- aucune boîte n'a une largeur ou une hauteur nulle ;
- aucune boîte ne sort des limites de l'image ;
- les tableaux ne contiennent pas de boîtes `text` imbriquées ;
- les logos et cachets sont `image` ;
- les titres principaux sont `doc_title` ;
- les titres de section et légendes sont `text` ;
- les pages d'une même facture restent dans le même split ;
- les décisions restent cohérentes entre les templates similaires.

## 10. Échantillon de calibration

Avant de corriger les 37 pages, appliquer ce guide à un petit échantillon :

```text
fa-002-26__p001.png
fa-002-26__p002.png
facture-auto-entrepreneur.jpg
facture-douane.jpg
facture-proforma-07-26__p001.png
```

Valider les décisions sur ces cinq pages, puis utiliser exactement les mêmes
règles pour tout le dataset.
