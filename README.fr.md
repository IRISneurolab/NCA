# Âge NeuroCognitif (NCA) — Framework

*Read this in [English](README.md).*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Affiliation: UdeS](https://img.shields.io/badge/Affiliation-UdeS-green.svg)](https://www.usherbrooke.ca/)

## Présentation

La **Boîte à outils de l'Âge NeuroCognitif (NCA)** est un pipeline de biomarqueurs multimodal conçu pour quantifier les trajectoires individuelles de vieillissement en intégrant la neuroimagerie structurelle (IRM) et la performance cognitive (fluence verbale sémantique et le Montreal Cognitive Assessment MoCA).

---

## 🛠 Architecture opérationnelle

Le framework fonctionne selon un pipeline à deux flux :

1.  **Flux structurel (Âge Cérébral — BA) :** exploite 625 caractéristiques morphométriques (volumes, surfaces et épaisseurs) extraites d'une IRM pondérée en T1.
2.  **Flux fonctionnel (Âge Cognitif — CA) :** utilise la fluence verbale sémantique et le niveau d'éducation, transformés via un pipeline Box-Cox spécifique.
3.  **Fusion multimodale :** synthétise les deux marqueurs en un unique **Indice NCA**.

### Exigences de prétraitement des données

Pour garantir la reproductibilité et la compatibilité avec les modèles, les données d'entrée doivent respecter ces standards :

* **Segmentation IRM :** les scans T1 bruts doivent être traités avec **FreeSurfer** (v7.0+ recommandé) afin d'extraire les statistiques standard `aseg` et `aparc`.
* **Harmonisation des caractéristiques :** pour atténuer les biais liés aux sites dans les cohortes multicentriques, nous recommandons fortement d'harmoniser les données morphométriques avec le pipeline NOMIS (NOrmative Morphometry Image Statistics) [https://git.valeria.science/medics/archives/github/nomis/]. Cela garantit que les prédictions d'Âge Cérébral restent robustes entre différents scanners et protocoles.

---

## 🚀 Démarrage rapide

### Prérequis
* **Python 3.9+**
* Les dépendances listées dans `requirements.txt` (en particulier `scikit-learn 1.6.1`).

### Installation
```bash
git clone https://github.com/IRISneurolab/NCA.git
cd NCA
pip install -r requirements.txt
```

### Lancer le prédicteur
Assurez-vous que votre fichier d'entrée (p. ex. `demo_nca_master.csv`) contient les 625 caractéristiques IRM harmonisées ainsi que `chron_age`, `sex`, `fluency`, `education` et `language`.

```bash
python predict_nca.py
```

## 🧠 Théorie de l'intégration multimodale

Le pipeline NCA met en œuvre un **modèle de moyenne pondérée** pour générer l'indice final. Les poids ont été identifiés afin de maximiser la corrélation avec le statut neurocognitif global :

$$NCA_{Index} = (0.754 \times CA) + (0.246 \times BA)$$

* **CA (Âge Cognitif) :** poids = 0.754 (principal déterminant du statut fonctionnel).
* **BA (Âge Cérébral) :** poids = 0.246 (modérateur structurel).
* **Écart NCA (NCA Gap) :** calculé comme $Gap = NCA_{Index} - Age_{Chronologique}$. Un écart positif indique un vieillissement neurocognitif accéléré.

## 📂 Structure du dépôt

**Fichiers d'inférence principaux (Racine) :**
* 📄 `predict_nca.py` : Script d'inférence principal pour le traitement par lots.
* 📄 `nca_brain_pipeline.joblib` : Modèle RidgeCV pré-entraîné pour les caractéristiques IRM.
* 📄 `nca_cognitive_pipeline.joblib` : Modèle SVR pré-entraîné pour les caractéristiques cognitives.
* 📄 `requirements.txt` : Liste des packages Python requis.

**Dossier de développement et de reproductibilité :**
* 📂 `development_workflow/` (dossier) : Scripts appuyant le développement, le benchmarking et l'évaluation statistique des modèles, incluant l'entraînement de BA et CA, la validation croisée imbriquée, la comparaison des modèles de fusion, les analyses *out-of-fold*, les fichiers de configuration et un *smoke test* sur données synthétiques. Ce workflow est fourni à des fins de reproductibilité méthodologique et n'est pas requis pour estimer le NCA à l'aide des modèles pré-entraînés.

---

## 👤 Auteure

**Élise Roger, PhD** <br>
Professeure adjointe, Dép. d'imagerie médicale et de sciences des radiations <br>
Faculté de médecine et des sciences de la santé (FMSS), **Université de Sherbrooke** <br>
Chercheuse au Centre de recherche sur le vieillissement (CdRV) <br>
📧 [elise.roger@usherbrooke.ca](mailto:elise.roger@usherbrooke.ca)

---

## Remerciements

Ce travail a été soutenu par les Instituts de recherche en santé du Canada (IRSC), le Fonds de recherche du Québec – Santé (FRQS), le réseau pancanadien AGE-WELL, le Centre de recherche sur le vieillissement (CdRV) et l'Université de Sherbrooke.
