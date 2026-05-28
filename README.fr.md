# Âge NeuroCognitif (NCA) — Framework

*Read this in [English](README.md).*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Affiliation: UdeS](https://img.shields.io/badge/Affiliation-UdeS-green.svg)](https://www.usherbrooke.ca/)

## Présentation

Le **Framework Âge NeuroCognitif (NCA)** est un pipeline de biomarqueurs multimodal conçu pour quantifier les trajectoires individuelles de vieillissement en intégrant la neuro-imagerie structurelle (IRM) et la performance cognitive fonctionnelle.

Contrairement aux modèles traditionnels d'« Âge Cérébral » (Brain Age), l'Indice NCA réconcilie l'intégrité structurelle du cerveau avec la réalité clinique en ancrant le score final sur la performance au **MoCA (Montreal Cognitive Assessment)**, via une fusion pondérée optimisée empiriquement.

---

## 🛠 Architecture opérationnelle

Le framework fonctionne selon un pipeline à deux flux :

1.  **Flux structurel (Âge Cérébral — BA) :** exploite 625 caractéristiques morphométriques (volumes, surfaces et épaisseurs) extraites d'une IRM pondérée en T1.
2.  **Flux fonctionnel (Âge Cognitif — CA) :** utilise la fluence sémantique et le niveau d'éducation, transformés via un pipeline Box-Cox spécifique.
3.  **Fusion multimodale :** synthétise les deux marqueurs en un unique **Indice NCA**.

### Exigences de prétraitement des données

Pour garantir la reproductibilité et la compatibilité avec les modèles, les données d'entrée doivent respecter ces standards :

* **Segmentation IRM :** les scans T1 bruts doivent être traités avec **FreeSurfer** (v7.0+ recommandé) afin d'extraire les statistiques standard `aseg` et `aparc`.
* **Harmonisation des caractéristiques :** pour atténuer les biais liés aux sites dans les cohortes multicentriques, nous recommandons fortement d'harmoniser les données morphométriques avec le pipeline **[NOMIS (Neuroimaging Outcomes Multicenter International Standard)](https://git.valeria.science/medics/archives/github/nomis/)**. Cela garantit que les prédictions d'Âge Cérébral restent robustes entre différents scanners et protocoles.

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

Le framework NCA met en œuvre un **modèle de moyenne pondérée** pour générer l'indice final. Les poids ont été identifiés afin de maximiser la corrélation avec le statut neurocognitif global :

$$NCA_{Index} = (0.754 \times CA) + (0.246 \times BA)$$

* **CA (Âge Cognitif) :** poids = 0.754 (principal déterminant du statut fonctionnel).
* **BA (Âge Cérébral) :** poids = 0.246 (modérateur structurel).
* **Écart NCA (NCA Gap) :** calculé comme $Gap = NCA_{Index} - Age_{Chronologique}$. Un écart positif indique un vieillissement neurocognitif accéléré.

## 📂 Structure du dépôt

* `predict_nca.py` : script principal d'inférence pour le traitement par lots.
* `nca_brain_pipeline.joblib` : modèle RidgeCV pré-entraîné pour les caractéristiques IRM.
* `nca_cognitive_pipeline.joblib` : modèle SVR pré-entraîné pour les caractéristiques cognitives.
* `requirements.txt` : liste des paquets Python requis.

## 🎓 Citation et recherche

Si vous utilisez ce framework dans vos travaux, merci de citer notre article de méthode :

> **Roger, E. (2026).** *Anchoring Brain Age on Cognitive Performance: The Multimodal NeuroCognitive Age Framework.* Université de Sherbrooke.

## 👤 Autrice

**Elise Roger, PhD** — Professeure adjointe, Département d'imagerie médicale et de sciences de la radiation
Faculté de médecine et des sciences de la santé (FMSS)
**Université de Sherbrooke** — *Chercheuse au CdRV | Membre du RQRV* 📧 [elise.roger@usherbrooke.ca](mailto:elise.roger@usherbrooke.ca)

---

## Remerciements

Nous remercions l'équipe **IRIS Neurolab** et le **Centre de recherche sur le vieillissement (CdRV)** pour leur soutien dans le développement de ce framework.
