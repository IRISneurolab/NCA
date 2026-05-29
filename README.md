# NeuroCognitive Age (NCA) Framework

*Lire en [français](README.fr.md).*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Affiliation: UdeS](https://img.shields.io/badge/Affiliation-UdeS-green.svg)](https://www.usherbrooke.ca/)

## Overview

The **NeuroCognitive Age (NCA) Framework** is a multimodal biomarker pipeline designed to quantify individual aging trajectories by integrating structural neuroimaging (MRI) and functional cognitive performance. 

Unlike traditional "Brain Age" models, the NCA Index reconciles structural brain integrity with clinical reality by anchoring the final score on **MoCA (Montreal Cognitive Assessment)** performance using an empirically optimized weighted fusion.

---

## 🛠 Operational Architecture

The framework operates through a two-stream pipeline:

1.  **Structural Stream (Brain Age - BA):** Leverages 625 morphometric features (volumes, surface areas, and thicknesses) extracted from T1-weighted MRI.
2.  **Functional Stream (Cognitive Age - CA):** Utilizes semantic fluency and educational background, transformed via a specific Box-Cox pipeline.
3.  **Multimodal Fusion:** Synthesizes both markers into a single **NCA Index**.

### Data Preprocessing Requirements

To ensure reproducibility and model compatibility, input data must follow these standards:

* **MRI Segmentation:** Raw T1 scans must be processed using **FreeSurfer** (v6 recommended) to extract the standard `aseg` and `aparc` statistics.
* **Feature Harmonization:** To mitigate site-related bias in multi-centric cohorts, we highly recommend harmonizing the morphometric data using the **[NOMIS (Neuroimaging Outcomes Multicenter International Standard)](https://git.valeria.science/medics/archives/github/nomis/)** pipeline. This ensures that the Brain Age predictions remain robust across different scanners and protocols.

---

## 🚀 Getting Started

### Prerequisites
* **Python 3.9+**
* Dependencies listed in `requirements.txt` (specifically `scikit-learn 1.6.1`).

### Installation
```bash
git clone https://github.com/IRISneurolab/NCA.git
cd NCA
pip install -r requirements.txt
```

### Running the Predictor
Ensure your input file (e.g., `demo_nca_master.csv`) contains the 625 harmonized MRI features along with `chron_age`, `sex`, `fluency`, `education`, and `language`.

```bash
python predict_nca.py
```

## 🧠 Multimodal Integration Theory

The NCA framework implements a **Weighted Mean Model** to generate the final index. The weights were identified to maximize correlation with global neurocognitive status:

$$NCA_{Index} = (0.754 \times CA) + (0.246 \times BA)$$

* **CA (Cognitive Age):** Weight = 0.754 (Primary driver of functional status).
* **BA (Brain Age):** Weight = 0.246 (Structural moderator).
* **NCA Gap:** Calculated as $Gap = NCA_{Index} - Age_{Chronological}$. A positive gap indicates accelerated neurocognitive aging.

## 📂 Repository Structure

* `predict_nca.py`: Main inference script for batch processing.
* `nca_brain_pipeline.joblib`: Pre-trained RidgeCV model for MRI features.
* `nca_cognitive_pipeline.joblib`: Pre-trained SVR model for cognitive features.
* `requirements.txt`: List of required Python packages.

## 🎓 Citation & Research

If you use this framework in your research, please cite our method article

## 👤 Author

**Elise Roger, PhD** Assistant Professor, Dept. of Medical Imaging and Radiation Sciences  
Faculty of Medicine and Health Sciences (FMSS)  
**University of Sherbrooke** *Researcher at CdRV | Member of RQRV* 📧 [elise.roger@usherbrooke.ca](mailto:elise.roger@usherbrooke.ca)

---

## Acknowledgments

This work was supported by the Canadian Institutes of Health Research (CIHR); the Fonds de recherche du Québec – Santé (FRQS); the pancanadian network AGEWELL, the Research Center ong Aging (CdRV)**  and the Université de Sherbrooke.


