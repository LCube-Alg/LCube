# Identifying Causal Direction via Dense Functional Classes (LCube)


## Reproduce the Experiments
- To execute the experiments, run run.py after updating the file path to specify the location of your dataset.
-  Results of experiments are saved in the `/results` directory


##  **Hyperparameter** 

The maximum number of knots (m_max) controls the precision of the spline estimation and the precision of the causal score:

### Recommended m_max values grouped by size:

**m = 10:** AN, AN-s, LS, MN-U

**m = 4:**  SIM, SIM-G, Mult, Net, LS-s

**m = 2:** SIM-ln, SIM-c, Cha

 m = n / 40:  Tübingen

Use smaller m_max for simpler or datasets with smaller n, and larger or adaptive values for more complex datasets.

## 📊 Evaluation Metrics

### ✅ Accuracy (ACC)
From:  
**J.M. Mooij, J. Peters, D. Janzing, J. Zscheischler, B. Schölkopf**  
*Distinguishing cause from effect using observational data: Methods and benchmarks.*  
JMLR 17(32):1–102, 2016.

📎 [Link to paper](https://webdav.tuebingen.mpg.de/cause-effect/)

---

### 📐 Area Under Decision Rate Curve (AUDRC)
From:  
**A. Immer, C. Schultheiss, J.E. Vogt, B. Schölkopf, P. Bühlmann, A. Marx**  
*On the identifiability and estimation of causal location-scale noise models.*  
ICML 2023, pp. 14316–14332. PMLR.

📎 [Link to paper](https://proceedings.mlr.press/v202/immer23a.html)

## 📂 Download the Datasets

# Methods and References

### AN, AN-s, LS, LS-s, MN-U  
**N. Tagasovska, V. Chavez-Demoulin, T. Vatter.**  
*Distinguishing cause from effect using quantiles: Bivariate quantile causal discovery.*  
ICML, pp. 9311–9323. PMLR, 2020.  
[https://github.com/tagas/bQCD](https://github.com/tagas/bQCD)

---

### SIM, SIM-c, SIM-ln, SIMG  
**J.M. Mooij, J. Peters, D. Janzing, J. Zscheischler, B. Schölkopf.**  
*Distinguishing cause from effect using observational data: Methods and benchmarks.*  
JMLR 17(32):1-102, 2016.  
[https://webdav.tuebingen.mpg.de/cause-effect/](https://webdav.tuebingen.mpg.de/cause-effect/)

---

### Multi, Net  
**O. Goudet, D. Kalainathan, P. Caillou, I. Guyon, D. Lopez-Paz, M. Sebag.**  
*Learning functional causal models with generative neural networks.*  
*Explainable and Interpretable Models in Computer Vision and Machine Learning*, pp. 39–80, 2018.  
[https://arxiv.org/abs/1709.05321](https://arxiv.org/abs/1709.05321)

---

### Cha  
**I. Guyon, A. Statnikov, B.B. Batu.**  
*Cause Effect Pairs in Machine Learning.*  
Springer, 2019.  
[https://link.springer.com/book/10.1007/978-3-030-21810-2](https://link.springer.com/book/10.1007/978-3-030-21810-2)

---

### Tübingen  
*Tübingen Cause-Effect Pairs*  
 Available at: [https://webdav.tuebingen.mpg.de/cause-effect/](https://webdav.tuebingen.mpg.de/cause-effect/)

## 🚀 Setup

📦 Install dependencies:
pip install -r requirements.txt

## 🛠️ **Dependencies**

- Python 3.x  
- NumPy  
- SciPy  




