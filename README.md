# Brain Tumor MRI Classification (CSE 475, Group 08)

Four-class brain tumor classification from MRI images (glioma, meningioma, pituitary tumor, no tumor).
This repository is the course project for **CSE 475: Machine Learning**, East West University.

> **Project status:** Task 1 (EDA + related-work review) is complete. Model training and evaluation are **not done yet**, so no classifier results are reported here. This README will be updated as each task is finished.

---

## Group 08

| Name | Student ID |
| --- | --- |
| Md. Zahid Hossain | 2024-1-60-074 |
| Nahidul Islam | 2022-3-60-072 |
| Mehjabin Mithila | 2023-2-60-001 |

| | |
| --- | --- |
| **University** | East West University, Dept. of Computer Science & Engineering |
| **Course** | CSE 475, Machine Learning (Section 02) |
| **Semester** | Fall 2026 |
| **Instructor** | Dr. Raihan Ul Islam, Associate Professor |

---

## Track

- **Problem type:** Multi-class image classification (4 classes)
- **Application area:** Medical imaging, brain tumor classification from MRI
- **Approach:** Deep learning / CNN-based classification (planned)

<!-- TODO: if the course assigns a named track (e.g. Track A / Track B), write it here. -->

---

## Dataset

**Brain Tumor MRI Dataset** by M. Nickparvar (Kaggle):
<https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset>

| Property | Value |
| --- | --- |
| Total images | 7,200 (all `.jpg`) |
| Split | Training 5,600 / Testing 1,600 |
| Classes | `glioma`, `meningioma`, `notumor`, `pituitary` |
| Per class | 1,400 Training + 400 Testing (perfectly balanced) |
| Color modes | RGB 4,129; grayscale (L) 3,067; RGBA 3; P 1 |
| Image sizes | 447 unique sizes; 512x512 is most common (5,014 images) |
| Corrupted files | 0 |
| Exact duplicates | 187 (171 in Training, 16 in Testing) |
| Train/Test overlap | 0 identical images |

The dataset is **not** included in this repository. Download it from Kaggle (see "How to run").
Scanner, MRI sequence and patient information are not provided with the dataset, so a patient-level split cannot be verified.

---

## Repository structure

```
.
├── code/           # Notebooks and scripts (Task 1 EDA notebook)
├── models/         # Trained models (empty for now, training not started)
├── related_work/   # Literature comparison table of 5 reviewed papers
├── report/         # Written reports (Task 1 report)
└── README.md
```

| File | Description |
| --- | --- |
| `cse475-brain-tumor-mri.ipynb` | Task 1 EDA notebook (Kaggle) |
| `Related_Work_Brain_Tumor_MRI.docx` | Related-work comparison table (5 IEEE papers) |
| `Group08_Brain-Tumor-MRI_task1_report.docx` | Task 1 report |

---

## How to run

### Option A: Kaggle (recommended, this is how the notebook was run)

1. Open a new Kaggle notebook and upload `cse475-brain-tumor-mri.ipynb`.
2. Click **+ Add Input** and add the dataset `masoudnickparvar/brain-tumor-mri-dataset`.
3. Click **Run All**. The notebook finds the folder that contains `Training` and `Testing` automatically.

### Option B: Local machine

1. Download and unzip the dataset from Kaggle so you have a folder containing `Training/` and `Testing/`.
2. Install the requirements:
   ```bash
   pip install numpy pandas matplotlib seaborn opencv-python pillow jupyter
   ```
3. In the first code cell, the notebook searches `/kaggle/input`. Replace it with your local path:
   ```python
   BASE = r"D:\path\to\brain-tumor-mri-dataset"   # folder that contains Training and Testing
   ```
   and remove the `os.walk("/kaggle/input")` loop.
4. Start Jupyter and run all cells:
   ```bash
   jupyter notebook code/cse475-brain-tumor-mri.ipynb
   ```

All random sampling uses seed `42`, so the results are reproducible.

---

## Task 1: EDA and cleaning

The notebook covers: folder structure, class balance, image size and color mode, visual inspection, pixel statistics, corrupted files, exact duplicates (MD5), train/test leakage, blur (Laplacian variance) and brightness.

### Main findings

- Classes are perfectly balanced in the original release (ratio 1.00).
- Images are very heterogeneous: 447 sizes and mixed grayscale/RGB, which suggests different sources or scanners.
- No corrupted files and no exact copies shared between Training and Testing.
- Some images contain burned-in annotations (for example a number in the corner).
- In a visual sample, the mean image of `notumor` looks like an axial view while the tumor classes look more coronal. A model might learn view or contrast instead of the tumor, so this should be checked later (e.g. with Grad-CAM).
- Blur score (Laplacian variance) is strongly right-skewed (median 68.99, mean 984.99). The blurriest samples still show clear tumors, so images are **not** removed by blur score.

### Pixel statistics (1,000 Training images, grayscale, 128x128, scale 0-1)

| Mean | Std |
| --- | --- |
| 0.1816 | 0.1954 |

### Cleaning rule

Duplicates are removed from the **Training** set only. The **Testing** set is kept unchanged as a fixed benchmark (its 16 duplicates are kept and should be mentioned when reporting results).

| Class (Training) | Original | After cleaning | Removed | Class weight |
| --- | --- | --- | --- | --- |
| glioma | 1,400 | 1,400 | 0 | 0.969 |
| meningioma | 1,400 | 1,386 | 14 | 0.979 |
| pituitary | 1,400 | 1,362 | 38 | 0.997 |
| notumor | 1,400 | 1,281 | 119 | 1.060 |
| **Total** | **5,600** | **5,429** | **171** | |

Final dataset: **7,029 images** (5,429 Training + 1,600 Testing). Max/min class ratio after cleaning is about 1.09.

### Preprocessing plan for later tasks

- Resize to 224x224; use 3 channels for ImageNet-pretrained models or 1 channel for a custom CNN.
- Normalize with the Training mean/std above (computed on Training data only).
- Create a stratified validation split from the cleaned Training set; never tune on the Testing set.
- Use light augmentation (small rotation, flip, brightness change).
- Metrics: accuracy, macro precision / recall / F1, per-class recall and confusion matrix.

---

## Related work

Five IEEE conference papers (2022 to 2026) on MRI brain tumor detection were reviewed. Full table: `related_work/Related_Work_Brain_Tumor_MRI.docx`.

| Paper | Task | Best model | Reported accuracy | Our check |
| --- | --- | --- | --- | --- |
| Sivamurugan et al. (2026) | 4-class | CBAM-Hybrid Net | 97.8% | 96.4% from confusion matrix (380/394) |
| Singh et al. (2025) | 4-class | DenseNet169 | 97.45% | About 96.4% from per-class counts |
| Keerthan et al. (2025) | Segmentation + classification | U-Net + EfficientNet-based CNN | Not reported ("XX%") | IoU 0.79 (train) vs 0.37 (val) |
| Anshul et al. (2024) | Binary | Custom CNN + segmentation | 95% | Not verifiable |
| Yadav et al. (2022) | Binary, 64x64 | VGG-16 | 96% | Training curves stay below reported value |

These numbers are context only. The papers use different datasets and splits, so we will re-run baselines on our own cleaned split instead of copying their results.

---

## Results

| Task | Status | Result |
| --- | --- | --- |
| Task 1: EDA + related work | Done | See sections above |
| Model training (custom CNN, ResNet50, DenseNet121, EfficientNet) | Not started | n/a |
| Evaluation (macro-F1, confusion matrix, training/inference time) | Not started | n/a |
| Explainability (Grad-CAM) | Not started | n/a |

**No classifier has been trained yet, so no accuracy is claimed.**

---

## Planned work

- Train the same architectures under one identical preprocessing and split.
- Report macro-F1, per-class recall, confusion matrices, training time and inference time.
- Add Grad-CAM to check that the model looks at the tumor and not at view/contrast shortcuts.

---

## References

1. C. Sivamurugan et al., "An MRI Scan Brain Tumor Detection Using Deep Learning Techniques," IEEE ICISCoIS, 2026.
2. L. Singh et al., "Multi-Class Brain Tumor Detection Using CNN-Based Medical Imaging Analysis," IEEE ICDT, 2025.
3. D. S. Keerthan et al., "Brain Tumor Detection Using Deep Learning-Based CNN Model," IEEE SENNET, 2025.
4. Anshul et al., "Brain Tumor Detection from MRI," IEEE ICAC2N, 2024.
5. H. Yadav et al., "Brain Tumor Detection with MRI Images," IEEE CISES, 2022.
6. M. Nickparvar, "Brain Tumor MRI Dataset," Kaggle. <https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset>
