# Cover letter (template -- fill the bracketed fields)

[Date]

Dear Editor-in-Chief [name] and members of the editorial board of Computerized Medical Imaging and Graphics,

We submit the manuscript "Leakage-controlled benchmarking of radiograph body-part classification: reliability, label audit, explainability and cross-dataset transfer on public data" for consideration as an original research article in Computerized Medical Imaging and Graphics.

**What the paper does.** Knowing which part of the body a radiograph shows is a routine need in PACS routing and cohort building, yet published accuracies are hard to compare. Splits, thresholds and uncertainty are rarely reported in enough detail, and most large studies use private data. We use only public data. On the UNIFESP body-part dataset (22 classes, 1,738 labelled radiographs) we split by patient, keep a held-out set apart, make every choice on out-of-fold predictions and report patient-level bootstrap intervals.

**What we found.** The ensemble reached a held-out sample-F1 of 0.919 (95 % CI 0.867 to 0.960) and a late-submission private Kaggle score of 0.926, between second and third place on the finished leaderboard. Nothing we tried gave a significant gain: not the ablations, not fusion, not ensembling, not pseudo-labelling. A gated CNN-Transformer fusion simply ignored its CNN branch. A random image-level split overstated the validation score by 0.0144, and tuning thresholds on the scored set added another 0.0083. On the official MURA validation set the model reached 0.975 accuracy (95 % CI 0.966 to 0.982), and zero-shot transfer through an explicit class map to MURA and FracAtlas gave 0.69 to 0.88. The ensemble was already reasonably calibrated, and it kept 94.5 % accuracy at 90 % coverage. An out-of-fold audit flagged 39 of 1,415 training images (2.8 %) as possible label errors.

**Why this journal.** Computerized Medical Imaging and Graphics is a Scopus Q1 journal in Radiology, Nuclear Medicine and Imaging (SCImago 2025), and its scope covers machine learning for medical images, including X-ray. We do not claim state-of-the-art accuracy. The paper offers a transparent protocol, honest negative results and tools for judging reliability.

**Declarations.** The study uses only public, de-identified datasets and needs no ethics approval [confirm with institution]. All Kaggle submissions made are reported in the manuscript. Code, split fingerprints and a reproduction guide are provided [repository and Zenodo link]. The manuscript is not under consideration elsewhere. Generative AI use is declared in the manuscript as the journal requires. The research received no specific funding, and we have no competing interests to declare [confirm].

**Limitations stated in the paper.** Single-centre data, a 323-image held-out set with 1-6 positives for several classes, image-level splits for the COVID and FracAtlas results, class maps for transfer and the label-audit suspects that are not clinician-reviewed, and a transductive pseudo-labelling step.

Suggested reviewers: [to be chosen by the authors; none suggested here].

Sincerely,
Sheikh Noor Abdullah and A.B.M. Kabir Hossain
Department of Information and Communication Engineering, Bangladesh Army University of Engineering & Technology (BAUET), Qadirabad Cantonment, Natore, Bangladesh
sheikhnoorabdullah02@gmail.com corresponding author: Sheikh Noor Abdullah; ORCID of S.N. Abdullah: 0009-0002-7193-4284
