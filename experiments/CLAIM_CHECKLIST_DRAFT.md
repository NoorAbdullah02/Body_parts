# CLAIM 2024 checklist -- draft status (fill the manuscript section numbers; items marked TODO need author input)
Source: Tejani et al., Radiol Artif Intell 2024;6(4):e240300 [VERIFY item wording against the official checklist before submission]

| Area | Item (paraphrased) | Status | Where |
|---|---|---|---|
| Title/abstract | Identify as AI study; structured abstract | done | Abstract |
| Intro | Clinical need, objectives | done | Sec. 1 |
| Methods: data | Sources, inclusion, licences, de-identification | done (licences VERIFY) | Sec. 3, 9 |
| | Preprocessing steps | done | Sec. 4 |
| | Annotation: who labelled, reference standard | PARTIAL (UNIFESP labels by residents per dataset page; VERIFY) | Sec. 3 |
| | Data partitions; no leakage; patient-level | done | Sec. 3, 4 |
| Methods: model | Architecture, training, hyperparameters, seeds | done | Sec. 4 |
| | Ensemble/selection decisions on validation only | done (OOF) | Sec. 4 |
| Methods: evaluation | Metrics, CIs, statistical tests | done | Sec. 4 |
| | Robustness / sensitivity analyses | done (ablations, transfer) | Sec. 5.4-5.8 |
| | Explainability methods | done (Grad-CAM; clinician rating TODO) | Sec. 5.9 |
| | Failure analysis | partial (audit + per-class; confusion table not added) | Sec. 5.7 |
| Results | Flow of data, demographics of the sample | MISSING: UNIFESP has no patient demographics; state it | Sec. 3 |
| | Performance with CIs | done | Sec. 5 |
| Discussion | Limitations, bias, generalisability | done | Sec. 7 |
| Other | Registration, funding, code/data availability | TODO (GitHub + Zenodo; funding) | Sec. 9 |
