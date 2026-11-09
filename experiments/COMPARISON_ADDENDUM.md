# Comparison addendum (Nov 2026): did I find anything above our results?
Method: 3 web searches (abstract/snippet level only; full texts NOT read). Not exhaustive.

| Source | Their task / data | Their number | Above ours? | Comparable? |
|---|---|---|---|---|
| Eur Radiol 2025 / medRxiv 2025.03.12.25322948 (EfficientNet, 7 body parts) | Private training (>860k), **MURA used as EXTERNAL test** | micro-acc 98.6 % on MURA | **Yes, numerically (0.986 vs our 0.9747)** | **No**: different training data (private, huge), zero-shot external test. Our 0.9747 is trained on MURA itself. Do NOT write "highest on MURA" without this caveat.  **[VERIFIED in full text (CITATIONS_VERIFIED.md, third pass): MURA was an external test with every image labelled 'Extremities'; micro-accuracy 98.6 %; not comparable with 7-class MURA body-part classification]** |
| Sabbar et al. 2025, iJOE 55419 (YOLOv8n-cls) | 22,500 radiographs, 11 body-part classes, dataset not identified in snippets | top-1 99.4 % | Yes | **No**: other data/classes, single-label |
| EAI 2024 (eetpht.10.5577, custom FastAI/TF CNN) | DICOM images, dataset not identified | 97.38 % | Yes (on its own data) | **No** |
| Pham 2021 (DICOM Imaging Router) | 5 coarse groups, 16,093 images | recall 0.982, F1 0.981 | Yes (on its own data) | **No** |
| Krishnan & Padmavathi 2025 (UNIFESP, EfficientNetB3) | UNIFESP 22 classes | accuracy 86.94 % | No | Partly (same dataset; metric/split unknown) |
| Shah et al. 2026 (MURA 7 classes) | MURA official valid set | acc 0.9643 / macro-F1 0.9574 | No (ours 0.9747 / 0.9707) | Yes (verify split in full text) |
| Zero-shot anatomical similarity, Sci Rep 2026 / Diagnostics 2023 (MURA per-region 89-93 %) | MURA abnormality (fracture) detection | 89.5-92.6 % per region | -- | **No**: abnormality task, not body-part identification |

## Conclusions (what can honestly be written)
0. The Eur Radiol 2025 row is now VERIFIED from the full text (CITATIONS_VERIFIED.md, third pass): MURA external accuracy 98.6 %, but every MURA image is labelled 'Extremities', so it is not comparable.
1. On the **same dataset and same task** (MURA body-part, official split, trained on MURA): Shah 2026 is the only paper found; ours is higher with CIs.
2. Several papers report HIGHER numbers (97-99 %) on OTHER data or with much larger private training sets. Do not claim "best" or "state of the art" anywhere.
3. I found **no multi-label (22-class) body-part paper** other than the UNIFESP one(s). Keep the claim modest: "to our knowledge, no prior work reports a patient-level, CI-reporting evaluation on UNIFESP" (verify on Google Scholar).
4. Every number above is from snippets: confirm in the papers' result tables before citing.
