"""Blind radiologist review sheet for the 39 suspected label errors (given label shown, model label hidden). uv run --with pydicom --with matplotlib --with pandas --with numpy python make_review_pack.py"""
import zipfile, io, re, numpy as np, pandas as pd
import pydicom, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
Z = zipfile.ZipFile("../Dataset.zip"); names = [n for n in Z.namelist() if n.endswith(".dcm")]
by_uid = {re.search(r"([\d.]+)-c\.dcm$", n).group(1): n for n in names}
sus = pd.read_csv("results/reliability/label_audit_suspects_kaggle.csv")
rng = np.random.RandomState(7); order = rng.permutation(len(sus)); sus = sus.iloc[order].reset_index(drop=True)   # shuffle: no pattern for the reader
sus["code"] = ["R%02d" % (i + 1) for i in range(len(sus))]
def load(uid):
    ds = pydicom.dcmread(io.BytesIO(Z.read(by_uid[uid])))
    a = ds.pixel_array.astype(float)
    a = a * float(getattr(ds, "RescaleSlope", 1) or 1) + float(getattr(ds, "RescaleIntercept", 0) or 0)
    lo, hi = np.percentile(a, [0.5, 99.5]); a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1": a = 1 - a
    return a
CLASSES = "Abdomen, Ankle, Cervical Spine, Chest, Clavicles, Elbow, Feet, Finger, Forearm, Hand, Hip, Knee, Lower Leg, Lumbar Spine, Others, Pelvis, Shoulder, Sinus, Skull, Thigh, Thoracic Spine, Wrist"
with PdfPages("clinician_review/Radiologist_review_sheet.pdf") as pdf:
    fig = plt.figure(figsize=(8.27, 11.69)); fig.text(0.07, 0.93, "Review of possible label errors in a public radiograph dataset", fontsize=15, fontweight="bold")
    txt = ("Purpose: a research paper on radiograph body-part classification flagged 39 training images whose given label may be wrong.\n"
           "Your task (about 30 minutes): for each image, say whether the GIVEN label is correct for the main body part shown.\n\n"
           "How to answer: on each case write 1 = given label correct, 2 = given label wrong (write the correct body part), 3 = cannot tell.\n"
           "Several images show more than one body part; a label is correct if it names a body part that is clearly shown.\n"
           "Please return the answers in review_answers_template.csv (or on paper). One reviewer is fine.\n\n"
           "Notes:\n - Images are de-identified, from the public UNIFESP X-Ray Body Part Classification dataset (CC BY-NC-SA 4.0).\n"
           " - This is not a diagnosis and no patient information is involved. Do not share the images further.\n"
           " - The model's opinion is hidden on purpose, so that it does not influence you.\n\n"
           "Possible body-part names: " + CLASSES + ".")
    fig.text(0.07, 0.88, txt, fontsize=10, va="top", wrap=True); pdf.savefig(fig); plt.close(fig)
    for pg in range(0, len(sus), 4):
        fig, axs = plt.subplots(2, 2, figsize=(8.27, 11.69)); fig.subplots_adjust(hspace=0.22, wspace=0.08, top=0.96, bottom=0.04, left=0.05, right=0.95)
        for k, ax in enumerate(axs.ravel()):
            i = pg + k; ax.axis("off")
            if i >= len(sus): continue
            r = sus.iloc[i]; ax.imshow(load(r.uid), cmap="gray"); ax.set_title("%s  given label: %s" % (r.code, r.given.replace("+", " + ")), fontsize=9, fontweight="bold", loc="left")
            ax.text(0, -0.04, "Answer: 1 correct | 2 wrong: ________ | 3 unsure", transform=ax.transAxes, fontsize=8.5, va="top")
        pdf.savefig(fig); plt.close(fig)
sus[["code", "given"]].assign(answer_1_2_3="", correct_body_part_if_2="", comment="").to_csv("clinician_review/review_answers_template.csv", index=False)
sus[["code", "uid", "patient_id", "given", "model_says", "p_model_class", "p_given_max"]].to_csv("clinician_review/answer_key_DO_NOT_SHARE.csv", index=False)
print(len(sus), "cases")
