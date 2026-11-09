Source: printed output of noorabdullah02/cross-validation (UNIFESP_v2_cross), Version 1, 55m21s. FracAtlas (both directions) CRASHED on truncated PNGs (OSError); not included. mura->unifesp n=184 = 184 of the 1,606 single-label UNIFESP training images fall in the 6 shared groups (per-group n: elbow 23, finger 15, forearm 12, hand 31, shoulder 40, wrist 63); unifesp->mura used 12,000 of 40,005 MURA images (random subset, per-group n shown in the log).

FracAtlas rows (added Nov 2026, noorabdullah02/unifesp-v2-cross-fracatlas, 2677 s, successful; zero-shot, no fine-tuning):
 - n = single-label FracAtlas images (3685) mapped to groups; mura->fracatlas uses only hand+shoulder (1382 images; hand n=1281, shoulder n=101: the result is dominated by 'hand').
 - Many FracAtlas JPEGs are truncated and were decoded partially (warnings in the log); count not tallied here.
 - Accuracy CIs are IMAGE-level bootstrap (no patient ids in FracAtlas) -> optimistic; do not compare with patient-level CIs.
 - Class map for UNIFESP->FracAtlas (hand,hip,leg,shoulder) is my own mapping, same status as the MURA map: needs clinician check.
