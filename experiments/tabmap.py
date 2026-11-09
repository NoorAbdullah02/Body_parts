"""Table renumbering shared by make_latex.py and build_docx.py (old working numbers -> numbers in the paper)."""
import re
MAINMAP = {16: 1, 1: 2, 3: 3, 11: 4, 4: 5, 7: 6, 9: 7}
SUPPMAP = {2: 1, 5: 2, 6: 3, 8: 4, 10: 5, 12: 6, 13: 7, 14: 8, 15: 9}
MAIN = set(MAINMAP)
def fixref(s):
    lab = lambda n: ("Table %d" % MAINMAP[n]) if n in MAINMAP else ("Supplementary Table S%d" % SUPPMAP[n])
    def sub(mm):
        a, sep, b = int(mm[2]), mm[3], mm[4]
        return lab(a) if not b else "%s %s %s" % (lab(a), sep, lab(int(b)))
    return re.sub(r"(Supplementary )?Tables? (\d+)(?: (and|to) (\d+))?", sub, s)
