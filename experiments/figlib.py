"""Figure numbering by order of appearance. Markers: [[FIG|key|file|caption]]; references: {{key}}. process() returns the text with numeric markers."""
import re
def process(md):
    keys = re.findall(r"\[\[FIG\|([^|]+)\|", md)
    num = {k: i + 1 for i, k in enumerate(keys)}
    md = re.sub(r"\[\[FIG\|([^|]+)\|", lambda m: "[[FIG|%d|" % num[m[1]], md)
    md = re.sub(r"\{\{([a-z0-9_]+)\}\}", lambda m: str(num.get(m[1], "?")), md)
    figs = re.findall(r"\[\[FIG\|(\d+)\|([^|]+)\|([^\]]+)\]\]", md)
    return md, figs
