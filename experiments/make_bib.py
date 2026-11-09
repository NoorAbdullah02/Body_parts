import re
s=open("latex/refs.bib",encoding="utf-8").read()
s=re.sub(r"(?m)^%.*$","",s)
def fields(body):
    d={};i=0
    for m in re.finditer(r"(\w+)\s*=\s*",body):
        pass
    pos=0
    while True:
        m=re.compile(r"(\w+)\s*=\s*").search(body,pos)
        if not m: break
        k=m.group(1).lower(); j=m.end()
        if body[j]=="{":
            dep=0;e=j
            while True:
                if body[e]=="{":dep+=1
                elif body[e]=="}":
                    dep-=1
                    if dep==0:break
                e+=1
            d[k]=body[j+1:e]; pos=e+1
        else:
            m2=re.compile(r"[^,}\s]+").match(body,j); d[k]=m2.group(0); pos=m2.end()
    return d
ents=[]
for m in re.finditer(r"@(\w+)\{([^,]+),",s):
    st=m.end();dep=1;e=st
    while dep:
        c=s[e]
        if c=="{":dep+=1
        elif c=="}":dep-=1
        e+=1
    ents.append((m.group(1),m.group(2).strip(),fields(s[st:e-1])))
def clean(t): return re.sub(r"\s+"," ",t.replace("{\\~a}","ã").replace("\\%","\\%")).strip()
def authors(a):
    out=[]
    for p in re.split(r"\s+and\s+",a):
        if p.strip()=="others": out.append("et al."); continue
        if "," in p:
            l,f=[x.strip() for x in p.split(",",1)]
        else:
            w=p.split(); l=w[-1]; f=" ".join(w[:-1])
        ini="".join(x[0]+"." for x in re.split(r"[\s-]+",f) if x and x[0].isupper())
        out.append(l+", "+ini)
    return out
def strip(t): return re.sub(r"[{}]","",t)
items=[]
for typ,key,d in ents:
    au=authors(d.get("author","")) if d.get("author") else [strip(d.get("organization",d.get("howpublished","")))]
    if len(au)>6: a=", ".join(au[:6])+", et al."
    elif au[-1]=="et al.": a=", ".join(au)
    elif len(au)>1: a=", ".join(au[:-1])+" and "+au[-1]
    else: a=au[0]
    first=au[0].split(",")[0]; yr=d.get("year","n.d.")
    t=strip(clean(d.get("title",""))); 
    src=[]
    if d.get("journal"): 
        j=clean(d["journal"]); 
        v=d.get("volume",""); n=d.get("number",""); p=d.get("pages","").replace("--","--")
        src.append("\\textit{"+j+"}"+(" "+v if v else "")+("("+n+")" if n else "")+(":"+p if p else ""))
    elif d.get("booktitle"): src.append("In: \\textit{"+clean(d["booktitle"])+"}"+(", "+d["pages"].replace("--","--") if d.get("pages") else ""))
    elif d.get("eprint"): src.append("arXiv:"+d["eprint"])
    elif d.get("howpublished"): src.append(clean(d["howpublished"]))
    elif d.get("publisher"): src.append(clean(d["publisher"]))
    if d.get("doi"): src.append("doi:\\,"+"\\url{https://doi.org/"+d["doi"]+"}")
    elif d.get("url"): src.append("\\url{"+d["url"]+"}")
    line=a+" ("+yr+"). "+t+". "+". ".join(src)+"."
    items.append((first.lower(),yr,"\\bibitem[{"+strip(first)+" ("+yr+")}]{"+key+"}\n"+line.replace("..",".").replace("?.","?").replace("?.","?")))
items.sort()
open("latex/references.tex","w",encoding="utf-8").write("\\begin{thebibliography}{"+str(len(items))+"}\n"+"\n\n".join(i[2] for i in items)+"\n\\end{thebibliography}\n")
print(len(items))
