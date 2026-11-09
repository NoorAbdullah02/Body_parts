import glob
from PIL import Image, ImageChops
for f in glob.glob("latex/figures/*.png"):
    im=Image.open(f).convert("RGB")
    bg=Image.new("RGB",im.size,(255,255,255))
    diff=ImageChops.difference(im,bg).convert("L").point(lambda p:255 if p>12 else 0)
    bb=diff.getbbox()
    if not bb: continue
    m=14; x0,y0,x1,y1=bb
    box=(max(0,x0-m),max(0,y0-m),min(im.width,x1+m),min(im.height,y1+m))
    if box!=(0,0,im.width,im.height):
        im.crop(box).save(f); print(f,im.size,"->",(box[2]-box[0],box[3]-box[1]))
