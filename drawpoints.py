def create_html(image,out='index.html'):
    with open('tail.txt') as g:
        with open(out,'w') as f:
            f.write('<br/><br/><img src="'+image+'" alt="Mouse coordinates"  id="imgid" style="float:left;" />')
            f.write(g.read())
