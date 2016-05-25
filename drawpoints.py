def create_html(image, out='index.html'):
    with open('tail.txt') as g:
        with open(out, 'w') as f:
            f.write('<html><body><div id="wrapper"><img src="'+image+'" alt="Mouse coordinates"  id="imgid"  />')
            f.write(g.read())
