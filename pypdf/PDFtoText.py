from pypdf import PdfReader
texts = []
textname = '準則'
s = 60
p = 64

reader = PdfReader("example.pdf")

for i in range(s, p):
    text = reader.pages[i].extract_text(extraction_mode="layout", layout_mode_space_vertically=False)
    texts.append(text)
    
f = open(f'{textname}.txt','w', encoding='UTF-8')
f.writelines(texts)
f.close
