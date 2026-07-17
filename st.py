import streamlit as st
import easyocr
import fitz
import numpy as np
from PIL import Image
st.title("Document & Image")
reader=easyocr.Reader(["en"])
def ocr(img):
   return "\n".join(reader.readtext(np.array(img),detail=0))
def pdf_ocr(file):
   doc=fitz.open(stream=file.read(),filetype="pdf")
   text=""
   for page in doc:
       pix=page.get_pixmap(dpi=200)
       img=Image.frombytes(("RGB"),(pix.width,pix.height),pix.samples)
       text=text+ocr(img)+"\n"
   return text 
def image_ocr(file):
   img=Image.open(file)
   st.image(img,use_container_width=True)
   return ocr(img)
file=st.file_uploader("Upload PDF/Image",["pdf","jpg","jpeg"])
if file:
   if file.name.endswith("pdf"):
      text=pdf_ocr(file)
   else:
      text=image_ocr(file)
   st.text_area("",text,height=300)