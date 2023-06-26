import PyPDF2
import pandas as pd
import os

#DATA INTAKE 
def pdf_to_string(file_path):
    pdf_text = ""
    with open(file_path, 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        num_pages = len(reader.pages)
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            pdf_text += page.extract_text()
    return pdf_text

def get_path(input): 
    pdf_path = os.path.join(os.getcwd(), 'resources', 'zPDFs', input)
    return pdf_path

def return_df(input):
    if input == 'default':
        doc1 = pdf_to_string(get_path('pdf1.pdf'))
        doc2 = pdf_to_string(get_path('pdf2.pdf'))
        doc3 = pdf_to_string(get_path('pdf3.pdf'))
        doc4 = pdf_to_string(get_path('pdf4.pdf'))
        df = pd.DataFrame({'docs':[doc1,doc2,doc3,doc4]})
        return df
        return doc1
    else:
        return 'Not supported yet.'

    
    

