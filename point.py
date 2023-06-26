import processor
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles

app = FastAPI()
templates =  Jinja2Templates(directory="templates")
app.mount("/resources", StaticFiles(directory="resources"), name="resources")

#TYPES: raw, dates, graded, spam, ham, sentiment
df = processor.display('raw',processor.create_df('default'))

@app.get("/{type}", response_class=HTMLResponse)
async def display(request:Request, type: str):
    df = processor.display(type,processor.create_df('default'))
    if type == 'images':
        return templates.TemplateResponse("images.html", {"request": request})
    return templates.TemplateResponse("index.html", {"request": request, "df":df})

"""
@app.get("/{type}", response_class=HTMLResponse)
async def display(request:Request, type: str):
    return templates.TemplateResponse("index.html", {"request": request})
"""
