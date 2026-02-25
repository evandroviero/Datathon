from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

from api.routers import router as tickers_router
from api.database import engine
from api.models import Base

from src import config

app = FastAPI(
    title="Datathon",
    description="API for the Datathon",
    version="1.0.0",
)

Base.metadata.create_all(bind=engine)
TEMPLATES_DIR = config.BASE_DIR / "template"             
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

app.include_router(tickers_router)