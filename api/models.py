from sqlalchemy import Column, Integer, Float, String, Boolean, DateTime
from datetime import datetime
from api.database import Base

class Student(Base):
    __tablename__ = "student"

    id = Column(Integer, primary_key=True, index=True)
    idade = Column(Integer, nullable=False)
    inde = Column(Float, nullable=False)
    ian = Column(Float, nullable=False)
    ida = Column(Float, nullable=False)
    ieg = Column(Float, nullable=False)
    iaa = Column(Float, nullable=False)
    ips = Column(Float, nullable=False)
    ipv = Column(Float, nullable=False)
    matem = Column(Float, nullable=False)
    portug = Column(Float, nullable=False)
    no_av = Column(Integer, nullable=False)
    genero = Column(String, nullable=False)
    instituicao_padronizada = Column(String, nullable=False)
    # fase = Column(String, nullable=False)
    rec_psicologia_padronizada = Column(String, nullable=False)
    classe_defas = Column(String, nullable=False)
    data_predicao = Column(DateTime, default=datetime.utcnow)