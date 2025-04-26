
from app.db.database import Base
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Float, Date
from datetime import datetime
from sqlalchemy.schema import ForeignKey
from sqlalchemy.orm import relationship
from app.db.database import Base


class User(Base):
    __tablename__ = "usuario"
    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String, unique=True, index=True)  # Cambiar username por email
    password = Column(String)  # Cambiar password por hashed_password



class Stop(Base):
    __tablename__ = "stops"
    id = Column(Integer, primary_key=True, autoincrement=True)
    stop_id = Column(String, index=True)  # Eliminamos unique=True
    trip_id = Column(String, index=True)  # Nueva columna para identificar el viaje
    stop_name = Column(String)
    stop_lat = Column(Float)
    stop_lon = Column(Float)
    arrival_time = Column(DateTime)  # Cambiamos a DateTime para manejar fechas completas
    headway_secs = Column(Float)
    wait_time = Column(Float, nullable=True)
    delay = Column(Float, nullable=True)
    simulated_delay = Column(Float, nullable=True)
    cluster = Column(Integer, nullable=True)


class Comment(Base):
    __tablename__ = "comments"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_email = Column(String, nullable=False)
    comentario = Column(String)
    source = Column(String)
    fecha = Column(Date)
    tiene_groseria = Column(Boolean)
    comentario_censurado = Column(String)
    etiqueta = Column(String)
    etiqueta_predicha = Column(String)
    relevancia = Column(Float)