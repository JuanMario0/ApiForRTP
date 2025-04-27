# from sqlalchemy import create_engine
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker
# from core.config import Settings



# #SQLALCHEMY_DATABASE_URL = ""
# SQLALCHEMY_DATABASE_URL = Settings.DATABASE_URL
# engine = create_engine(SQLALCHEMY_DATABASE_URL)
# SessionLocal = sessionmaker(bind = engine, autocommit=False, autoflush=False)
# Base = declarative_base()


# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()


from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Leer DATABASE_URL desde las variables de entorno
# Valor por defecto: base de datos local para pruebas
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:2000@localhost:5432/ApiForRTP")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()