
from app.db.database import Base
from sqlalchemy import Column, Integer, String, Boolean, DateTime
from datetime import datetime
from sqlalchemy.schema import ForeignKey
from sqlalchemy.orm import relationship

class User(Base):
    __tablename__ = "usuario"
    id = Column(Integer, primary_key = True, autoincrement = True)
    username = Column(String, unique=True)
    password = Column(String)
  

