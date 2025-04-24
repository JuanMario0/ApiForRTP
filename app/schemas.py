from pydantic import BaseModel
from typing import Optional, List
from datetime import date, datetime

# Modelos existentes
class UserBase(BaseModel):
    email: str


class UserID(BaseModel):
    id: int

class UserCreate(UserBase):
    email: str
    password: str

    
class User(UserBase):
    id: int 
    hashed_password: str

    class Config:
        orm_mode = True

class ShowUser(BaseModel):
    email: str

    class Config:
        orm_mode = True

class UpdateUser(BaseModel):
    email: Optional[str] = None
    password: Optional[str] = None


class Login(BaseModel):
    email:str
    password:str

class StopDetails(BaseModel):
    stop_id: str
    stop_name: str
    stop_lat: float
    stop_lon: float
    simulated_delay: float

class ClusterDetails(BaseModel):
    cluster_id: int
    stops: List[StopDetails]

class Token(BaseModel):
    access_token: str
    token_type: str

# Nuevos modelos
class Stop(BaseModel):
    id: int
    stop_id: str
    stop_name: str
    stop_lat: float
    stop_lon: float
    arrival_time: Optional[datetime] = None
    headway_secs: Optional[float] = None
    wait_time: Optional[float] = None
    delay: Optional[float] = None
    simulated_delay: Optional[float] = None
    cluster: Optional[int] = None

    class Config:
        orm_mode = True

class Comment(BaseModel):
    id: int
    comentario: str
    source: str
    fecha: date
    tiene_groseria: bool
    comentario_censurado: str
    etiqueta: str
    etiqueta_predicha: str
    relevancia: float

    class Config:
        orm_mode = True

class CommentCreate(BaseModel):
    comentario: str
    source: str
    fecha: date
    tiene_groseria: bool
    comentario_censurado: str
    etiqueta: str
    etiqueta_predicha: str
    relevancia: float


class CommentRequest(BaseModel):
    comment: str