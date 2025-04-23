from typing import Optional, Union
from datetime import datetime
from pydantic import BaseModel

#USER MODEL 
class User(BaseModel): #ESQUEMA
    username:str
    password:str
    creacion:datetime = datetime.now()


class UpdateUser(BaseModel): #ESQUEMA
    username:str = None
    password:str = None


class UserID(BaseModel):
    id:int


class ShowUser(BaseModel):
    username:str
    class Config():
        orm_mode = True


class Login(BaseModel):
    username:str
    password:str


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None