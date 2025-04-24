from fastapi import APIRouter, Depends, status
from app.schemas import User, UserID, ShowUser, UpdateUser, UserCreate
from app.db.database import get_db
from sqlalchemy.orm import Session
from typing import List
from app.routers.repository import user
from app.routers.repository.user import CrearUsuario
from app.routers.repository.user import ObtenerUsuario
from app.routers.repository.user import EliminarUsuario
from app.routers.repository.user import ActualizarUsuario
from app.oauth import get_current_user




router = APIRouter(
    prefix="/user",
    tags=["Users"]
)


@router.get('/', response_model=List[ShowUser], status_code=status.HTTP_200_OK)
def ObtenerUusuarios(db:Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    data = user.ObtenerUsuario(db)
    return data


@router.post("/", response_model=ShowUser)
def create_user(usuario: UserCreate, db: Session = Depends(get_db)):
    return CrearUsuario(usuario, db)


@router.get('/{userId}', response_model=ShowUser,  status_code=status.HTTP_200_OK) 
def ObtenerUsuario(userId:int, db:Session = Depends(get_db), current_user: User = Depends(get_current_user)):
        usuario = user.ObtenerUsuario(userId, db)

        return usuario


@router.delete('/',  status_code=status.HTTP_200_OK)
def ElimarUsuario(userId:int, db:Session = Depends(get_db), current_user: User = Depends(get_current_user)):
   res = user.EliminarUsuario(userId, db)
   return res



@router.patch('/{userId}', status_code=status.HTTP_200_OK)
def ActualizarUsuario(userId:int, updateUser: UpdateUser, db:Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    res = user.ActualizarUsuario(userId, updateUser,db)
    return res



