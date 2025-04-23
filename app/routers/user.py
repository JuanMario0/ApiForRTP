from fastapi import APIRouter, Depends, status
from app.schemas import User, UserID, ShowUser, UpdateUser
from app.db.database import get_db
from sqlalchemy.orm import Session
from typing import List
from app.routers.repository import user



router = APIRouter(
    prefix="/user",
    tags=["Users"]
)


@router.get('/', response_model=List[ShowUser], status_code=status.HTTP_200_OK)
def ObtenerUusuarios(db:Session = Depends(get_db)):
    data = user.ObtenerUsuario(db)
    return data


@router.post('/', status_code = status.HTTP_201_CREATED)
def CrearUsuario(usuario:User, db:Session = Depends(get_db)):
    user.CrearUsuario(usuario, db)

    return {"Usario":"Usuario creado Correctamente!!"}


@router.get('/{userId}', response_model=ShowUser,  status_code=status.HTTP_200_OK) 
def ObtenerUsuario(userId:int, db:Session = Depends(get_db)):
        usuario = user.ObtenerUsuario(userId, db)

        return usuario


@router.delete('/')
def ElimarUsuario(userId:int, db:Session = Depends(get_db),  status_code=status.HTTP_200_OK):
   res = user.EliminarUsuario(userId, db)
   return res



@router.patch('/{userId}')
def ActualizarUsuario(userId:int, updateUser: UpdateUser, db:Session = Depends(get_db),  status_code=status.HTTP_200_OK):
    res = user.ActualizarUsuario(userId, updateUser,db)
    return res

    

