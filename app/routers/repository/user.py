from sqlalchemy.orm import Session
from app.db import models
from fastapi import HTTPException, status
from app.hashing import Hash
from app.schemas import UserCreate, UpdateUser
from app.oauth import get_current_user
from app.db.models import User


def CrearUsuario(usuario: UserCreate, db: Session):
    usuario = usuario.dict()
    try:
        nuevo_usuario = models.User(
            email=usuario["email"],
            password=Hash.hash_password(usuario["password"])  # Hashear la contraseña
        )
        db.add(nuevo_usuario)
        db.commit()
        db.refresh(nuevo_usuario)
        return nuevo_usuario
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Error creando usuario: {e}"
        )


def ObtenerUsuario(userId: int, db: Session):
    usuario = db.query(User).filter(User.id == userId).first()
    if not usuario:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    return usuario


def EliminarUsuario(userId, db:Session):
      usuario = db.query(models.User).filter(models.User.id == userId)
      if not usuario.first():
            raise HTTPException(
                 status_code=status.HTTP_404_NOT_FOUND,
                 detail= f"No existe el usuario con el id: {userId} por lo tanto no se puede elimianr"
            )
      usuario.delete(synchronize_session = False)
      db.commit()

      return {"Respuesta":"Usuario borrado correctamente"}


def ObtenerUsuarios(db:Session):
      data = db.query(models.User).all()
      return data 


def ActualizarUsuario(userId, updateUser, db:Session):
    usuario = db.query(models.User).filter(models.User.id == userId)
    if not usuario.first():
        raise HTTPException(
             status_code=status.HTTP_404_NOT_FOUND,
             detail = f"No existe el usario para actualizar con el id: {userId}, asi que no se puede hacer la tarea"
        )
    
    usuario.update(updateUser.dict(exclude_unset=True))
    db.commit()
        
    return {"Respuesta":"Usuario actualizado"}
