from sqlalchemy.orm import Session
from app.db import models
from fastapi import HTTPException, status
from app.hashing import Hash
from app.token import create_access_token

def auth_user(user: "Login", db: Session):
    # No necesitamos convertir a dict, ya que user es un objeto Login
    # Filtrar por email en lugar de username
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Usuario no encontrado con el email {user.email}"
        )

    # Verificar la contraseña (ajustar el nombre del método según tu implementación)
    # Si el método se llama Hash.verify (como en ejemplos anteriores), usa eso
    # Si el método se llama Hash.verify_password, déjalo como está
    print("Contraseña ingresada:", user.password)
    print("Contraseña hasheada en DB:", db_user.password)
    if not Hash.verify_password(user.password, db_user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Contraseña incorrecta"
        )

    # Generar token usando email
    access_token = create_access_token(data={"sub": db_user.email})
    return {"access_token": access_token, "token_type": "bearer"}