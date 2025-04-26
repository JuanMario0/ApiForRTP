from fastapi import APIRouter, Depends, status, Request, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.schemas import Login, Token, UserCreate  # Añadimos UserCreate
from app.routers.repository import auth
from app.db.models import User  # Importamos el modelo User
from passlib.context import CryptContext  # Para cifrar contraseñas

router = APIRouter(
    prefix="/login",
    tags=["Login"]
)

# Configuración para cifrar contraseñas
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Función para cifrar contraseñas
def hash_password(password: str):
    return pwd_context.hash(password)

# Endpoint para registrar usuarios
@router.post("/register/", status_code=status.HTTP_201_CREATED)
async def register(
    request: Request,
    db: Session = Depends(get_db)
):
    content_type = request.headers.get("Content-Type")

    # Si es JSON
    if content_type == "application/json":
        try:
            user_data = await request.json()
            user = UserCreate(**user_data)  # Valida y convierte el JSON al esquema UserCreate
        except ValueError as e:
            raise HTTPException(status_code=422, detail=f"Invalid JSON format: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing JSON: {str(e)}")
    
    # Si es form-data (OAuth2)
    elif content_type == "application/x-www-form-urlencoded":
        try:
            form_data = await request.form()
            if "email" not in form_data or "password" not in form_data:
                raise HTTPException(
                    status_code=422,
                    detail="Missing required fields: email and password are required"
                )
            user = UserCreate(email=form_data["email"], password=form_data["password"])
        except ValueError as e:
            raise HTTPException(status_code=422, detail=f"Invalid form data: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing form data: {str(e)}")
    
    # Si no es ninguno de los dos
    else:
        raise HTTPException(status_code=415, detail="Unsupported Media Type: Expected 'application/json' or 'application/x-www-form-urlencoded'")

    # Verificar si el usuario ya existe
    existing_user = db.query(User).filter(User.email == user.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="El email ya está registrado")

    # Crear nuevo usuario con contraseña cifrada
    hashed_password = hash_password(user.password)
    new_user = User(email=user.email, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {"message": "Usuario registrado exitosamente"}

# Endpoint para login (sin cambios)
@router.post("/", status_code=status.HTTP_200_OK, response_model=Token)
async def login(
    request: Request,
    db: Session = Depends(get_db)
):
    content_type = request.headers.get("Content-Type")

    if content_type == "application/json":
        try:
            user_data = await request.json()
            user = Login(**user_data)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=f"Invalid JSON format: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing JSON: {str(e)}")
    
    elif content_type == "application/x-www-form-urlencoded":
        try:
            form_data = await request.form()
            if "username" not in form_data or "password" not in form_data:
                raise HTTPException(
                    status_code=422,
                    detail="Missing required fields: username and password are required"
                )
            user = Login(email=form_data["username"], password=form_data["password"])
        except ValueError as e:
            raise HTTPException(status_code=422, detail=f"Invalid form data: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing form data: {str(e)}")
    
    else:
        raise HTTPException(status_code=415, detail="Unsupported Media Type: Expected 'application/json' or 'application/x-www-form-urlencoded'")

    auth_token = auth.auth_user(user, db)
    if not auth_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return auth_token