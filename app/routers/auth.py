from fastapi import APIRouter, Depends, status, Request, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.schemas import Login, Token
from app.routers.repository import auth

router = APIRouter(
    prefix="/login",
    tags=["Login"]
)

@router.post("/", status_code=status.HTTP_200_OK, response_model=Token)
async def login(
    request: Request,
    db: Session = Depends(get_db)
):
    content_type = request.headers.get("Content-Type")

    # Si es JSON
    if content_type == "application/json":
        try:
            user_data = await request.json()
            user = Login(**user_data)  # Valida y convierte el JSON al esquema Login
        except ValueError as e:
            raise HTTPException(status_code=422, detail=f"Invalid JSON format: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing JSON: {str(e)}")
    
    # Si es form-data (OAuth2)
    elif content_type == "application/x-www-form-urlencoded":
        try:
            form_data = await request.form()
            # Validar que los campos requeridos estén presentes
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
    
    # Si no es ninguno de los dos
    else:
        raise HTTPException(status_code=415, detail="Unsupported Media Type: Expected 'application/json' or 'application/x-www-form-urlencoded'")

    # Autenticar al usuario
    auth_token = auth.auth_user(user, db)
    if not auth_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return auth_token