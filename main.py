from fastapi import FastAPI
import uvicorn 
from app.routers import user, auth, data_processing
from app.db.database import Base, engine 
from app.routers import user
from app.routers import auth
from fastapi.middleware.cors import CORSMiddleware


#def create_tables():
#    Base.metadata.create_all(bind=engine)
#create_tables()


app = FastAPI()


# Configurar CORS sirve para permitir los puertos. 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(user.router)
app.include_router(auth.router)
app.include_router(data_processing.router)


if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)
