from fastapi import FastAPI
import uvicorn 
from app.routers import user, auth
from app.db.database import Base, engine 
from app.routers import user
from .app.routers import auth


#def create_tables():
#    Base.metadata.create_all(bind=engine)
#create_tables()


app = FastAPI()
app.include_router(user.router)
app.include_router(auth.router)


if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)
