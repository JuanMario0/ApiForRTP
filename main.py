from fastapi import FastAPI
import uvicorn 
from app.routers import user, auth, data_processing
from app.db.database import Base, engine, SessionLocal
from fastapi.middleware.cors import CORSMiddleware
from app.routers import data_processing
from app.db.models import Stop
import pandas as pd



#def create_tables():
#    Base.metadata.create_all(bind=engine)
#create_tables()


app = FastAPI()

# Crear las tablas en la base de datos
Base.metadata.create_all(bind=engine)

# Función para poblar la tabla stops al inicio
def populate_stops():
    db = SessionLocal()
    try:
        # Verificar si la tabla stops ya tiene datos
        if db.query(Stop).count() == 0:
            print("Poblando la tabla stops desde full_data.csv...")
            df = pd.read_csv("full_data.csv")
            print(f"Cargados {len(df)} registros desde full_data.csv")

            # Insertar datos en la tabla stops
            for _, row in df.iterrows():
                stop = Stop(
                    stop_id=row['stop_id'],
                    trip_id=row.get('trip_id', ''),
                    stop_name=row['stop_name'],
                    stop_lat=row['stop_lat'],
                    stop_lon=row['stop_lon'],
                    arrival_time=row['arrival_time'],
                    headway_secs=row['headway_secs'],
                    wait_time=0,
                    delay=0,
                    simulated_delay=0,
                    cluster=-1
                )
                db.add(stop)
            db.commit()
            print("Datos insertados exitosamente en la tabla stops")
        else:
            print("La tabla stops ya tiene datos, no se realizará la carga.")
    except Exception as e:
        print(f"Error al insertar datos: {str(e)}")
        db.rollback()
    finally:
        db.close()

# Ejecutar la función al iniciar la aplicación
@app.on_event("startup")
async def startup_event():
    populate_stops()

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
