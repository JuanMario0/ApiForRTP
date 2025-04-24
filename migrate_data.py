from sqlalchemy.orm import Session
from app.db.database import SessionLocal
from app.db.models import Stop, Comment
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sqlalchemy.dialects.postgresql import insert
from datetime import datetime

# Crear una sesión
db = SessionLocal()

try:
    # Migrar datos de full_data.csv
    full_data = pd.read_csv("full_data.csv")
    for _, row in full_data.iterrows():
        # Convertir arrival_time
        try:
            arrival_time = pd.to_datetime(row['arrival_time'], format='%H:%M:%S', errors='coerce')
            if pd.isna(arrival_time):
                print(f"Fecha de llegada inválida en el registro stop_id {row['stop_id']}: {row['arrival_time']}, se omitirá.")
                continue
        except Exception as e:
            print(f"Error al procesar arrival_time en el registro stop_id {row['stop_id']}: {row['arrival_time']}. Error: {str(e)}. Se omitirá.")
            continue

        # Insertar el registro
        stop = Stop(
            stop_id=row['stop_id'],
            trip_id=row['trip_id'],
            stop_name=row['stop_name'],
            stop_lat=row['stop_lat'],
            stop_lon=row['stop_lon'],
            arrival_time=arrival_time,
            headway_secs=row['headway_secs']
        )
        db.add(stop)

    db.commit()
    print("Datos de full_data.csv migrados exitosamente.")

    # Procesar datos para calcular wait_time, delay, simulated_delay y cluster
    stops = db.query(Stop).all()
    if stops:
        full_data = pd.DataFrame([{
            "stop_id": stop.stop_id,
            "trip_id": stop.trip_id,
            "stop_name": stop.stop_name,
            "stop_lat": stop.stop_lat,
            "stop_lon": stop.stop_lon,
            "arrival_time": stop.arrival_time,
            "headway_secs": stop.headway_secs,
            "wait_time": stop.wait_time,
            "delay": stop.delay,
            "simulated_delay": stop.simulated_delay,
            "cluster": stop.cluster
        } for stop in stops])

        # Depuración: Verificar cuántos registros hay por stop_id
        print("Conteo de registros por stop_id:")
        print(full_data.groupby('stop_id').size().describe())

        # Depuración: Verificar cuántos registros hay por trip_id
        print("Conteo de registros por trip_id:")
        print(full_data.groupby('trip_id').size().describe())

        # Convertir arrival_time a datetime
        print("Valores de arrival_time antes del cálculo:")
        print(full_data['arrival_time'].head(10))

        full_data['arrival_time'] = pd.to_datetime(full_data['arrival_time'], errors='coerce')

        # Verificar si hay NaT en arrival_time
        if full_data['arrival_time'].isna().any():
            print("Advertencia: Algunos valores de arrival_time son NaT:")
            print(full_data[full_data['arrival_time'].isna()])
            full_data['arrival_time'] = full_data['arrival_time'].fillna(pd.to_datetime('00:00:00', format='%H:%M:%S'))

        # Calcular wait_time como el tiempo entre llegadas consecutivas a un mismo stop_id
        full_data = full_data.sort_values(by=['stop_id', 'arrival_time'])
        full_data['wait_time'] = full_data.groupby('stop_id')['arrival_time'].diff().dt.total_seconds() / 60
        print("Valores de wait_time después del cálculo (antes de fillna):")
        print(full_data['wait_time'].head(10))
        # Reemplazar NaN con headway_secs / 60 (convertido a minutos)
        full_data['wait_time'] = full_data['wait_time'].fillna(full_data['headway_secs'] / 60)
        # Depuración: Mostrar registros con wait_time = 0
        print("Registros con wait_time = 0 (después de fillna):")
        print(full_data[full_data['wait_time'] == 0][['stop_id', 'trip_id', 'arrival_time']].head(10))

        full_data['headway_mins'] = full_data['headway_secs'] / 60
        full_data['delay'] = full_data['wait_time'] - full_data['headway_mins']
        full_data['delay'] = full_data['delay'].fillna(0)
        np.random.seed(42)
        simulated_delays = np.random.poisson(lam=5, size=len(full_data))
        full_data['simulated_delay'] = np.where(full_data['delay'] <= 0, simulated_delays, full_data['delay'])

        # Clustering
        X = full_data[['stop_lat', 'stop_lon', 'simulated_delay']].dropna()
        kmeans = KMeans(n_clusters=5, random_state=42).fit(X)
        full_data['cluster'] = kmeans.predict(X)

        # Actualizar la base de datos con los resultados
        for idx, row in full_data.iterrows():
            stop = db.query(Stop).filter(Stop.stop_id == row['stop_id'], Stop.trip_id == row['trip_id']).first()
            if stop:
                stop.wait_time = row['wait_time']
                stop.delay = row['delay']
                stop.simulated_delay = row['simulated_delay']
                stop.cluster = row['cluster']
        db.commit()
        print("Datos de stops procesados exitosamente (wait_time, delay, simulated_delay, cluster).")

    # Migrar datos de comments_processed.csv
    try:
        comments_df = pd.read_csv("comments_processed.csv", encoding='utf-8')
    except UnicodeDecodeError:
        print("Error de codificación UTF-8, intentando con 'latin1'...")
        comments_df = pd.read_csv("comments_processed.csv", encoding='latin1')
    except FileNotFoundError:
        print("Archivo comments_processed.csv no encontrado, inicializando DataFrame vacío.")
        comments_df = pd.DataFrame(columns=[
            "comentarios", "source", "fecha", "tiene_groseria", 
            "comentarios_censurados", "etiqueta", "etiqueta_predicha", "relevancia"
        ])

    for idx, row in comments_df.iterrows():
        print(f"Procesando comentario {idx + 1}, fecha: {row['fecha']}")
        try:
            fecha = pd.to_datetime(row['fecha'], errors='coerce', format='%Y-%m-%d')
            if pd.isna(fecha):
                fecha = pd.to_datetime(row['fecha'], errors='coerce', format='%d/%m/%Y')
            if pd.isna(fecha):
                fecha = pd.to_datetime(row['fecha'], errors='coerce', format='%d-%m-%Y')
            if pd.isna(fecha):
                print(f"Fecha inválida en el registro {idx + 1}: {row['fecha']}. Formatos esperados: 'YYYY-MM-DD', 'DD/MM/YYYY', o 'DD-MM-YYYY'. Se omitirá.")
                continue
        except Exception as e:
            print(f"Error al procesar la fecha en el registro {idx + 1}: {row['fecha']}. Error: {str(e)}. Se omitirá.")
            continue

        comment = Comment(
            comentario=row['comentarios'],
            source=row['source'],
            fecha=fecha.date(),
            tiene_groseria=row['tiene_groseria'],
            comentario_censurado=row['comentarios_censurados'],
            etiqueta=row['etiqueta'],
            etiqueta_predicha=row['etiqueta_predicha'],
            relevancia=row['relevancia']
        )
        db.add(comment)

    db.commit()
    print("Datos de comments_processed.csv migrados exitosamente.")

except Exception as e:
    print(f"Error durante la migración: {str(e)}")
    db.rollback()

finally:
    db.close()
    print("Sesión de base de datos cerrada.")