from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import Stop, Comment
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import folium
from folium.plugins import HeatMap
import joblib
import os
import re
from datetime import datetime
from app.oauth import get_current_user  # Aseguramos que se use el sistema de autenticación existente
from app.schemas import User, StopDetails, ClusterDetails, Token, Comment as CommentSchema, CommentCreate, CommentRequest
from typing import List, Dict

router = APIRouter(
    prefix="/data",
    tags=["Data Processing"]
)

# Cargar el modelo y vectorizador de PLN
try:
    comment_classifier = joblib.load("app\\models\\comment_classifier_final.pkl")
    vectorizer = joblib.load("app\\models\\vectorizer_final.pkl")
except FileNotFoundError:
    print("Error: No se encontraron los archivos comment_classifier.pkl o vectorizer.pkl.")
    comment_classifier = None
    vectorizer = None

# Lista de groserías y funciones para manejarlas
groserias = [
    "puto", "puta", "chinga", "cabrón", "cabron", "idiota", "estúpido", "estupido", 
    "mierda", "joder", "pendejo", "culero", "verga"
]
patron_groserias = "|".join([re.escape(groseria) for groseria in groserias])
patron_groserias = patron_groserias.replace("0", "[0o]").replace("e", "[e3]").replace("i", "[i1]")

def contiene_groseria(comentario):
    comentario = comentario.lower()
    return bool(re.search(patron_groserias, comentario))

def censurar_groseria(comentario):
    comentario_lower = comentario.lower()
    for groseria in groserias:
        censura = "*" * len(groseria)
        patron = groseria.replace("e", "[e3]").replace("i", "[i1]").replace("o", "[0o]")
        comentario = re.sub(patron, censura, comentario, flags=re.IGNORECASE)
    return comentario

# Variable global para almacenar los resultados del procesamiento
_processed_data = None

def load_and_process_data(db: Session):
    global _processed_data
    if _processed_data is not None:
        return _processed_data  # Retornar datos procesados si ya existen

    # Cargar datos de paradas desde la base de datos
    stops = db.query(Stop).all()
    if not stops:
        raise HTTPException(status_code=404, detail="No se encontraron paradas en la base de datos")

    # Convertir a DataFrame
    full_data = pd.DataFrame([{
        "stop_id": stop.stop_id,
        "trip_id": stop.trip_id,  # Incluimos trip_id para mantener consistencia
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

    # Procesar datos
    full_data['arrival_time'] = pd.to_datetime(full_data['arrival_time'], errors='coerce')
    if full_data['arrival_time'].isna().any():
        print("Advertencia: Algunos valores de arrival_time son NaT:")
        print(full_data[full_data['arrival_time'].isna()])
        full_data['arrival_time'] = full_data['arrival_time'].fillna(pd.to_datetime('00:00:00', format='%H:%M:%S'))

    full_data = full_data.sort_values(by=['stop_id', 'arrival_time'])
    full_data['wait_time'] = full_data.groupby('stop_id')['arrival_time'].diff().dt.total_seconds() / 60
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

    # Actualizar la base de datos con los resultados (en lotes para mejor rendimiento)
    batch_size = 10000
    for start in range(0, len(full_data), batch_size):
        batch = full_data.iloc[start:start + batch_size]
        for idx, row in batch.iterrows():
            stop = db.query(Stop).filter(Stop.stop_id == row['stop_id'], Stop.trip_id == row['trip_id']).first()
            if stop:
                stop.wait_time = row['wait_time']
                stop.delay = row['delay']
                stop.simulated_delay = row['simulated_delay']
                stop.cluster = row['cluster']
        db.commit()

    # Métricas
    silhouette_avg = silhouette_score(X, kmeans.labels_)
    cluster_delays = full_data.groupby('cluster')['simulated_delay'].mean()
    delays_by_stop = full_data.groupby('stop_id').agg({
        'stop_lat': 'first',
        'stop_lon': 'first',
        'stop_name': 'first',
        'simulated_delay': 'mean'
    }).reset_index()
    top_delays = delays_by_stop.sort_values('simulated_delay', ascending=False).head(10)

    # Guardar modelo de clustering
    joblib.dump(kmeans, 'kmeans_model.pkl')

    # Generar mapa de calor
    map_center = [delays_by_stop['stop_lat'].mean(), delays_by_stop['stop_lon'].mean()]
    m = folium.Map(location=map_center, zoom_start=12)
    heat_data = [[row['stop_lat'], row['stop_lon'], row['simulated_delay']] 
                 for _, row in delays_by_stop.iterrows()]
    HeatMap(heat_data, radius=15).add_to(m)
    for _, row in top_delays.iterrows():
        folium.Marker(
            location=[row['stop_lat'], row['stop_lon']],
            popup=f"{row['stop_name']}: {row['simulated_delay']:.1f} min",
            icon=folium.Icon(color='red', icon='bus', prefix='fa')
        ).add_to(m)
    if not os.path.exists("static"):
        os.makedirs("static")
    m.save("static/heatmap.html")

    # Almacenar los resultados en la variable global
    _processed_data = (full_data, silhouette_avg, cluster_delays, delays_by_stop, top_delays, map_center)
    return _processed_data

def load_comments(db: Session):
    comments = db.query(Comment).all()
    if not comments:
        return pd.DataFrame(columns=[
            "comentarios", "source", "fecha", "tiene_groseria", 
            "comentarios_censurados", "etiqueta", "etiqueta_predicha", "relevancia"
        ])
    return pd.DataFrame([{
        "comentarios": comment.comentario,
        "source": comment.source,
        "fecha": comment.fecha,
        "tiene_groseria": comment.tiene_groseria,
        "comentarios_censurados": comment.comentario_censurado,
        "etiqueta": comment.etiqueta,
        "etiqueta_predicha": comment.etiqueta_predicha,
        "relevancia": comment.relevancia
    } for comment in comments])

# Rutas de datos protegidas
@router.get("/stops/{cluster_id}", response_model=List[StopDetails])
async def get_stops(cluster_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    full_data, _, _, _, _, _ = load_and_process_data(db)
    stops_in_cluster = full_data[full_data['cluster'] == cluster_id][['stop_id', 'stop_name', 'stop_lat', 'stop_lon', 'simulated_delay']]
    if stops_in_cluster.empty:
        raise HTTPException(status_code=404, detail="No stops found for this cluster")
    return stops_in_cluster.to_dict(orient='records')

@router.get("/stop_details/{stop_id}", response_model=StopDetails)
async def get_stop_details(stop_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    full_data, _, _, _, _, _ = load_and_process_data(db)
    stop_details = full_data[full_data['stop_id'] == stop_id][['stop_id', 'stop_name', 'stop_lat', 'stop_lon', 'simulated_delay']]
    if stop_details.empty:
        raise HTTPException(status_code=404, detail="Stop not found")
    return stop_details.iloc[0].to_dict()

@router.get("/metrics")
async def get_metrics(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    full_data, silhouette_avg, cluster_delays, _, top_delays, map_center = load_and_process_data(db)
    comments_df = load_comments(db)

    # Preparar datos para las cards
    positivos = comments_df[comments_df["etiqueta_predicha"] == "positivo_sugerencia"]
    top_positivos = positivos.sort_values(by="relevancia", ascending=False).head(5)[["comentarios_censurados"]].to_dict(orient="records")
    top_positivos = [item["comentarios_censurados"] for item in top_positivos]

    negativos = comments_df[comments_df["etiqueta_predicha"] == "negativo"]
    top_negativos = negativos.sort_values(by="relevancia", ascending=False).head(5)[["comentarios_censurados"]].to_dict(orient="records")
    top_negativos = [item["comentarios_censurados"] for item in top_negativos]

    neutros = comments_df[comments_df["etiqueta_predicha"] == "neutral"]
    top_neutros = neutros.sort_values(by="relevancia", ascending=False).head(5)[["comentarios_censurados"]].to_dict(orient="records")
    top_neutros = [item["comentarios_censurados"] for item in top_neutros]

    cluster_ids = []
    for x in full_data['cluster'].unique():
        try:
            x_int = int(float(x))
            if not pd.isna(x_int):
                cluster_ids.append(x_int)
        except (ValueError, TypeError):
            continue
    cluster_ids.sort()

    return {
        "silhouette_avg": silhouette_avg,
        "cluster_delays": cluster_delays.to_dict(),
        "top_delays": top_delays.to_dict(orient='records'),
        "map_center": map_center,
        "cluster_ids": cluster_ids,
        "num_clusters": len(set(cluster_ids)),
        "top_positivos": top_positivos,
        "top_negativos": top_negativos,
        "top_neutros": top_neutros
    }

@router.get("/heatmap")
async def get_heatmap(current_user: User = Depends(get_current_user)):
    return FileResponse("static/heatmap.html")

@router.post("/classify_comment")
async def classify_comment(
    data: CommentRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    nuevo_comentario = data.comment
    if not nuevo_comentario:
        raise HTTPException(status_code=400, detail="No se proporcionó un comentario")

    tiene_groseria = contiene_groseria(nuevo_comentario)
    comentario_censurado = censurar_groseria(nuevo_comentario)

    vector = vectorizer.transform([comentario_censurado])
    prediccion = comment_classifier.predict(vector)[0]
    relevancia = float(vector.sum())

    fecha_actual = datetime(2025, 4, 18).date()
    new_comment = Comment(
        comentario=nuevo_comentario,
        source="Formulario Web",
        fecha=fecha_actual,
        tiene_groseria=tiene_groseria,
        comentario_censurado=comentario_censurado,
        etiqueta="negativo" if tiene_groseria else prediccion,
        etiqueta_predicha=prediccion,
        relevancia=relevancia
    )
    db.add(new_comment)
    db.commit()

    return {
        "comment": nuevo_comentario,
        "censored_comment": comentario_censurado,
        "category": prediccion,
        "relevance": relevancia
    }

@router.get("/download_comments_csv")
async def download_comments_csv(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    comments_df = load_comments(db)
    csv_path = "comments_processed.csv"
    comments_df.to_csv(csv_path, index=False, encoding='utf-8')
    return FileResponse(
        csv_path,
        filename="comments_processed.csv",
        media_type="text/csv"
    )