import os
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
import folium
from folium.plugins import HeatMap
import joblib
import re
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import Stop, Comment
from app.oauth import get_current_user
from app.schemas import User, StopDetails, ClusterDetails, CommentRequest, CommentCreate
from typing import List, Dict
import logging
import time

router = APIRouter(
    prefix="/data",
    tags=["Data Processing"]
)

# Configurar logging
logging.basicConfig(level=logging.INFO)

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
    "mierda", "joder", "pendejo", "culero", "verga", "hijo de puta", "hijoeputa", "maña"
]
patron_groserias = "|".join([re.escape(groseria) for groseria in groserias])
patron_groserias = patron_groserias.replace("0", "[0o]").replace("e", "[e3]").replace("i", "[i1]")

def contiene_groseria(comentario):
    comentario = comentario.lower()
    return bool(re.search(patron_groserias, comentario))

def censurar_groseria(comentario):
    comentario = comentario.lower()
    for groseria in groserias:
        censura = "*" * len(groseria)
        patron = groseria.replace("e", "[e3]").replace("i", "[i1]").replace("o", "[0o]")
        comentario = re.sub(patron, censura, comentario, flags=re.IGNORECASE)
    return comentario

# Variables globales para cachear resultados
_processed_data = None
_cached_comments = None
_silhouette_avg = None

# Cargar silhouette_avg desde un archivo (si existe)
SILHOUETTE_FILE = "silhouette_avg.txt"
if os.path.exists(SILHOUETTE_FILE):
    try:
        with open(SILHOUETTE_FILE, "r") as f:
            _silhouette_avg = float(f.read().strip())
        print(f"Silhouette_avg cargado desde archivo: {_silhouette_avg}")
    except Exception as e:
        print(f"Error al cargar silhouette_avg desde archivo: {str(e)}")
        _silhouette_avg = None

def load_and_process_data(db: Session):
    global _processed_data, _silhouette_avg
    if _processed_data is not None:
        print("Usando datos cacheados.")
        return _processed_data

    start_time = time.time()
    print("Cargando datos de la base de datos...")

    # Cargar solo las columnas necesarias
    stops = db.query(Stop.stop_id, Stop.trip_id, Stop.stop_name, Stop.stop_lat, Stop.stop_lon,
                     Stop.arrival_time, Stop.headway_secs, Stop.wait_time, Stop.delay,
                     Stop.simulated_delay, Stop.cluster).all()
    if not stops:
        raise HTTPException(status_code=404, detail="No se encontraron paradas en la base de datos")

    # Convertir a DataFrame
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

    print(f"Datos cargados en {time.time() - start_time:.2f} segundos:")
    print(full_data.head())

    # Verificar si los datos ya están procesados
    start_time = time.time()
    already_processed = (
        full_data['wait_time'].notnull().all() and
        full_data['delay'].notnull().all() and
        full_data['simulated_delay'].notnull().all() and
        full_data['cluster'].notnull().all() and
        (full_data['cluster'] != -1).any()
    )

    if not already_processed:
        print("Datos no considerados como procesados. Razones:")
        if not full_data['wait_time'].notnull().all():
            print(f"Valores nulos en wait_time: {full_data['wait_time'].isnull().sum()}")
        if not full_data['delay'].notnull().all():
            print(f"Valores nulos en delay: {full_data['delay'].isnull().sum()}")
        if not full_data['simulated_delay'].notnull().all():
            print(f"Valores nulos en simulated_delay: {full_data['simulated_delay'].isnull().sum()}")
        if not full_data['cluster'].notnull().all():
            print(f"Valores nulos en cluster: {full_data['cluster'].isnull().sum()}")
        if not (full_data['cluster'] != -1).any():
            print("Todos los valores de cluster son -1.")
        raise HTTPException(
            status_code=500,
            detail="Los datos en la base de datos no están completamente procesados. Faltan valores en wait_time, delay, simulated_delay o cluster."
        )
    print(f"Verificación de datos procesados en {time.time() - start_time:.2f} segundos")

    # Calcular métricas usando los datos ya procesados
    start_time = time.time()
    if _silhouette_avg is None:
        silhouette_avg = silhouette_score(
            full_data[['stop_lat', 'stop_lon', 'simulated_delay']].dropna(),
            full_data['cluster'].dropna()
        ) if (full_data['cluster'] != -1).any() else 0
        # Guardar silhouette_avg en un archivo
        with open(SILHOUETTE_FILE, "w") as f:
            f.write(str(silhouette_avg))
        print(f"Silhouette_avg calculado y guardado: {silhouette_avg}")
    else:
        silhouette_avg = _silhouette_avg
        print("Usando silhouette_avg cacheado.")
    print(f"Cálculo de silhouette_avg en {time.time() - start_time:.2f} segundos")

    start_time = time.time()
    cluster_delays = full_data.groupby('cluster')['simulated_delay'].mean()
    # Excluir el clúster -1
    cluster_delays = cluster_delays[cluster_delays.index != -1]
    # Convertir a una lista de diccionarios para el frontend
    cluster_delays_list = [
        {"cluster": int(cluster), "average_delay": float(delay)}
        for cluster, delay in cluster_delays.items()
    ]
    
    delays_by_stop = full_data.groupby('stop_id').agg({
        'stop_lat': 'first',
        'stop_lon': 'first',
        'stop_name': 'first',
        'simulated_delay': 'mean'
    }).reset_index()
 
    top_delays = delays_by_stop.sort_values('simulated_delay', ascending=False).head(10)
    print(f"Agrupación y cálculo de top_delays en {time.time() - start_time:.2f} segundos")

    # Generar el mapa de calor (usando un muestreo para reducir el tiempo)
    start_time = time.time()
    map_center = [delays_by_stop['stop_lat'].mean(), delays_by_stop['stop_lon'].mean()]
    m = folium.Map(location=map_center, zoom_start=12)
    # Muestreo: Usar solo el 10% de los datos para el mapa de calor
    sampled_data = delays_by_stop.sample(frac=0.1, random_state=42)
    heat_data = [[row['stop_lat'], row['stop_lon'], row['simulated_delay']] 
                 for _, row in sampled_data.iterrows()]
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
    print(f"Generación del mapa de calor en {time.time() - start_time:.2f} segundos")

    # Usar cluster_delays_list en lugar de cluster_delays
    _processed_data = (full_data, silhouette_avg, cluster_delays_list, delays_by_stop, top_delays, map_center)
    return _processed_data

def load_comments(db: Session):
    global _cached_comments
    if _cached_comments is not None:
        print("Usando comentarios cacheados.")
        return _cached_comments

    start_time = time.time()
    # Limitar el número de comentarios cargados (por ejemplo, los más recientes 1000)
    comments = db.query(Comment).order_by(Comment.fecha.desc()).limit(1000).all()
    if not comments:
        df = pd.DataFrame(columns=[
            "comentarios", "source", "fecha", "tiene_groseria", 
            "comentarios_censurados", "etiqueta", "etiqueta_predicha", "relevancia"
        ])
        _cached_comments = df
        return df

    df = pd.DataFrame([{
        "comentarios": comment.comentario,
        "source": comment.source,
        "fecha": comment.fecha,
        "tiene_groseria": comment.tiene_groseria,
        "comentarios_censurados": comment.comentario_censurado,
        "etiqueta": comment.etiqueta,
        "etiqueta_predicha": comment.etiqueta_predicha,
        "relevancia": comment.relevancia
    } for comment in comments])
    _cached_comments = df
    print(f"Carga de comentarios en {time.time() - start_time:.2f} segundos")
    return df

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
    start_time = time.time()
    logging.info("Procesando solicitud para /metrics...")

    full_data, silhouette_avg, cluster_delays, _, top_delays, map_center = load_and_process_data(db)
    comments_df = load_comments(db)

    # Eliminar duplicados basados en el comentario censurado
    start_time_comments = time.time()
    comments_df = comments_df.drop_duplicates(subset=["comentarios_censurados"])
    print(f"Eliminación de duplicados en comentarios en {time.time() - start_time_comments:.2f} segundos")

    # Preparar datos para las cards
    start_time_cards = time.time()
    positivos = comments_df[comments_df["etiqueta_predicha"] == "positivo_sugerencia"]
    top_positivos = positivos.sort_values(by="relevancia", ascending=False).head(5)[["comentarios_censurados"]].to_dict(orient="records")
    top_positivos = [item["comentarios_censurados"] for item in top_positivos]

    negativos = comments_df[comments_df["etiqueta_predicha"] == "negativo"]
    top_negativos = negativos.sort_values(by="relevancia", ascending=False).head(5)[["comentarios_censurados"]].to_dict(orient="records")
    top_negativos = [item["comentarios_censurados"] for item in top_negativos]

    neutros = comments_df[comments_df["etiqueta_predicha"] == "neutral"]
    top_neutros = neutros.sort_values(by="relevancia", ascending=False).head(5)[["comentarios_censurados"]].to_dict(orient="records")
    top_neutros = [item["comentarios_censurados"] for item in top_neutros]
    print(f"Preparación de tarjetas de comentarios en {time.time() - start_time_cards:.2f} segundos")

    cluster_ids = sorted([int(x) for x in full_data['cluster'].unique() if x != -1])

    total_time = time.time() - start_time
    logging.info(f"Solicitud para /metrics procesada en {total_time:.2f} segundos")
    return {
        "silhouette_avg": silhouette_avg,
        "cluster_delays": cluster_delays,  # Esto ahora es cluster_delays_list (lista de diccionarios)
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
    start_time = time.time()
    logging.info("Procesando solicitud para /classify_comment...")

    nuevo_comentario = data.comment
    if not nuevo_comentario:
        raise HTTPException(status_code=400, detail="No se proporcionó un comentario")

    # Verificar si el comentario ya existe para este usuario
    existing_comment = db.query(Comment).filter(
        Comment.comentario == nuevo_comentario,
        Comment.user_email == current_user.email
    ).first()
    if existing_comment:
        logging.info(f"Solicitud para /classify_comment procesada en {time.time() - start_time:.2f} segundos (comentario existente)")
        return {
            "comment": existing_comment.comentario,
            "censored_comment": existing_comment.comentario_censurado,
            "category": existing_comment.etiqueta_predicha,
            "relevance": existing_comment.relevancia
        }

    tiene_groseria = contiene_groseria(nuevo_comentario)
    comentario_censurado = censurar_groseria(nuevo_comentario)

    vector = vectorizer.transform([comentario_censurado])
    prediccion = comment_classifier.predict(vector)[0]
    relevancia = float(vector.sum())

    fecha_actual = datetime(2025, 4, 18).date()
    new_comment = Comment(
        user_email=current_user.email,
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
    # Invalidar el caché de comentarios después de agregar uno nuevo
    global _cached_comments
    _cached_comments = None

    logging.info(f"Solicitud para /classify_comment procesada en {time.time() - start_time:.2f} segundos")
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