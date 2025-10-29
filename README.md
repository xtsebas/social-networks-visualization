# Dashboard TraficoGT - Visualizaciones Interactivas

Este proyecto contiene 8 implementaciones de dashboards interactivos para analizar datos de TraficoGT usando diferentes frameworks.

## Requisitos

```bash
pip install -r requirements.txt
```

## Estructura de Archivos

```
proyecto/
├── data/
│   └── traficogt_clean.csv
├── streamlit.py
├── plotly_dash.py
├── voila.ipynb
├── panel.py
├── requirements.txt
└── README.md
```

## Ejecución de cada Dashboard

### 1. Streamlit

```bash
streamlit run streamlit.py
```

Se abrirá automáticamente en el navegador en `http://localhost:8501`

### 2. Plotly Dash

```bash
python plotly_dash.py
```

Abrir el navegador en `http://127.0.0.1:8050`

### 3. Voila

Primero convertir el archivo a notebook si es necesario, luego:

```bash
voila voila.ipynb
```

Se abrirá en el navegador en `http://localhost:8866`

### 4. Panel

```bash
panel serve panel.py --show
```

Se abrirá automáticamente en el navegador en `http://localhost:5006`

## Características de cada Dashboard

Todos los dashboards incluyen:

- 8 visualizaciones interactivas
- Filtros por mes y día de la semana
- 3 modelos de clasificación (Logistic Regression, Decision Tree, Random Forest)
- Gráficos enlazados que actualizan al cambiar filtros
- Comparación de modelos con métricas
- Matrices de confusión
- Análisis de engagement por período

## Visualizaciones Incluidas

1. Tweets por día de la semana
2. Tweets por hora del día
3. Engagement por mes
4. Top 10 usuarios más activos
5. Nube de palabras
6. Matriz de confusión del modelo seleccionado
7. Métricas de clasificación
8. Tabla comparativa de modelos
9. Gráfico de comparación de accuracy

