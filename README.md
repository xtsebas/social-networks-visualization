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
<img width="1897" height="990" alt="image" src="https://github.com/user-attachments/assets/e7d93621-fe8c-4e29-a809-dfec5c808aa1" />


### 2. Plotly Dash

```bash
python plotly_dash.py
```

Abrir el navegador en `http://127.0.0.1:8050`
<img width="1887" height="984" alt="image" src="https://github.com/user-attachments/assets/65aeff52-e4f1-4202-9d7c-1d809772a1dc" />


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
<img width="1918" height="951" alt="image" src="https://github.com/user-attachments/assets/fc3a67e7-8b26-43ef-8c42-bcc7f89dceb2" />


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

