import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

st.set_page_config(page_title="Dashboard TraficoGT", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('data/traficogt_clean.csv')
    df['datetime_gt'] = pd.to_datetime(df['datetime_gt'])
    return df

@st.cache_data
def prepare_classification_data(df):
    df_model = df.copy()
    df_model['engagement'] = df_model['likeCount'] + df_model['retweetCount'] + df_model['replyCount']
    df_model['high_engagement'] = (df_model['engagement'] > df_model['engagement'].median()).astype(int)

    features = ['hora', 'dia', 'mes_num', 'retweetCount']
    X = df_model[features].fillna(0)
    y = df_model['high_engagement']

    return train_test_split(X, y, test_size=0.3, random_state=42)

@st.cache_data
def train_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            'model': model,
            'predictions': y_pred,
            'accuracy': accuracy_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    return results

df = load_data()
X_train, X_test, y_train, y_test = prepare_classification_data(df)
models_results = train_models(X_train, X_test, y_train, y_test)

st.title("Dashboard de Análisis TraficoGT")

st.sidebar.header("Filtros")
selected_months = st.sidebar.multiselect("Seleccionar meses", options=df['mes_nombre'].unique(), default=df['mes_nombre'].unique())
selected_days = st.sidebar.multiselect("Seleccionar días", options=df['dia_semana'].unique(), default=df['dia_semana'].unique())

df_filtered = df[df['mes_nombre'].isin(selected_months) & df['dia_semana'].isin(selected_days)]

tab1, tab2, tab3 = st.tabs(["Exploración de Datos", "Modelos Predictivos", "Comparación de Modelos"])

with tab1:
    st.header("Exploración de Datos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        tweets_por_dia = df_filtered.groupby('dia_semana').size().reindex(['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'])
        fig1 = px.bar(x=tweets_por_dia.index, y=tweets_por_dia.values, title="Tweets por Día de la Semana", labels={'x': 'Día', 'y': 'Cantidad'}, color=tweets_por_dia.values, color_continuous_scale='Blues')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        tweets_por_hora = df_filtered.groupby('hora').size()
        fig2 = px.line(x=tweets_por_hora.index, y=tweets_por_hora.values, title="Tweets por Hora del Día", labels={'x': 'Hora', 'y': 'Cantidad'}, markers=True)
        fig2.update_traces(line_color='#1f77b4')
        st.plotly_chart(fig2, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        engagement_por_mes = df_filtered.groupby('mes_nombre')[['likeCount', 'retweetCount', 'replyCount']].sum()
        fig3 = px.bar(engagement_por_mes, x=engagement_por_mes.index, y=['likeCount', 'retweetCount', 'replyCount'], title="Engagement por Mes", labels={'value': 'Cantidad', 'variable': 'Tipo', 'mes_nombre': 'Mes'}, barmode='group')
        st.plotly_chart(fig3, use_container_width=True)
    
    with col4:
        top_users = df_filtered['user.username'].value_counts().head(10)
        fig4 = px.bar(x=top_users.values, y=top_users.index, orientation='h', title="Top 10 Usuarios más Activos", labels={'x': 'Tweets', 'y': 'Usuario'}, color=top_users.values, color_continuous_scale='Teal')
        st.plotly_chart(fig4, use_container_width=True)

with tab2:
    st.header("Modelos Predictivos")

    st.info("""
    **Objetivo del Modelo:**
    Este modelo intenta predecir si un tweet sobre tráfico tendrá **alto engagement** (muchos likes, retweets y replies)
    basándose en características temporales y en el número de retweets que recibe en tiempo real.

    **Features utilizados:**
    - **Hora del tweet** (0-23): La hora del día en que se publica
    - **Día de la semana** (1-7): Lunes, martes, etc.
    - **Mes del año** (1-12): El mes en que se publica
    - **Retweet Count**: Cantidad de retweets recibidos

    **Variable objetivo:**
    - **Alto engagement** (1): Engagement por encima de la mediana
    - **Bajo engagement** (0): Engagement por debajo de la mediana
    """)

    selected_model = st.selectbox("Seleccionar Modelo", list(models_results.keys()))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Accuracy: {models_results[selected_model]['accuracy']:.2%}")
        st.caption(f"Porcentaje de predicciones correctas del modelo en datos de prueba")

        cm = models_results[selected_model]['confusion_matrix']
        fig_cm = px.imshow(cm, text_auto=True, title=f"Matriz de Confusión - {selected_model}", labels=dict(x="Predicción", y="Real", color="Cantidad"), color_continuous_scale='Blues')
        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        st.subheader("Métricas del Modelo")
        st.caption("Detalles de rendimiento: Precisión, Recall y F1-Score")
        report = classification_report(y_test, models_results[selected_model]['predictions'], output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        st.dataframe(df_report)

with tab3:
    st.header("Comparación de Modelos")
    
    selected_models = st.multiselect("Seleccionar modelos a comparar", list(models_results.keys()), default=list(models_results.keys()))
    
    if selected_models:
        comparison_data = []
        for model_name in selected_models:
            comparison_data.append({
                'Modelo': model_name,
                'Accuracy': models_results[model_name]['accuracy']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(df_comparison, use_container_width=True)
        
        with col2:
            fig_comparison = px.bar(df_comparison, x='Modelo', y='Accuracy', title="Comparación de Accuracy entre Modelos", color='Accuracy', color_continuous_scale='Viridis')
            st.plotly_chart(fig_comparison, use_container_width=True)