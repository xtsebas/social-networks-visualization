import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc, precision_recall_curve, f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import time

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
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Calcular métricas
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_proba)

        # Learning curve
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train, cv=5, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
        )

        results[name] = {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_proba,
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'confusion_matrix': cm,
            'classification_report': report,
            'roc_curve': {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc},
            'precision_recall': {'precision': precision, 'recall': recall},
            'learning_curve': {'train_sizes': train_sizes, 'train_scores': train_scores, 'val_scores': val_scores},
            'training_time': training_time
        }

    return results

df = load_data()
X_train, X_test, y_train, y_test = prepare_classification_data(df)
models_results = train_models(X_train, X_test, y_train, y_test)

st.title("Dashboard de Análisis TraficoGT")

# Inicializar session state para drill-down
if 'selected_day' not in st.session_state:
    st.session_state.selected_day = None
if 'selected_month' not in st.session_state:
    st.session_state.selected_month = None

st.sidebar.header("Filtros Globales")
selected_months = st.sidebar.multiselect("Seleccionar meses", options=df['mes_nombre'].unique(), default=df['mes_nombre'].unique())
selected_days = st.sidebar.multiselect("Seleccionar días", options=df['dia_semana'].unique(), default=df['dia_semana'].unique())

# Botón para resetear drill-down
if st.sidebar.button("Resetear Selección Detallada"):
    st.session_state.selected_day = None
    st.session_state.selected_month = None

df_filtered = df[df['mes_nombre'].isin(selected_months) & df['dia_semana'].isin(selected_days)]

tab1, tab2, tab3 = st.tabs(["Exploración de Datos", "Modelos Predictivos", "Comparación de Modelos"])

with tab1:
    st.header("Exploración de Datos")

    # Selector de visualizaciones
    st.sidebar.subheader("Selector de Visualizaciones")
    show_temporal = st.sidebar.checkbox("Visualizaciones Temporales", value=True)
    show_engagement = st.sidebar.checkbox("Análisis de Engagement", value=True)
    show_users = st.sidebar.checkbox("Análisis de Usuarios", value=True)
    show_content = st.sidebar.checkbox("Análisis de Contenido", value=True)
    show_advanced = st.sidebar.checkbox("Visualizaciones Avanzadas", value=True)

    # Mostrar información de drill-down si está activo
    if st.session_state.selected_day:
        st.info(f"Vista detallada: {st.session_state.selected_day}")
        df_filtered = df_filtered[df_filtered['dia_semana'] == st.session_state.selected_day]

    if st.session_state.selected_month:
        st.info(f"Vista detallada: {st.session_state.selected_month}")
        df_filtered = df_filtered[df_filtered['mes_nombre'] == st.session_state.selected_month]

    # VISUALIZACIONES TEMPORALES (Gráficos enlazados)
    if show_temporal:
        st.subheader("Análisis Temporal")
        col1, col2 = st.columns(2)

        with col1:
            # Gráfico principal: Tweets por día
            tweets_por_dia = df_filtered.groupby('dia_semana').size().reindex(['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'])
            fig1 = px.bar(x=tweets_por_dia.index, y=tweets_por_dia.values,
                         title="Tweets por Día de la Semana (Click para detalles)",
                         labels={'x': 'Día', 'y': 'Cantidad'},
                         color=tweets_por_dia.values,
                         color_continuous_scale='Blues')
            fig1.update_layout(clickmode='event+select')
            st.plotly_chart(fig1, use_container_width=True, key='tweets_dia')

            # Selector manual para drill-down
            selected_day_manual = st.selectbox("O selecciona un día para ver detalles:",
                                              ['Todos'] + list(tweets_por_dia.index),
                                              index=0)
            if selected_day_manual != 'Todos':
                st.session_state.selected_day = selected_day_manual
                st.rerun()

        with col2:
            # Gráfico enlazado: Tweets por hora (afectado por selección de día)
            if st.session_state.selected_day:
                tweets_por_hora = df_filtered.groupby('hora').size()
                title = f"Tweets por Hora - {st.session_state.selected_day}"
            else:
                tweets_por_hora = df_filtered.groupby('hora').size()
                title = "Tweets por Hora del Día (Todos los días)"

            fig2 = px.line(x=tweets_por_hora.index, y=tweets_por_hora.values,
                          title=title,
                          labels={'x': 'Hora', 'y': 'Cantidad'},
                          markers=True)
            fig2.update_traces(line_color='#1f77b4', marker=dict(size=8))
            st.plotly_chart(fig2, use_container_width=True)

    # ANÁLISIS DE ENGAGEMENT
    if show_engagement:
        st.subheader("Análisis de Engagement")
        col3, col4 = st.columns(2)

        with col3:
            # Gráfico principal: Engagement por mes
            engagement_por_mes = df_filtered.groupby('mes_nombre')[['likeCount', 'retweetCount', 'replyCount']].sum()
            fig3 = px.bar(engagement_por_mes, x=engagement_por_mes.index,
                         y=['likeCount', 'retweetCount', 'replyCount'],
                         title="Engagement por Mes (Click para detalles)",
                         labels={'value': 'Cantidad', 'variable': 'Tipo', 'mes_nombre': 'Mes'},
                         barmode='group',
                         color_discrete_sequence=['#3498db', '#2ecc71', '#e74c3c'])
            st.plotly_chart(fig3, use_container_width=True)

            # Selector manual para drill-down de mes
            selected_month_manual = st.selectbox("Selecciona un mes para ver detalles:",
                                                ['Todos'] + list(df['mes_nombre'].unique()),
                                                index=0)
            if selected_month_manual != 'Todos':
                st.session_state.selected_month = selected_month_manual
                st.rerun()

        with col4:
            # Gráfico enlazado: Distribución de engagement
            df_filtered_copy = df_filtered.copy()
            df_filtered_copy['engagement_total'] = df_filtered_copy['likeCount'] + df_filtered_copy['retweetCount'] + df_filtered_copy['replyCount']

            if st.session_state.selected_month:
                title = f"Distribución del Engagement - {st.session_state.selected_month}"
            else:
                title = "Distribución del Engagement Total"

            fig5 = px.histogram(df_filtered_copy, x='engagement_total', nbins=30,
                               title=title,
                               labels={'engagement_total': 'Engagement Total', 'count': 'Cantidad de Tweets'},
                               color_discrete_sequence=['#1f77b4'])
            fig5.update_layout(showlegend=False)
            st.plotly_chart(fig5, use_container_width=True)

    # ANÁLISIS DE USUARIOS
    if show_users:
        st.subheader("Análisis de Usuarios")
        col5, col6 = st.columns(2)

        with col5:
            top_users = df_filtered['user.username'].value_counts().head(10)
            fig4 = px.bar(x=top_users.values, y=top_users.index, orientation='h',
                         title="Top 10 Usuarios más Activos",
                         labels={'x': 'Tweets', 'y': 'Usuario'},
                         color=top_users.values,
                         color_continuous_scale='Teal')
            st.plotly_chart(fig4, use_container_width=True)

        with col6:
            # Nueva visualización: Engagement promedio por usuario top
            top_user_names = df_filtered['user.username'].value_counts().head(10).index
            df_top_users = df_filtered[df_filtered['user.username'].isin(top_user_names)]
            df_top_users_eng = df_top_users.copy()
            df_top_users_eng['engagement_total'] = df_top_users_eng['likeCount'] + df_top_users_eng['retweetCount'] + df_top_users_eng['replyCount']
            avg_engagement = df_top_users_eng.groupby('user.username')['engagement_total'].mean().sort_values(ascending=True)

            fig_eng_users = px.bar(x=avg_engagement.values, y=avg_engagement.index, orientation='h',
                                  title="Engagement Promedio - Top 10 Usuarios",
                                  labels={'x': 'Engagement Promedio', 'y': 'Usuario'},
                                  color=avg_engagement.values,
                                  color_continuous_scale='Viridis')
            st.plotly_chart(fig_eng_users, use_container_width=True)

    # ANÁLISIS DE CONTENIDO
    if show_content:
        st.subheader("Análisis de Contenido")
        col7, col8 = st.columns(2)

        with col7:
            st.subheader("Nube de Palabras - Contenido de Tweets")
            if len(df_filtered) > 0 and df_filtered['clean_text'].notna().sum() > 0:
                text = ' '.join(df_filtered['clean_text'].dropna().astype(str))
                wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Blues').generate(text)
                fig_wc, ax = plt.subplots(figsize=(10, 6))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig_wc, use_container_width=True)
            else:
                st.info("No hay datos disponibles para generar la nube de palabras")

        with col8:
            # Nueva visualización: Longitud promedio de tweets por día
            df_filtered_copy = df_filtered.copy()
            df_filtered_copy['text_length'] = df_filtered_copy['clean_text'].fillna('').astype(str).apply(len)
            avg_length = df_filtered_copy.groupby('dia_semana')['text_length'].mean().reindex(['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'])

            fig_length = px.bar(x=avg_length.index, y=avg_length.values,
                               title="Longitud Promedio de Tweets por Día",
                               labels={'x': 'Día', 'y': 'Caracteres Promedio'},
                               color=avg_length.values,
                               color_continuous_scale='Oranges')
            st.plotly_chart(fig_length, use_container_width=True)

    # VISUALIZACIONES AVANZADAS
    if show_advanced:
        st.subheader("Visualizaciones Avanzadas")
        col9, col10 = st.columns(2)

        with col9:
            # Scatter plot: Relación entre retweets y likes
            df_scatter = df_filtered.sample(min(1000, len(df_filtered)))  # Limitar para rendimiento
            fig_scatter = px.scatter(df_scatter, x='retweetCount', y='likeCount',
                                    title="Relación entre Retweets y Likes",
                                    labels={'retweetCount': 'Retweets', 'likeCount': 'Likes'},
                                    color='replyCount',
                                    color_continuous_scale='Plasma',
                                    hover_data=['user.username'])
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col10:
            # Sunburst: Jerarquía Mes > Día > Hora
            df_sunburst = df_filtered.groupby(['mes_nombre', 'dia_semana']).size().reset_index(name='count')
            fig_sunburst = px.sunburst(df_sunburst,
                                       path=['mes_nombre', 'dia_semana'],
                                       values='count',
                                       title="Jerarquía Temporal: Mes > Día",
                                       color='count',
                                       color_continuous_scale='Blues')
            st.plotly_chart(fig_sunburst, use_container_width=True)

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

    # Métricas principales
    col_metrics = st.columns(4)
    with col_metrics[0]:
        st.metric("Accuracy", f"{models_results[selected_model]['accuracy']:.2%}")
    with col_metrics[1]:
        st.metric("F1-Score", f"{models_results[selected_model]['f1_score']:.2%}")
    with col_metrics[2]:
        st.metric("ROC-AUC", f"{models_results[selected_model]['roc_curve']['auc']:.2%}")
    with col_metrics[3]:
        st.metric("Tiempo de Entrenamiento", f"{models_results[selected_model]['training_time']:.3f}s")

    # Primera fila: Matriz de confusión y métricas detalladas
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Matriz de Confusión")
        cm = models_results[selected_model]['confusion_matrix']
        fig_cm = px.imshow(cm, text_auto=True,
                          title=f"Matriz de Confusión - {selected_model}",
                          labels=dict(x="Predicción", y="Real", color="Cantidad"),
                          color_continuous_scale='Blues',
                          x=['Bajo Engagement', 'Alto Engagement'],
                          y=['Bajo Engagement', 'Alto Engagement'])
        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        st.subheader("Métricas Detalladas")
        df_report = pd.DataFrame(models_results[selected_model]['classification_report']).transpose()
        st.dataframe(df_report.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)

    # Segunda fila: Curva ROC y Curva Precision-Recall
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Curva ROC")
        roc_data = models_results[selected_model]['roc_curve']
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=roc_data['fpr'], y=roc_data['tpr'],
                                     mode='lines',
                                     name=f'ROC (AUC = {roc_data["auc"]:.3f})',
                                     line=dict(color='#2ecc71', width=3)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                                     mode='lines',
                                     name='Random Classifier',
                                     line=dict(color='red', width=2, dash='dash')))
        fig_roc.update_layout(title=f'Curva ROC - {selected_model}',
                            xaxis_title='False Positive Rate',
                            yaxis_title='True Positive Rate',
                            showlegend=True)
        st.plotly_chart(fig_roc, use_container_width=True)

    with col4:
        st.subheader("Curva Precision-Recall")
        pr_data = models_results[selected_model]['precision_recall']
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=pr_data['recall'], y=pr_data['precision'],
                                   mode='lines',
                                   name='Precision-Recall',
                                   line=dict(color='#3498db', width=3),
                                   fill='tozeroy'))
        fig_pr.update_layout(title=f'Curva Precision-Recall - {selected_model}',
                           xaxis_title='Recall',
                           yaxis_title='Precision',
                           showlegend=True)
        st.plotly_chart(fig_pr, use_container_width=True)

    # Tercera fila: Curva de Aprendizaje
    st.subheader("Curva de Aprendizaje")
    lc_data = models_results[selected_model]['learning_curve']
    train_mean = np.mean(lc_data['train_scores'], axis=1)
    train_std = np.std(lc_data['train_scores'], axis=1)
    val_mean = np.mean(lc_data['val_scores'], axis=1)
    val_std = np.std(lc_data['val_scores'], axis=1)

    fig_lc = go.Figure()
    # Training score
    fig_lc.add_trace(go.Scatter(x=lc_data['train_sizes'], y=train_mean,
                               mode='lines+markers',
                               name='Training Score',
                               line=dict(color='#3498db', width=2)))
    fig_lc.add_trace(go.Scatter(x=lc_data['train_sizes'], y=train_mean + train_std,
                               mode='lines',
                               line=dict(width=0),
                               showlegend=False))
    fig_lc.add_trace(go.Scatter(x=lc_data['train_sizes'], y=train_mean - train_std,
                               mode='lines',
                               line=dict(width=0),
                               fill='tonexty',
                               fillcolor='rgba(52, 152, 219, 0.2)',
                               showlegend=False))
    # Validation score
    fig_lc.add_trace(go.Scatter(x=lc_data['train_sizes'], y=val_mean,
                               mode='lines+markers',
                               name='Validation Score',
                               line=dict(color='#2ecc71', width=2)))
    fig_lc.add_trace(go.Scatter(x=lc_data['train_sizes'], y=val_mean + val_std,
                               mode='lines',
                               line=dict(width=0),
                               showlegend=False))
    fig_lc.add_trace(go.Scatter(x=lc_data['train_sizes'], y=val_mean - val_std,
                               mode='lines',
                               line=dict(width=0),
                               fill='tonexty',
                               fillcolor='rgba(46, 204, 113, 0.2)',
                               showlegend=False))

    fig_lc.update_layout(title=f'Curva de Aprendizaje - {selected_model}',
                        xaxis_title='Tamaño del Conjunto de Entrenamiento',
                        yaxis_title='Score',
                        showlegend=True)
    st.plotly_chart(fig_lc, use_container_width=True)

    st.caption("""
    **Interpretación de la Curva de Aprendizaje:**
    - Si ambas curvas convergen y el score es alto: el modelo funciona bien
    - Si hay mucha diferencia entre training y validation: puede haber overfitting
    - Si ambas curvas tienen score bajo: puede necesitar más features o un modelo más complejo
    """)

with tab3:
    st.header("Comparación de Modelos")

    st.info("Compara el desempeño de diferentes modelos para elegir el mejor según tus necesidades.")

    selected_models = st.multiselect("Seleccionar modelos a comparar",
                                     list(models_results.keys()),
                                     default=list(models_results.keys()))

    if selected_models:
        # Tabla comparativa completa
        comparison_data = []
        for model_name in selected_models:
            comparison_data.append({
                'Modelo': model_name,
                'Accuracy': f"{models_results[model_name]['accuracy']:.4f}",
                'F1-Score': f"{models_results[model_name]['f1_score']:.4f}",
                'ROC-AUC': f"{models_results[model_name]['roc_curve']['auc']:.4f}",
                'Tiempo (s)': f"{models_results[model_name]['training_time']:.4f}",
                'Precisión (clase 1)': f"{models_results[model_name]['classification_report']['1']['precision']:.4f}",
                'Recall (clase 1)': f"{models_results[model_name]['classification_report']['1']['recall']:.4f}"
            })

        df_comparison = pd.DataFrame(comparison_data)

        st.subheader("Tabla Comparativa Completa")
        st.dataframe(df_comparison, use_container_width=True, hide_index=True)

        # Gráficos de comparación
        st.subheader("Visualización de Métricas")

        # Preparar datos para gráficos
        metrics_to_plot = ['Accuracy', 'F1-Score', 'ROC-AUC']
        comparison_numeric = []
        for model_name in selected_models:
            comparison_numeric.append({
                'Modelo': model_name,
                'Accuracy': models_results[model_name]['accuracy'],
                'F1-Score': models_results[model_name]['f1_score'],
                'ROC-AUC': models_results[model_name]['roc_curve']['auc'],
                'Tiempo': models_results[model_name]['training_time']
            })

        df_comparison_numeric = pd.DataFrame(comparison_numeric)

        col1, col2 = st.columns(2)

        with col1:
            # Gráfico de barras agrupadas
            df_melted = df_comparison_numeric.melt(id_vars=['Modelo'],
                                                   value_vars=metrics_to_plot,
                                                   var_name='Métrica',
                                                   value_name='Valor')
            fig_bars = px.bar(df_melted, x='Modelo', y='Valor', color='Métrica',
                            title="Comparación de Métricas Principales",
                            barmode='group',
                            color_discrete_sequence=['#3498db', '#2ecc71', '#e74c3c'])
            fig_bars.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig_bars, use_container_width=True)

        with col2:
            # Gráfico de radar para comparación multidimensional
            if len(selected_models) <= 3:  # Solo para máximo 3 modelos para claridad
                fig_radar = go.Figure()
                for model_name in selected_models:
                    values = [
                        models_results[model_name]['accuracy'],
                        models_results[model_name]['f1_score'],
                        models_results[model_name]['roc_curve']['auc'],
                        models_results[model_name]['classification_report']['1']['precision'],
                        models_results[model_name]['classification_report']['1']['recall']
                    ]
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values,
                        theta=['Accuracy', 'F1-Score', 'ROC-AUC', 'Precision', 'Recall'],
                        fill='toself',
                        name=model_name
                    ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    title="Comparación Radar de Métricas"
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            else:
                # Si hay más de 3 modelos, mostrar tiempo de entrenamiento
                fig_time = px.bar(df_comparison_numeric, x='Modelo', y='Tiempo',
                                title="Tiempo de Entrenamiento (segundos)",
                                color='Tiempo',
                                color_continuous_scale='Oranges')
                st.plotly_chart(fig_time, use_container_width=True)

        # Comparación de curvas ROC
        st.subheader("Comparación de Curvas ROC")
        fig_roc_comparison = go.Figure()

        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
        for idx, model_name in enumerate(selected_models):
            roc_data = models_results[model_name]['roc_curve']
            fig_roc_comparison.add_trace(go.Scatter(
                x=roc_data['fpr'],
                y=roc_data['tpr'],
                mode='lines',
                name=f'{model_name} (AUC = {roc_data["auc"]:.3f})',
                line=dict(color=colors[idx % len(colors)], width=3)
            ))

        # Línea diagonal de referencia
        fig_roc_comparison.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='gray', width=2, dash='dash')
        ))

        fig_roc_comparison.update_layout(
            title='Comparación de Curvas ROC',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            showlegend=True,
            height=500
        )
        st.plotly_chart(fig_roc_comparison, use_container_width=True)

        # Matriz de comparación de confusión
        st.subheader("Matrices de Confusión Comparadas")
        cols_cm = st.columns(len(selected_models))

        for idx, model_name in enumerate(selected_models):
            with cols_cm[idx]:
                cm = models_results[model_name]['confusion_matrix']
                fig_cm_small = px.imshow(cm, text_auto=True,
                                        title=model_name,
                                        labels=dict(x="Pred", y="Real", color="Count"),
                                        color_continuous_scale='Blues',
                                        x=['Bajo', 'Alto'],
                                        y=['Bajo', 'Alto'])
                fig_cm_small.update_layout(height=300)
                st.plotly_chart(fig_cm_small, use_container_width=True)

        # Recomendación
        st.subheader("Recomendación")
        best_accuracy = max([models_results[m]['accuracy'] for m in selected_models])
        best_f1 = max([models_results[m]['f1_score'] for m in selected_models])
        best_roc = max([models_results[m]['roc_curve']['auc'] for m in selected_models])

        best_models = {
            'Accuracy': [m for m in selected_models if models_results[m]['accuracy'] == best_accuracy][0],
            'F1-Score': [m for m in selected_models if models_results[m]['f1_score'] == best_f1][0],
            'ROC-AUC': [m for m in selected_models if models_results[m]['roc_curve']['auc'] == best_roc][0]
        }

        st.success(f"""
        **Mejores modelos por métrica:**
        - **Mejor Accuracy ({best_accuracy:.2%}):** {best_models['Accuracy']}
        - **Mejor F1-Score ({best_f1:.2%}):** {best_models['F1-Score']}
        - **Mejor ROC-AUC ({best_roc:.2%}):** {best_models['ROC-AUC']}

        **Tip:** Si necesitas balance entre precisión y recall, elige el modelo con mejor F1-Score.
        Si te importa más la capacidad de discriminación, elige el de mejor ROC-AUC.
        """)

    else:
        st.warning("Por favor selecciona al menos un modelo para comparar.")