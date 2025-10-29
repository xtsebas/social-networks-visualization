import panel as pn
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from wordcloud import WordCloud
import matplotlib.pyplot as plt

pn.extension('plotly')

df = pd.read_csv('data/traficogt_clean.csv')
df['datetime_gt'] = pd.to_datetime(df['datetime_gt'])

def prepare_classification_data(df):
    df_model = df.copy()
    df_model['engagement'] = df_model['likeCount'] + df_model['retweetCount'] + df_model['replyCount']
    df_model['high_engagement'] = (df_model['engagement'] > df_model['engagement'].median()).astype(int)

    features = ['hora', 'dia', 'mes_num', 'retweetCount']
    X = df_model[features].fillna(0)
    y = df_model['high_engagement']

    return train_test_split(X, y, test_size=0.3, random_state=42)

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

X_train, X_test, y_train, y_test = prepare_classification_data(df)
models_results = train_models(X_train, X_test, y_train, y_test)

month_selector = pn.widgets.MultiChoice(
    name='Seleccionar Meses',
    options=list(df['mes_nombre'].unique()),
    value=list(df['mes_nombre'].unique())
)

day_selector = pn.widgets.MultiChoice(
    name='Seleccionar Días',
    options=list(df['dia_semana'].unique()),
    value=list(df['dia_semana'].unique())
)

model_selector = pn.widgets.Select(
    name='Seleccionar Modelo',
    options=list(models_results.keys()),
    value=list(models_results.keys())[0]
)

comparison_selector = pn.widgets.MultiChoice(
    name='Seleccionar Modelos para Comparar',
    options=list(models_results.keys()),
    value=list(models_results.keys())
)

@pn.depends(month_selector.param.value, day_selector.param.value)
def exploration_view(months, days):
    df_filtered = df[df['mes_nombre'].isin(months) & df['dia_semana'].isin(days)]

    tweets_por_dia = df_filtered.groupby('dia_semana').size().reindex(['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'])
    fig1 = px.bar(x=tweets_por_dia.index, y=tweets_por_dia.values, title="Tweets por Día de la Semana", labels={'x': 'Día', 'y': 'Cantidad'}, color=tweets_por_dia.values, color_continuous_scale='Blues')

    tweets_por_hora = df_filtered.groupby('hora').size()
    fig2 = px.line(x=tweets_por_hora.index, y=tweets_por_hora.values, title="Tweets por Hora del Día", labels={'x': 'Hora', 'y': 'Cantidad'}, markers=True)
    fig2.update_traces(line_color='#1f77b4')

    engagement_por_mes = df_filtered.groupby('mes_nombre')[['likeCount', 'retweetCount', 'replyCount']].sum()
    fig3 = px.bar(engagement_por_mes, x=engagement_por_mes.index, y=['likeCount', 'retweetCount', 'replyCount'], title="Engagement por Mes", labels={'value': 'Cantidad', 'variable': 'Tipo', 'mes_nombre': 'Mes'}, barmode='group')

    top_users = df_filtered['user.username'].value_counts().head(10)
    fig4 = px.bar(x=top_users.values, y=top_users.index, orientation='h', title="Top 10 Usuarios más Activos", labels={'x': 'Tweets', 'y': 'Usuario'}, color=top_users.values, color_continuous_scale='Teal')

    # Gráfico de distribución de engagement
    df_filtered['engagement_total'] = df_filtered['likeCount'] + df_filtered['retweetCount'] + df_filtered['replyCount']
    fig5 = px.histogram(df_filtered, x='engagement_total', nbins=30, title="Distribución del Engagement Total", labels={'engagement_total': 'Engagement Total', 'count': 'Cantidad de Tweets'}, color_discrete_sequence=['#1f77b4'])
    fig5.update_layout(showlegend=False)

    # Nube de palabras
    wordcloud_pane = pn.pane.Markdown("*Generando nube de palabras...*")
    if len(df_filtered) > 0 and df_filtered['clean_text'].notna().sum() > 0:
        text = ' '.join(df_filtered['clean_text'].dropna().astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Blues').generate(text)
        fig_wc, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        wordcloud_pane = pn.pane.Matplotlib(fig_wc, tight=True)

    return pn.Column(
        pn.Row(pn.pane.Plotly(fig1), pn.pane.Plotly(fig2)),
        pn.Row(pn.pane.Plotly(fig3), pn.pane.Plotly(fig4)),
        pn.Row(pn.pane.Plotly(fig5), wordcloud_pane)
    )

@pn.depends(model_selector.param.value)
def models_view(selected_model):
    accuracy = models_results[selected_model]['accuracy']

    explanation = """
    ### Objetivo del Modelo

    Este modelo intenta predecir si un tweet sobre tráfico tendrá **alto engagement** (muchos likes, retweets y replies)
    basándose en características temporales y en el número de retweets que recibe en tiempo real.

    #### Features utilizados:
    - **Hora del tweet** (0-23): La hora del día en que se publica
    - **Día de la semana** (1-7): Lunes, martes, etc.
    - **Mes del año** (1-12): El mes en que se publica
    - **Retweet Count**: Cantidad de retweets recibidos

    #### Variable objetivo:
    - **Alto engagement** (1): Engagement por encima de la mediana
    - **Bajo engagement** (0): Engagement por debajo de la mediana
    """

    cm = models_results[selected_model]['confusion_matrix']
    fig_cm = px.imshow(cm, text_auto=True, title=f"Matriz de Confusión - {selected_model}", labels=dict(x="Predicción", y="Real", color="Cantidad"), color_continuous_scale='Blues')

    report = classification_report(y_test, models_results[selected_model]['predictions'], output_dict=True)
    df_report = pd.DataFrame(report).transpose()

    return pn.Column(
        pn.pane.Markdown(explanation),
        pn.pane.Markdown(f"### Accuracy: {accuracy:.2%}"),
        pn.pane.Markdown("*Porcentaje de predicciones correctas del modelo en datos de prueba*"),
        pn.Row(
            pn.pane.Plotly(fig_cm, width=500),
            pn.pane.DataFrame(df_report, width=500)
        )
    )

@pn.depends(comparison_selector.param.value)
def comparison_view(selected_models):
    if not selected_models:
        return pn.pane.Markdown("Seleccione al menos un modelo")
    
    comparison_data = []
    for model_name in selected_models:
        comparison_data.append({
            'Modelo': model_name,
            'Accuracy': models_results[model_name]['accuracy']
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    fig = px.bar(df_comparison, x='Modelo', y='Accuracy', title="Comparación de Accuracy entre Modelos", color='Accuracy', color_continuous_scale='Viridis')
    
    return pn.Column(
        pn.Row(
            pn.pane.DataFrame(df_comparison, width=400),
            pn.pane.Plotly(fig, width=600)
        )
    )

tabs = pn.Tabs(
    ('Exploración de Datos', pn.Column(
        pn.Row(month_selector, day_selector),
        exploration_view
    )),
    ('Modelos Predictivos', pn.Column(
        model_selector,
        models_view
    )),
    ('Comparación de Modelos', pn.Column(
        comparison_selector,
        comparison_view
    ))
)

dashboard = pn.template.FastListTemplate(
    title="Dashboard de Análisis TraficoGT",
    main=[tabs]
)

dashboard.servable()