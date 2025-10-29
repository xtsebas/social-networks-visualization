import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

df = pd.read_csv('data/traficogt_clean.csv')
df['datetime_gt'] = pd.to_datetime(df['datetime_gt'])

def prepare_classification_data(df):
    df_model = df.copy()
    df_model['engagement'] = df_model['likeCount'] + df_model['retweetCount'] + df_model['replyCount']
    df_model['high_engagement'] = (df_model['engagement'] > df_model['engagement'].median()).astype(int)
    
    features = ['hora', 'dia', 'mes_num', 'retweetCount', 'likeCount', 'replyCount']
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

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Dashboard de Análisis TraficoGT", style={'textAlign': 'center', 'color': '#2c3e50'}),
    
    html.Div([
        html.Div([
            html.Label("Seleccionar Meses:"),
            dcc.Dropdown(
                id='month-selector',
                options=[{'label': m, 'value': m} for m in df['mes_nombre'].unique()],
                value=list(df['mes_nombre'].unique()),
                multi=True
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label("Seleccionar Días:"),
            dcc.Dropdown(
                id='day-selector',
                options=[{'label': d, 'value': d} for d in df['dia_semana'].unique()],
                value=list(df['dia_semana'].unique()),
                multi=True
            )
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    ], style={'padding': '20px'}),
    
    dcc.Tabs([
        dcc.Tab(label='Exploración de Datos', children=[
            html.Div([
                html.Div([
                    dcc.Graph(id='tweets-por-dia')
                ], style={'width': '48%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(id='tweets-por-hora')
                ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
            ]),
            
            html.Div([
                html.Div([
                    dcc.Graph(id='engagement-por-mes')
                ], style={'width': '48%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(id='top-usuarios')
                ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
            ])
        ]),
        
        dcc.Tab(label='Modelos Predictivos', children=[
            html.Div([
                html.Label("Seleccionar Modelo:"),
                dcc.Dropdown(
                    id='model-selector',
                    options=[{'label': m, 'value': m} for m in models_results.keys()],
                    value=list(models_results.keys())[0]
                )
            ], style={'padding': '20px'}),
            
            html.Div([
                html.Div([
                    html.H3(id='accuracy-text'),
                    dcc.Graph(id='confusion-matrix')
                ], style={'width': '48%', 'display': 'inline-block'}),
                
                html.Div([
                    html.H3("Métricas del Modelo"),
                    html.Div(id='metrics-table')
                ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
            ])
        ]),
        
        dcc.Tab(label='Comparación de Modelos', children=[
            html.Div([
                html.Label("Seleccionar modelos a comparar:"),
                dcc.Dropdown(
                    id='models-comparison-selector',
                    options=[{'label': m, 'value': m} for m in models_results.keys()],
                    value=list(models_results.keys()),
                    multi=True
                )
            ], style={'padding': '20px'}),
            
            html.Div([
                html.Div([
                    html.Div(id='comparison-table')
                ], style={'width': '48%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(id='comparison-chart')
                ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
            ])
        ])
    ])
])

@app.callback(
    [Output('tweets-por-dia', 'figure'),
     Output('tweets-por-hora', 'figure'),
     Output('engagement-por-mes', 'figure'),
     Output('top-usuarios', 'figure')],
    [Input('month-selector', 'value'),
     Input('day-selector', 'value')]
)
def update_exploration(selected_months, selected_days):
    df_filtered = df[df['mes_nombre'].isin(selected_months) & df['dia_semana'].isin(selected_days)]
    
    tweets_por_dia = df_filtered.groupby('dia_semana').size().reindex(['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'])
    fig1 = px.bar(x=tweets_por_dia.index, y=tweets_por_dia.values, title="Tweets por Día de la Semana", labels={'x': 'Día', 'y': 'Cantidad'}, color=tweets_por_dia.values, color_continuous_scale='Blues')
    
    tweets_por_hora = df_filtered.groupby('hora').size()
    fig2 = px.line(x=tweets_por_hora.index, y=tweets_por_hora.values, title="Tweets por Hora del Día", labels={'x': 'Hora', 'y': 'Cantidad'}, markers=True)
    fig2.update_traces(line_color='#1f77b4')
    
    engagement_por_mes = df_filtered.groupby('mes_nombre')[['likeCount', 'retweetCount', 'replyCount']].sum()
    fig3 = px.bar(engagement_por_mes, x=engagement_por_mes.index, y=['likeCount', 'retweetCount', 'replyCount'], title="Engagement por Mes", labels={'value': 'Cantidad', 'variable': 'Tipo', 'mes_nombre': 'Mes'}, barmode='group')
    
    top_users = df_filtered['user.username'].value_counts().head(10)
    fig4 = px.bar(x=top_users.values, y=top_users.index, orientation='h', title="Top 10 Usuarios más Activos", labels={'x': 'Tweets', 'y': 'Usuario'}, color=top_users.values, color_continuous_scale='Teal')
    
    return fig1, fig2, fig3, fig4

@app.callback(
    [Output('accuracy-text', 'children'),
     Output('confusion-matrix', 'figure'),
     Output('metrics-table', 'children')],
    [Input('model-selector', 'value')]
)
def update_model(selected_model):
    accuracy_text = f"Accuracy: {models_results[selected_model]['accuracy']:.2%}"
    
    cm = models_results[selected_model]['confusion_matrix']
    fig_cm = px.imshow(cm, text_auto=True, title=f"Matriz de Confusión - {selected_model}", labels=dict(x="Predicción", y="Real", color="Cantidad"), color_continuous_scale='Blues')
    
    report = classification_report(y_test, models_results[selected_model]['predictions'], output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    
    table = html.Table([
        html.Thead(html.Tr([html.Th(col) for col in df_report.columns])),
        html.Tbody([
            html.Tr([html.Td(df_report.iloc[i][col]) for col in df_report.columns])
            for i in range(len(df_report))
        ])
    ])
    
    return accuracy_text, fig_cm, table

@app.callback(
    [Output('comparison-table', 'children'),
     Output('comparison-chart', 'figure')],
    [Input('models-comparison-selector', 'value')]
)
def update_comparison(selected_models):
    if not selected_models:
        return html.Div("Seleccione al menos un modelo"), go.Figure()
    
    comparison_data = []
    for model_name in selected_models:
        comparison_data.append({
            'Modelo': model_name,
            'Accuracy': f"{models_results[model_name]['accuracy']:.2%}"
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    table = html.Table([
        html.Thead(html.Tr([html.Th(col) for col in df_comparison.columns])),
        html.Tbody([
            html.Tr([html.Td(df_comparison.iloc[i][col]) for col in df_comparison.columns])
            for i in range(len(df_comparison))
        ])
    ])
    
    df_comparison['Accuracy_numeric'] = [models_results[m]['accuracy'] for m in selected_models]
    fig = px.bar(df_comparison, x='Modelo', y='Accuracy_numeric', title="Comparación de Accuracy entre Modelos", color='Accuracy_numeric', color_continuous_scale='Viridis', labels={'Accuracy_numeric': 'Accuracy'})
    
    return table, fig

if __name__ == '__main__':
    app.run(debug=True)