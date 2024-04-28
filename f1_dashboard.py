import matplotlib
matplotlib.use('Agg')

from datetime import datetime, timedelta
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import fastf1
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import base64
import io
import plotly.graph_objs as go
import fastf1.plotting
from plotly.validator_cache import ValidatorCache
from plotly.graph_objects import Layout
def update_session(year, race):
    global session
    session = fastf1.get_session(year, race, 'R')
    session.load(weather=False, messages=False)
    return session

app = dash.Dash(__name__)
app.title = "F1 Dashboard"

races_by_year = {
    2018: ['australia', 'bahrain', 'china', 'azerbaijan', 'spain', 'monaco', 
           'canada', 'france', 'austria', 'britain', 'germany', 'hungary', 
           'belgium', 'italy', 'singapore', 'russia', 'japan', 'usa', 
           'mexico', 'brazil', 'abudhabi'],
    2019: ['australia', 'bahrain', 'china', 'azerbaijan', 'spain', 'monaco', 
           'canada', 'france', 'austria', 'britain', 'germany', 'hungary', 
           'belgium', 'italy', 'singapore', 'russia', 'japan', 'usa', 
           'mexico', 'brazil', 'abudhabi'],
    2020: ['austria', 'styria', 'hungary', 'greatbritain', 'spain', 
           'belgium', 'italy', 'tuscany', 'russia', 'eifel', 'portugal', 
           'imola', 'turkey', 'bahrain', 'sakhir', 'abudhabi'],
    2021: ['bahrain', 'emiliaromagna', 'portugal', 'spain', 'monaco', 'azerbaijan', 
           'france', 'styria', 'austria', 'britain', 'hungary', 'belgium', 
           'netherlands', 'italy', 'russia', 'turkey', 'usa', 'mexico', 
           'brazil', 'qatar', 'saudi', 'abudhabi'],
    2022: ['bahrain', 'saudi', 'australia', 'vietnam', 'netherlands', 'spain', 
           'monaco', 'azerbaijan', 'canada', 'france', 'austria', 'britain', 
           'hungary', 'belgium', 'italy', 'singapore', 'japan', 'usa', 
           'mexico', 'brazil', 'qatar', 'abudhabi'],
    2023: ['australia', 'bahrain', 'china', 'spain', 'monaco', 'azerbaijan', 
           'canada', 'france', 'austria', 'britain', 'hungary', 'belgium', 
           'netherlands', 'italy', 'russia', 'singapore', 'japan', 'usa', 
           'mexico', 'brazil', 'abudhabi'],
    2024: ['china', 'japan', 'australia', 'bahrain']
}

initial_year = 2023
initial_race = 'spain'
session = fastf1.get_session(initial_year, initial_race, 'R')
session.load(weather=False, messages=False)

drivers = list(session.results['FullName'])
legend_colors = {'HARD': '#f0f0ec', 'INTERMEDIATE': '#43b02a', 'MEDIUM': '#ffd12e', 'SOFT': '#da291c', 'TEST-UNKNOWN': '#434649', 'UNKNOWN': '#00ffff', 'WET': '#0067ad'}
legend_entries = []
for compound, color in legend_colors.items():
    legend_entries.append(
        html.Div([
            html.Div(style={'background-color': color, 'width': '20px', 'height': '20px', 'display': 'inline-block'}),
            html.Span(f" {compound}", style={'margin-left': '5px'})
        ], style={'margin-right': '10px', 'margin-bottom': '5px'})
    )

with open("logo.png", "rb") as logo_file:
    encoded_logo = base64.b64encode(logo_file.read()).decode('ascii')

app.layout = html.Div([
    html.Div([
        html.Img(src='data:image/png;base64,{}'.format(encoded_logo), style={'height': '100px', 'margin-right': '20px'}),
        html.H1("Formula 1 Dashboard", style={'text-align': 'center', 'font-family': 'Arial, sans-serif'}),
    ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
    
    html.Div([
        html.Label('Select Year', style={'font-size': '25px' , 'margin-right' : '20px' , 'margin-bottom' : '20px'}),
        dcc.Dropdown(
            id='year-dropdown',
            options=[{'label': str(year), 'value': year} for year in range(2018, 2025)],  
            value=2023,
            style={'width': '170px', 'display': 'inline-block', 'font-size': '23px'}
        ),
        html.Label('Select Track', style={'font-size': '25px', 'margin-left': '120px', 'margin-right': '20px'}),
        dcc.Dropdown(
            id='race-dropdown',
            value='spain',  
            style={'width': '270px', 'display': 'inline-block', 'font-size': '23px'}
        )
    ], style={'text-align': 'center', 'margin-bottom': '20px'}),

    html.Div([
        html.H1("Top 3 Drivers in the Race", style={'text-align': 'center'}),
        html.Div(id='top3-drivers', style={'text-align': 'center', 'display': 'flex', 'justify-content': 'center'})
    ], style={'margin-bottom': '20px'}),
    html.Div([
        html.Div([
            html.H1("Leaderboard", style={'text-align': 'center'}),
            html.Div(id='leaderboard'), 
        ], style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top'}),
        html.Div([
            html.H1("Track Dominance Plot for top 3 drivers", style={'text-align': 'center'}),
            html.Div(id='track-dominance-plot'),
        ], style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top' }),
    ], style={'text-align': 'center', 'margin-bottom': '20px', 'box-shadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2)', 'border-radius': '10px'}),

    html.Div([
        html.H1("Lap-Time vs Lap Number Plot", style={'text-align': 'center'}),
        dcc.Graph(id='lap-time-comparison-chart'),
        dcc.Dropdown(
            id='driver-dropdown',
            options=[{'label': driver, 'value': driver} for driver in drivers],
            value=['Lewis Hamilton'], 
            multi=True,
            style={'width': '100%', 'margin': '20px auto', 'display': 'block' }
        )
    ], style={'text-align': 'center','width': '90%', 'margin': 'auto', 'padding': '20px', 'box-shadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2)', 'border-radius': '10px','margin-bottom': '20px' }),

    html.Div([
    html.H1("Tyre Strategies During the Race", style={'text-align': 'center', 'margin-bottom': '20px'}),
    html.Div(legend_entries,style={'display': 'flex', 'justify-content': 'center'}),
    dcc.Graph(id='tyre-strategy-plot'),
], style={'width': '80%', 'margin': 'auto', 'margin-bottom': '20px' ,  'padding': '20px', 'box-shadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2)', 'border-radius': '10px'}),

html.Div([
    html.H1("Fastest lap Speed trace plot", style={'text-align': 'center'}),
    dcc.Graph(id='speed-trace-graph'),
    dcc.Dropdown(
            id='driver-dropdown2',
            options=[{'label': driver, 'value': driver} for driver in drivers],
            value=['Lewis Hamilton'], 
            multi=True,
            style={'width': '100%', 'margin': '20px auto', 'display': 'block' }
        )
] , style={'text-align': 'center','width': '90%', 'margin': 'auto', 'padding': '20px', 'box-shadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2)', 'border-radius': '10px','margin-bottom': '20px' })
])

@app.callback(
    Output('race-dropdown', 'options'),
    [Input('year-dropdown', 'value')]
)
def update_race_options(selected_year):
    races_for_year = races_by_year.get(selected_year, [])
    race_options = [{'label': race.capitalize(), 'value': race} for race in races_for_year]
    return race_options

@app.callback(
    Output('top3-drivers', 'children'),
    [Input('year-dropdown', 'value'),
     Input('race-dropdown', 'value')]
)
def update_top3_drivers(year, race):
    session = update_session(year, race)
    top3_drivers = session.results.head(3)
    i = 0
    top3_boxes = []
    for idx, row in top3_drivers.iterrows():
        driver = row['FullName']
        abbreviation = row['Abbreviation']
        team = row['TeamName']
        i += 1
        headshot_url = row['HeadshotUrl']
        team_color = row['TeamColor']
        top3_boxes.append(
            html.Div([
                html.Img(src=headshot_url, style={'width': '180px', 'height': '180px', 'border-radius': '50%', 'background-color': 'white'}),
                html.P("Position " + str(i), style={'font-size': '18px', 'margin': '0px', 'font-weight': 'bold'}),
                html.P(driver, style={'font-size': '24px', 'margin-bottom': '5px', 'font-weight': 'bold'}),
                html.P("Code : " + abbreviation, style={'font-size': '18px', 'margin': '0px', 'font-weight': 'bold'}),
                html.P(team, style={'font-size': '18px', 'margin': '0px', 'font-weight': 'bold'})
            ], style={'background-color': f'#{team_color}', 'padding-left': '20px', 'padding-right': '20px', 'padding-top': '10px', 'padding-bottom': '10px', 'margin': '10px', 'text-align': 'center', 'border-radius': '10px', 'display': 'inline-block'})
        )

    return html.Div(top3_boxes, style={'display': 'flex'})

@app.callback(
    Output('leaderboard', 'children'),
    [Input('year-dropdown', 'value'),
     Input('race-dropdown', 'value')]
)
def update_leaderboard(year, race):
    session = update_session(year, race)
    results = session.results
    leaderboard_table = html.Table([
        html.Thead(html.Tr([
            html.Th("Position", style={'color': 'white', 'background-color': '#1f77b4', 'padding-left': '20px' , 'padding-right': '20px' , 'padding-top': '10px' , 'padding-bottom': '10px'}),
            html.Th("Driver", style={'color': 'white', 'background-color': '#1f77b4', 'padding-left': '20px' , 'padding-right': '20px' , 'padding-top': '10px' , 'padding-bottom': '10px'}),
            html.Th("Team", style={'color': 'white', 'background-color': '#1f77b4', 'padding-left': '20px' , 'padding-right': '20px' , 'padding-top': '10px' , 'padding-bottom': '10px'}),
            html.Th("Time", style={'color': 'white', 'background-color': '#1f77b4', 'padding-left': '20px' , 'padding-right': '20px' , 'padding-top': '10px' , 'padding-bottom': '10px'}),
            html.Th("Race Status", style={'color': 'white', 'background-color': '#1f77b4', 'padding-left': '20px' , 'padding-right': '20px' , 'padding-top': '10px' , 'padding-bottom': '10px'}),
            html.Th("Points", style={'color': 'white', 'background-color': '#1f77b4', 'padding-left': '20px' , 'padding-right': '20px' , 'padding-top': '10px' , 'padding-bottom': '10px'})
        ]))
    ])
    for idx, row in results.iterrows():
        position = row['Position']
        driver = row['FullName']
        team = row['TeamName']
        if row['Status'] != 'Finished':
            time = ' - '
            race_status = row['Status']
        else:
            time = '+' + str(row['Time']).split()[-1][:-3] if int(idx) > 1 else str(row['Time']).split()[-1][:-3]  # Extract time part from Timedelta and truncate last three zeros
            race_status = 'Finished'
        points = row['Points']
        leaderboard_table.children.append(
            html.Tr([
                html.Td(position, style={'padding-left': '20px' , 'padding-right': '20px' , 'padding-top': '10px' , 'padding-bottom': '10px'}),
                html.Td(driver, style={'padding-left': '20px' , 'padding-right': '20px' , 'padding-top': '10px' , 'padding-bottom': '10px'}),
                html.Td(team, style={'padding-left': '20px' , 'padding-right': '20px' , 'padding-top': '10px' , 'padding-bottom': '10px'}),
                html.Td(time, style={'padding-left': '20px' , 'padding-right': '20px' , 'padding-top': '10px' , 'padding-bottom': '10px'}),
                html.Td(race_status, style={'padding-left': '20px' , 'padding-right': '20px' , 'padding-top': '10px' , 'padding-bottom': '10px'}),
                html.Td(points, style={'padding-left': '20px' , 'padding-right': '20px' , 'padding-top': '10px' , 'padding-bottom': '10px'})
            ])
        )

    return leaderboard_table

@app.callback(
    Output('track-dominance-plot', 'children'),
    [Input('year-dropdown', 'value'),
     Input('race-dropdown', 'value')]
)
def update_track_dominance(year, race):
    image_base64 = generate_track_dominance_image(year, race)
    return html.Img(src='data:image/png;base64,{}'.format(image_base64))

def generate_track_dominance_image(year, race):
    session_event = fastf1.get_session(year, race, 'Race')
    session_event.load()
    results = session_event.results
    top3_drivers = results.head(3)
    top3_driver_names = top3_drivers['Abbreviation'].tolist()
    telemetry_data = []
    for driver_name in top3_driver_names:
        fastest_lap_driver = session_event.laps.pick_driver(driver_name).pick_fastest()
        telemetry_driver = fastest_lap_driver.get_telemetry().add_distance()
        telemetry_driver['Driver'] = driver_name
        telemetry_data.append(telemetry_driver)
    telemetry_drivers = pd.concat(telemetry_data, ignore_index=True)
    num_minisectors = 7 * 3
    total_distance = max(telemetry_drivers['Distance'])
    minisector_length = total_distance / num_minisectors
    minisectors = [0]
    for i in range(0, (num_minisectors - 1)):
        minisectors.append(minisector_length * (i + 1))
    telemetry_drivers['Minisector'] = telemetry_drivers['Distance'].apply(
        lambda dist: int((dist // minisector_length) + 1)
    )
    average_speed = telemetry_drivers.groupby(['Minisector', 'Driver'])['Speed'].mean().reset_index()
    fastest_driver = average_speed.loc[average_speed.groupby(['Minisector'])['Speed'].idxmax()]
    fastest_driver = fastest_driver[['Minisector', 'Driver']].rename(columns={'Driver': 'Fastest_driver'})
    telemetry_drivers = telemetry_drivers.merge(fastest_driver, on=['Minisector'])
    telemetry_drivers = telemetry_drivers.sort_values(by=['Distance'])
    telemetry_drivers.loc[telemetry_drivers['Fastest_driver'] == top3_driver_names[0], 'Fastest_driver_int'] = 1
    telemetry_drivers.loc[telemetry_drivers['Fastest_driver'] == top3_driver_names[1], 'Fastest_driver_int'] = 2
    telemetry_drivers.loc[telemetry_drivers['Fastest_driver'] == top3_driver_names[2], 'Fastest_driver_int'] = 3
    x = np.array(telemetry_drivers['X'].values)
    y = np.array(telemetry_drivers['Y'].values)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    fastest_driver_array = telemetry_drivers['Fastest_driver_int'].to_numpy().astype(float)
    cmap = plt.get_cmap('brg', 3)
    lc_comp = LineCollection(segments, norm=plt.Normalize(1, cmap.N + 1), cmap=cmap)
    lc_comp.set_array(fastest_driver_array)
    lc_comp.set_linewidth(5)
    plt.rcParams['figure.figsize'] = [8, 8] 
    plt.gca().add_collection(lc_comp)
    plt.axis('equal')
    plt.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)
    legend_colors = ['#0000FF', '#FF0000', '#00FF00'] 
    legend_labels = top3_driver_names 
    for i, color in enumerate(legend_colors):
        plt.scatter([], [], color=color, label=legend_labels[i], s=100) 
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=len(legend_labels), fontsize='large')
    plt.tight_layout()
    img_bytes_io = io.BytesIO()
    plt.savefig(img_bytes_io, format='png', bbox_inches='tight', facecolor='black')
    img_bytes_io.seek(0)
    image_base64 = base64.b64encode(img_bytes_io.read()).decode('ascii')
    plt.close()
    return image_base64

def lap_time_to_seconds(lap_time):
    if isinstance(lap_time, pd.Timedelta):
        return lap_time.total_seconds()
    elif pd.isna(lap_time): 
        return None
    else: 
        lap_time_split = lap_time.split()
        if len(lap_time_split) == 3: 
            lap_time = lap_time_split[2]
        elif len(lap_time_split) == 1: 
            lap_time = lap_time_split[0]
        else:
            return np.nan
        lap_time = datetime.strptime(lap_time, "%H:%M:%S.%f")
        return lap_time.second + lap_time.minute * 60 + lap_time.microsecond / 1000000

@app.callback(
    Output('lap-time-comparison-chart', 'figure'),
    [Input('driver-dropdown', 'value'),
     Input('year-dropdown', 'value'),
     Input('race-dropdown', 'value')]
)
def update_chart(selected_drivers, year, race):
    session = update_session(year, race)
    result = session.results
    traces = []
    for driver in selected_drivers:
        print("name ", driver)
        if driver in result["FullName"].values:
            driver_code  = result.loc[result["FullName"] == driver, "Abbreviation"].iloc[0]
        lap_times = session.laps['LapTime'][session.laps['Driver'] == driver_code]
        lap_times_seconds = [lap_time_to_seconds(lap_time) for lap_time in lap_times]
        x_values = []
        y_values = []
        for lap_num, lap_time_sec in enumerate(lap_times_seconds, start=1):
            if lap_time_sec is not None:
                x_values.append(lap_num)
                y_values.append(lap_time_sec)
            elif lap_num > 1: 
                x_values.append(lap_num)
                y_values.append(y_values[-1])
        trace = go.Scatter(
            x=x_values,
            y=y_values,
            mode='markers+lines',
            marker=dict(symbol='circle', size=8),
            name=driver,
            hoverinfo='text',
            hovertext=[f'Lap: {lap_num}<br>Lap Time: {str(pd.Timedelta(seconds=lap_time))[2:-3]}<br>Driver: {driver}' 
                       for lap_num, lap_time in zip(x_values, y_values)],
            line=dict(color=fastf1.plotting.driver_color(driver)) 
        )
        traces.append(trace)

    layout = go.Layout(
        xaxis=dict(title='Lap Number'),
        yaxis=dict(title='Lap Time (Seconds)'),
        hovermode='closest',
        showlegend=True,
        margin=dict(l=40, r=40, t=40, b=40)
    )

    return {'data': traces, 'layout': layout}

@app.callback(
    Output('tyre-strategy-plot', 'figure'),
    [Input('year-dropdown', 'value'),
     Input('race-dropdown', 'value')]
)
def update_plot(year, race):
    session = update_session(year , race)
    laps = session.laps
    drivers = session.drivers
    print(drivers)
    drivers = [session.get_driver(driver)["Abbreviation"] for driver in drivers]
    stints = laps[["Driver", "Stint", "Compound", "LapNumber"]]
    stints = stints.groupby(["Driver", "Stint", "Compound"])
    stints = stints.count().reset_index()
    stints = stints.rename(columns={"LapNumber": "StintLength"})
    data = []
    for idx, driver in enumerate(drivers):
        driver_stints = stints.loc[stints["Driver"] == driver]
        previous_stint_end = 0
        for idx, row in driver_stints.iterrows():
            data.append(go.Bar(
                y=[driver],
                x=[row["StintLength"]],
                orientation='h',
                marker=dict(color=fastf1.plotting.COMPOUND_COLORS[row["Compound"]]),
                name=row["Compound"],
                base=previous_stint_end,
                hoverinfo='text',
                hovertext='Driver: {}<br>Tyre Type: {}<br>Laps Used: {}'.format(driver, row["Compound"],
                                                                                  row["StintLength"]),
                width=0.8 
            ))
            previous_stint_end += row["StintLength"]

    layout = go.Layout(
        title="{} {} Grand Prix Strategies".format(year, race),
        xaxis=dict(title="Lap Number", tickfont=dict(family="Arial", size=15, color="#444")),
        yaxis=dict(autorange="reversed", tickfont=dict(family="Arial", size=15, color="#444")),
        showlegend=False,
        margin=dict(l=100, r=20, t=60, b=120), 
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial", size=12, color="#444"),
        hoverlabel=dict(font_size=14, font_family="Arial", font_color="#444"),
        barmode='stack'
    )

    return {'data': data, 'layout': layout}

@app.callback(
    Output('speed-trace-graph', 'figure'),
    [Input('driver-dropdown2', 'value'),
     Input('year-dropdown', 'value'),
     Input('race-dropdown', 'value')]
)
def update_speed_trace(selected_driver_name , year ,race):
    session = update_session(year, race)
    result = session.results
    data = []
    for driver_name in selected_driver_name:
        if driver_name in result["FullName"].values:
            driver_number  = result.loc[result["FullName"] == driver_name, "DriverNumber"].iloc[0]
        selected_lap = session.laps.pick_driver(driver_number).pick_fastest()
        selected_tel = selected_lap.get_car_data().add_distance()
        driver_color = fastf1.plotting.driver_color(driver_name)
        speed_trace = go.Scatter(
            x=selected_tel['Distance'],
            y=selected_tel['Speed'],
            mode='lines',
            name=driver_name,
            line=dict(width=1.9, color=driver_color),
            hoverinfo='text',
            hovertext=[f'<b>Driver:</b> {driver_name}<br><b>Distance:</b> {x:.2f} m<br><b>Speed:</b> {y} km/h'
                       for x, y in zip(selected_tel['Distance'], selected_tel['Speed'])]
        )
        data.append(speed_trace)

    layout = go.Layout(
        xaxis=dict(title='Distance in m'),
        yaxis=dict(title='Speed in km/h'),
        title="Speed Trace for Selected Drivers' Fastest Laps",
        showlegend=True,
    )

    return {'data': data, 'layout': layout}

if __name__ == '__main__':
    app.run_server(debug=True)
