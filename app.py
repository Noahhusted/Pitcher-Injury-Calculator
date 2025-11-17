import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import numpy as np
import pickle

# Load trained model and scaler
with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

input_ranges = {
    "p_formatted_ip": "0 - 300",
    "ip_py": "0 - 300",
    "fastball_avg_spin": "1500 - 3000 rpm",
    "offspeed_avg_break": "0 - 20 inches",
    "fastball_avg_break_z_induced": "0 - 30 inches",
    "fastball_avg_speed": "80 - 105 mph",
    "breaking_avg_speed": "70 - 95 mph",
    "arm_angle": "-75 - 90 degrees",
    "offspeed_avg_speed": "60 - 85 mph",
}

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Injury Risk Calculator", className="text-center text-primary mb-4"), width=12)
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Label(f"Formatted Innings Pitched (Range: {input_ranges['p_formatted_ip']})"),
            dbc.Input(id="p_formatted_ip", type="number", placeholder="Enter formatted innings pitched", value=100),

            dbc.Label(f"Innings Pitched Previous Year (Range: {input_ranges['ip_py']})"),
            dbc.Input(id="ip_py", type="number", placeholder="Enter previous year's innings pitched", value=80),

            dbc.Label(f"Fastball Avg Spin Rate (Range: {input_ranges['fastball_avg_spin']})"),
            dbc.Input(id="fastball_avg_spin", type="number", placeholder="Enter fastball spin rate (rpm)", value=2200),

            dbc.Label(f"Offspeed Avg Break (Range: {input_ranges['offspeed_avg_break']})"),
            dbc.Input(id="offspeed_avg_break", type="number", placeholder="Enter offspeed break (inches)", value=12),

            dbc.Label(f"Fastball Avg Break Z-Induced (Range: {input_ranges['fastball_avg_break_z_induced']})"),
            dbc.Input(id="fastball_avg_break_z_induced", type="number", placeholder="Enter z-induced fastball break", value=15),
        ], width=6),
        dbc.Col([
            dbc.Label(f"Fastball Avg Speed (Range: {input_ranges['fastball_avg_speed']})"),
            dbc.Input(id="fastball_avg_speed", type="number", placeholder="Enter fastball speed (mph)", value=95),

            dbc.Label(f"Breaking Avg Speed (Range: {input_ranges['breaking_avg_speed']})"),
            dbc.Input(id="breaking_avg_speed", type="number", placeholder="Enter breaking speed (mph)", value=85),

            dbc.Label(f"Arm Angle (Range: {input_ranges['arm_angle']})"),
            dbc.Input(id="arm_angle", type="number", placeholder="Enter arm angle (degrees)", value=50),

            dbc.Label(f"Offspeed Avg Speed (Range: {input_ranges['offspeed_avg_speed']})"),
            dbc.Input(id="offspeed_avg_speed", type="number", placeholder="Enter offspeed speed (mph)", value=80),

            html.Br(),
            dbc.Button("Calculate Risk", id="calculate-btn", color="primary", className="mt-2"),
        ], width=6),
    ]),
    dbc.Row([
        dbc.Col(html.H3(id="risk-output", className="text-center mt-4"))
    ])
])

@app.callback(
    Output("risk-output", "children"),
    Input("calculate-btn", "n_clicks"),
    [
        Input("p_formatted_ip", "value"),
        Input("ip_py", "value"),
        Input("fastball_avg_spin", "value"),
        Input("offspeed_avg_break", "value"),
        Input("fastball_avg_break_z_induced", "value"),
        Input("fastball_avg_speed", "value"),
        Input("breaking_avg_speed", "value"),
        Input("arm_angle", "value"),
        Input("offspeed_avg_speed", "value"),
    ]
)
def calculate_risk(n_clicks, p_ip, ip_py, spin, offspeed_break, break_z,
                   fastball_speed, breaking_speed, arm_angle, offspeed_speed):
    if not n_clicks:
        return "Enter values and click Calculate Risk."

    features = np.array([[
        p_ip,
        ip_py,
        spin,
        offspeed_break,
        break_z,
        fastball_speed,
        breaking_speed,
        arm_angle,
        offspeed_speed
    ]])

    try:
        features_scaled = scaler.transform(features)
        risk = model.predict_proba(features_scaled)[0][1] * 100
        return f"Injury Risk: {risk:.2f}%"
    except Exception as e:
        return f"Error: {str(e)}"

server = app.server  # for gunicorn / Render

if __name__ == "__main__":
    app.run(debug=True)