import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from utils.data_loader import load_all_data

# Set page config to use wide layout
st.set_page_config(
    layout="wide"
)

# --- Shared Data Loading (refactor as needed) ---
@st.cache_data
def read_data(path):
    return pd.read_csv(path)

# --------------------------------------------
# Load Data
# --------------------------------------------
data = load_all_data()
forecasts_df = data['forecasts']
historic_df = data['historic']
combined_df = data['legacy']
deposits_sector_df = data['sector']

# Parse dates
historic_df['date'] = pd.to_datetime(historic_df['date'])
forecasts_df['date'] = pd.to_datetime(forecasts_df['date'])
combined_df['date'] = pd.to_datetime(combined_df['date'])
deposits_sector_df['date'] = pd.to_datetime(deposits_sector_df['date'])

# Filter to recent months
# Add user controls
st.sidebar.markdown("### Display Options")

# Checkbox to show all data
show_all = st.sidebar.checkbox("Show all data", value=False)

if show_all:
    filtered_combined = combined_df.copy()
    filtered_history = historic_df.copy()
    filtered_sectors = deposits_sector_df.copy()
else:
    months_slider = st.sidebar.slider(
        "Select how many months of data to show",
        min_value=12,
        max_value=len(historic_df),
        value=24,
        step=1
    )
    cutoff_date = pd.to_datetime("today") - pd.DateOffset(months=months_slider)
    filtered_combined = combined_df[combined_df['date'] >= cutoff_date]
    filtered_history = historic_df[historic_df['date'] >= cutoff_date]
    filtered_sectors = deposits_sector_df[deposits_sector_df['date'] >= cutoff_date]


# UI
st.title("Overlay Other Variables for Exploration")

st.markdown("*Note: The majority of the variables overlayed are not used for the model forecasting but displayed only for exploration purposes.*")
variables = [x for x in filtered_combined.columns if x not in ['date', 'household_deposits']]

feat = st.selectbox("Select variable to overlay", options=variables, placeholder="Choose a variable", key='columns')
deposits_type_options = ["Transaction", "Savings", "Term deposit"]
selected_deposits_type = st.multiselect(
    "Overlay historical deposits by sector:",
    options=deposits_type_options
)

# Chart
fig = go.Figure()

# Historic deposits
fig.add_trace(go.Scatter(
    x=filtered_history['date'],
    y=filtered_history['household_deposits'],
    mode='lines+markers',
    name='Historic',
    line=dict(color='#4f5d75'),
    marker=dict(size=8),
    hovertemplate="Date: %{x|%b %Y}<br>Value: $%{y:,.0f} NZDm<extra>%{fullData.name}</extra>"
))

# Overlay selected variable
fig.add_trace(go.Scatter(
    x=filtered_combined['date'],
    y=filtered_combined[feat],
    name=feat.replace('_', ' ').title(),
    yaxis='y2',
    mode='lines',
    line=dict(dash='dot'),
    hovertemplate=f"Date: %{{x|%b %Y}}<br>{feat.replace('_', ' ').title()}: %{{y:,.0f}}<extra></extra>"
))

# Deposits by sector
for deposit_type in selected_deposits_type:
    col_name = f'{deposit_type} balances'
    if col_name in filtered_sectors.columns:
        fig.add_trace(go.Scatter(
            x=filtered_sectors['date'],
            y=filtered_sectors[col_name],
            mode='lines+markers',
            name=f"{deposit_type} Deposits",
            line=dict(dash='dot'),
            marker=dict(size=8),
            hovertemplate=f"Date: %{{x|%b %Y}}<br>{deposit_type}: $%{{y:,.0f}} NZDm<extra></extra>"
        ))

# Layout
fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Deposits ($NZDm)',
    yaxis2=dict(
        title=feat.replace('_', ' ').title(),
        overlaying='y',
        side='right',
        showgrid=False,
        autorange=True
    ),
    template='plotly_white',
    height=450,
    plot_bgcolor='#f9f9f9',
    paper_bgcolor='#f2f4f3',
    margin=dict(l=40, r=40, t=40, b=40),
    legend=dict(orientation='h', y=1.15, x=0.0, xanchor='left'),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False)
)

st.plotly_chart(fig, use_container_width=True)

# Data table 
filtered_data = filtered_combined.copy()
filtered_data['date'] = filtered_data['date'].dt.strftime('%Y-%m')
st.dataframe(filtered_data)
