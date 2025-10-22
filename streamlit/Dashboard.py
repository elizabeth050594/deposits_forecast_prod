import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from utils.data_loader import load_all_data

# --------------------------------------------
# Page Setup
# --------------------------------------------
st.set_page_config(
    page_title="NZ National Household Deposits Forecast",
    layout="wide"
)

# Title Styling
st.markdown("""
    <style>
    @import url("https://fonts.googleapis.com/css2?family=Dosis:wght@200..800&family=Inconsolata:wdth,wght@50..200,200..900&display=swap");

    .custom-title {
        font-family: 'Inconsolata', monospace;
        font-size: 40px;
        font-weight: 500;
        color: #333333;
        margin-bottom: 20px;
    }
    </style>
    <div class="custom-title">NZ National Household Deposits Forecast</div>
""", unsafe_allow_html=True)

# Metric Tooltip Styling
st.markdown("""
<style>
.tooltip {
  position: relative;
}

.tooltip .tooltiptext {
  visibility: hidden;
  width: 200px;
  background-color: #333;
  color: #fff;
  text-align: center;
  padding: 6px;
  border-radius: 6px;
  position: absolute;
  z-index: 1;
  left: 50%;
  margin-left: -100px;
  opacity: 0;
  transition: opacity 0.3s;
  font-size: 0.8rem;
}

.tooltip:hover .tooltiptext {
  visibility: visible;
  opacity: 1;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------
# Helper Functions
# --------------------------------------------
def format_dollar_with_sign(value):
    sign = "+" if value > 0 else ("−" if value < 0 else "")
    return f"{sign}${abs(value):,.0f}"

def format_pct_with_sign(value):
    sign = "+" if value > 0 else ("−" if value < 0 else "")
    return f"{sign}{abs(value):.1f}%"

# Map for forecast horizon
horizon_mapping = {
    "Current-month": 2,
    "3-month": 5,
    "6-month": 8,
    "12-month": 14
}
inverse_horizon = {1: 'Previous', 2: 'Current', 5: '3', 8: '6', 14: '12'}
# --------------------------------------------
# Load Data
# --------------------------------------------
data = load_all_data()
forecasts_df = data['forecasts']
historic_df = data['historic']
val_dict = data['val_dict']
val_metric = data['val_metric']
combined_df = data['historic'].copy()

# --------------------------------------------
# Preprocessing
# --------------------------------------------
forecasts_df['date'] = pd.to_datetime(forecasts_df['date'])
forecasts_df['formatted_date'] = forecasts_df['date'].dt.strftime('%b %Y')
historic_df['date'] = pd.to_datetime(historic_df['date'])
combined_df['date'] = pd.to_datetime(combined_df['date'])

latest_date = historic_df['date'].iloc[-1]
latest_date_str = latest_date.strftime('%b %Y')
latest_value = historic_df['household_deposits'].iloc[-1]

# Forecast comparisons
current_month_forecast = forecasts_df.loc[forecasts_df['horizon'] == 2, 'forecast'].values[0]
twelve_month_forecast = forecasts_df.loc[forecasts_df['horizon'] == 14, 'forecast'].values[0]

diff_1m = current_month_forecast - latest_value
pct_1m = (diff_1m / latest_value) * 100
diff_12m = twelve_month_forecast - latest_value
pct_12m = (diff_12m / latest_value) * 100

forecast_month_2 = forecasts_df.loc[forecasts_df['horizon'] == 2, 'formatted_date'].values[0]
forecast_month_14 = forecasts_df.loc[forecasts_df['horizon'] == 14, 'formatted_date'].values[0]

# --------------------------------------------
# Summary Text
# --------------------------------------------
summary_text = f"""
Based on the latest forecast, household deposits are expected to **{'increase' if diff_1m > 0 else 'decrease'}** by approximately **\${abs(diff_1m):,.0f} NZDm ({abs(pct_1m):.1f}%)** in {forecast_month_2}, compared to the latest available value of \${latest_value:,.0f} NZDm in {latest_date_str}. 

Looking 12 months ahead to {forecast_month_14}, deposits are forecasted to **{'rise' if diff_12m > 0 else 'fall'}** by approximately **\${abs(diff_12m):,.0f} NZDm ({abs(pct_12m):.1f}%)**.
"""
st.markdown(summary_text)

# --------------------------------------------
# Metric Cards
# --------------------------------------------
metrics_forecasts_df = forecasts_df[forecasts_df['horizon'] != 1].reset_index(drop=True)
cols = st.columns(len(metrics_forecasts_df) + 1, gap="small")

# Latest value card
with cols[0]:
    st.markdown(f"""
    <div class="tooltip">
        <div style="border:2px solid #555; border-radius:12px; padding:1rem; background-color:#463f3a; text-align:center; color:#f2f4f3;"> 
        <div style="font-size:1.1rem;">{latest_date_str}</div> 
        <div style="font-size:1.8rem; font-weight:700;">${latest_value:,.0f}</div> 
        <div style="font-size:1rem; font-weight:600;">Latest Data</div> <div style="font-size:0.85rem;">(Sourced from RBNZ)</div>
        <span class="tooltiptext">Latest household deposits value from RBNZ</span>
    </div>
    """, unsafe_allow_html=True)


# Forecast cards
for col, (_, row) in zip(cols[1:], metrics_forecasts_df.iterrows()):
    label = row['formatted_date']
    forecast_value = row['forecast']
    dollar_diff = forecast_value - latest_value
    pct_change = (dollar_diff / latest_value) * 100

    text_color = "#439775" if dollar_diff >= 0 else "#E53935"

    card_html = f"""
    <div class="tooltip">
        <div style="border:2px solid #463f3a; border-radius:12px; padding:1rem; background-color:#ffffff; text-align:center;">
            <div style="font-size:1.1rem;">{label}</div>
            <div style="font-size:1.8rem; font-weight:700;">${forecast_value:,.0f}</div>
            <div style="margin-top:0.4rem; font-size:1rem; color:{text_color}; font-weight:600;">
                {format_dollar_with_sign(dollar_diff)}
            </div>
            <div style="font-size:0.85rem; color:{text_color}; font-weight:500;">
                ({format_pct_with_sign(pct_change)})
            </div>
        </div>
        <span class="tooltiptext">{inverse_horizon[row['horizon']]}-month model forecast, compared against lastest actual data in {latest_date_str}.</span>
    </div>
    """

    col.markdown(card_html, unsafe_allow_html=True)

# --------------------------------------------
# Chart Section
# --------------------------------------------
st.empty()
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### History & Forecast")
        months_of_history = st.slider(
            "Select how many months of history to display",
            min_value=12,
            max_value=70,
            value=24,
            step=1
        )
    with col2:
        st.write("")
        st.write("")
        st.write("")
        options = ["Current-month", "3-month", "6-month", "12-month"]
        selected_horizon = st.pills(
            "Overlay historical forecasts for each model (validation data):",
            options=options,
            selection_mode="single",
            width='stretch'
        )

    cutoff_date = pd.to_datetime("today") - pd.DateOffset(months=months_of_history)
    filtered_history = historic_df[historic_df['date'] >= cutoff_date]
    filtered_forecasts = forecasts_df[forecasts_df['date'] > filtered_history['date'].max()]
    filtered_combined = combined_df[combined_df['date'] >= cutoff_date]

    fig = go.Figure()

    # Add historic data
    fig.add_trace(go.Scatter(
        x=filtered_history['date'],
        y=filtered_history['household_deposits'],
        mode='lines+markers',
        name="Actual Historic Values",
        line=dict(color='#4f5d75'),
        marker=dict(size=8),
        hovertemplate=(
             f"<span style='color:#4f5d75'><b>Actual Historic Value</b></span><br>"
             "Date: %{x|%b %Y}<br>"
             "Value: $%{y:,.0f} NZDm"
             "<extra></extra>"),
        hoverlabel=dict(
            bordercolor='#4f5d75',
            bgcolor='white',  
            font_color='grey',  
            )
    ))
    

    # Add validation forecasts if selected
    if selected_horizon:
        h_val = horizon_mapping[selected_horizon]
        filtered_forecasts = forecasts_df[
            (forecasts_df['horizon'] == h_val) &
            (forecasts_df['date'] > filtered_history['date'].max())
        ].copy()
        filtered_forecasts['adjusted_horizon'] = inverse_horizon[h_val]

        try:
            val_df = val_dict[h_val]
            val_df['date'] = pd.to_datetime(val_df['date'])
            val_df['diff'] = val_df['y_pred'] - val_df['y_true']
            val_df['diff_label'] = val_df['diff'].apply(lambda x: f"+{x:,.0f}" if x >= 0 else f"-{abs(x):,.0f}")

            metrics = val_metric[val_metric['model_name'] == f"model_forecast_h{h_val}"]
            if not metrics.empty:
                rmse = metrics['rmse'].values[0]
                mape = metrics['mape'].values[0]
                max_err = metrics['max_error'].values[0]
            else:
                rmse = mape = max_err = None

            name = f"{selected_horizon} Model Past Forecasts"
            fig.add_trace(go.Scatter(
                x=val_df['date'],
                y=val_df['y_pred'],
                mode='lines+markers',
                name = name,
                line=dict(color="#ef8354", dash='dot'),
                marker=dict(size=7, symbol='diamond'),
                customdata=val_df[['diff_label']],
                hovertemplate=(
                    f"<span style='color:#ef8354'><b>{name}</b></span><br>"
                    "Date: %{x|%b %Y}<br>"
                    "Predicted: $%{y:,.0f}<br>"
                    "<b>Diff: %{customdata[0]} NZDm</b>"
                    "<extra></extra>"),
                hoverlabel=dict(
                bordercolor='#ef8354',
                bgcolor='white',  
                font_color='grey',  
                )
            ))

        
            for _, row in val_df.iterrows():
                fig.add_annotation(
                    x=row['date'],
                    y=row['y_pred'],
                    text=row['diff_label'],
                    showarrow=False,
                    yshift=12,  # moves text above the point
                    font=dict(size=11,color='rgba(128, 128, 128, 0.5)'),
                    align='center'
                )

        except Exception as e:
            st.warning(f"Could not load validation data: {e}")

    # Add forecast line
    fig.add_trace(go.Scatter(
        x=filtered_forecasts['date'],
        y=filtered_forecasts['forecast'],
        mode='lines+markers',
        name='Forecasted Values',
        line=dict(color="#ef8354", dash='dash'),
        marker=dict(size=10),
        hovertemplate=
       ("<span style='color:#ef8354'><b>Forecast</b></span><br>"             
        "Date: %{x|%b %Y}<br>"
        "Value: $%{y:,.0f} NZDm<br>"
        "<extra></extra>"),
        hoverlabel=dict(
        bordercolor='#ef8354',
        bgcolor='white',  
        font_color='grey',  

    )
    ))

    # Add annotations for each forecast point
    for _, row in filtered_forecasts.iterrows():
        horizon_label = inverse_horizon.get(row['horizon'], '')
        st_yshift = 12 if row['horizon'] != 1 else -12
        fig.add_annotation(
            x=row['date'],
            y=row['forecast'],
            text=f"{horizon_label}-month",
            showarrow=False,
            yshift=st_yshift,
            font=dict(size=11, color='rgba(128, 128, 128, 0.5)')
        )

    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Deposits ($NZDm)',
        height=400,
        plot_bgcolor='#f9f9f9',
        paper_bgcolor='#f2f4f3',
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation='h', y=1.15, x=0.0),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        hovermode = 'x'
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("*Note: the 1-month forecast is derived using the 2-month model with 1 fewer data point.*")

# --------------------------------------------
# Download Forecast Table
# --------------------------------------------
with st.expander("Expand to download forecasts"):
    inverse_h = {1:'Previous', 2:'Current', 5:'3', 8:'6', 14:'12'}
    export_df = forecasts_df[['date', 'horizon', 'forecast']].copy()
    export_df['horizon'] = export_df['horizon'].map(inverse_h)
    export_df['date'] = export_df['date'].dt.strftime('%Y-%m')

    df = export_df.rename(columns={
        'forecast': 'Deposits Forecast (NZDM)',
        'horizon': 'Forecast Horizon (month)'
    }).set_index('Forecast Horizon (month)')

    st.dataframe(df.style.format({'Deposits Forecast (NZDM)': "${:,.0f}"}))
