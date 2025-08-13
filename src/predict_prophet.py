import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

def clean_period(value):
    """Clean and standardize period to '20XX_s'."""
    if pd.isna(value):
        return None
    value = str(value).lower().strip()
    match = re.search(r"(\d{4})\D*s?(\d)", value)
    if match:
        year, semester = match.groups()
        if semester in ['1', '2']:  # Only allow s1 or s2
            return f"{year}_s{int(semester)}"
    return None

def convert_period_to_date(period):
    """Convert 'YYYY_s1' or 'YYYY_s2' to datetime ('YYYY_06' or 'YYYY_12')."""
    if pd.isna(period):
        return None
    period = period.lower().strip().replace("s1", "06").replace("s2", "12")
    try:
        return pd.to_datetime(period, format="%Y_%m")
    except:
        return None


def prepare_data_for_prophet(df, airport_name):
    """Prepare data for Prophet with 'ds' and 'y' columns."""
    airport_data = df[df['airport'] == airport_name].copy()
    airport_data['period'] = airport_data['period'].apply(clean_period)
    airport_data['date'] = airport_data['period'].apply(convert_period_to_date)
    airport_data = airport_data.dropna(subset=['period', 'date', 'traffic'])
    airport_data = airport_data.sort_values('date')
    prophet_df = pd.DataFrame({'ds': airport_data['date'], 'y': airport_data['traffic']})
    return prophet_df

def add_custom_seasonalities_and_events(model):
    """Add custom seasonalities and holidays."""
    ramadan_dates = pd.DataFrame([
        {'holiday': 'ramadan_impact', 'ds': date, 'lower_window': -15, 'upper_window': 15}
        for date in ['2014-06-28','2015-06-17','2016-06-06','2017-05-26','2018-05-15', '2019-05-06', '2020-04-24', '2021-04-13',
                     '2022-04-02', '2023-03-22', '2024-03-11', '2025-02-28', '2026-02-18']
    ])
    hajj_dates = pd.DataFrame([
        {'holiday': 'hajj_impact', 'ds': date, 'lower_window': -5, 'upper_window': 5}
        for date in ['2014-10-02','2015-09-21','2016-09-10','2017-08-30','2018-08-19', '2019-08-09', '2020-07-28', '2021-07-17',
                     '2022-07-07', '2023-06-26', '2024-06-14', '2025-06-04', '2026-05-29']
    ])
    covid_dates = pd.DataFrame([
        {'holiday': 'covid_lockdown', 'ds': '2020-03-20', 'lower_window': -10, 'upper_window': 180},
        {'holiday': 'covid_recovery', 'ds': '2021-06-01', 'lower_window': -30, 'upper_window': 365},
    ])
    summer_dates = pd.DataFrame([
        {'holiday': 'summer_peak', 'ds': f'{year}-07-15', 'lower_window': -30, 'upper_window': 30}
        for year in range(2014, 2027)
    ])
    
    holidays = pd.concat([ramadan_dates, hajj_dates, covid_dates, summer_dates])
    holidays['ds'] = pd.to_datetime(holidays['ds'])
    return holidays

def create_prophet_model(df_prophet, airport_name):
    """Initialize, configure and fit Prophet model."""
    model = Prophet(
        growth='linear',
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.1,
        seasonality_prior_scale=2,
        n_changepoints=30,
        interval_width=0.95
    )
    holidays = add_custom_seasonalities_and_events(model)
    model.holidays = holidays
    print(f"Training Prophet model for {airport_name}...")
    model.fit(df_prophet)
    return model

def make_forecast(model, periods=4):
    """Forecast next N semesters (periods=4 => 2 years)."""
    future = model.make_future_dataframe(periods=periods, freq='6MS')
    forecast = model.predict(future)
    forecast['ds'] = pd.to_datetime(forecast['ds'])
    return forecast, future

def evaluate_model(df_prophet, forecast, train_size=0.8):
    """Evaluate forecast using train/test split."""
    split_idx = int(len(df_prophet) * train_size)
    train_data = df_prophet.iloc[:split_idx]
    test_data = df_prophet.iloc[split_idx:]

    if len(test_data) == 0:
        print("Not enough data for evaluation")
        return None

    model_eval = Prophet(
        growth='linear',
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='additive',
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10
    )
    model_eval.fit(train_data)
    future_eval = model_eval.make_future_dataframe(periods=len(test_data), freq='6MS')
    forecast_eval = model_eval.predict(future_eval)

    test_predictions = forecast_eval.iloc[-len(test_data):]['yhat'].values
    test_actual = test_data['y'].values

    mae = mean_absolute_error(test_actual, test_predictions)
    mse = mean_squared_error(test_actual, test_predictions)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((test_actual - test_predictions) / test_actual)) * 100

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'test_actual': test_actual,
        'test_predictions': test_predictions
    }

def plot_forecast(model, forecast, df_prophet, airport_name):
    """Plot forecast, trend, seasonality, and residuals."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f'{airport_name} - Prophet Forecast Analysis', fontsize=16, fontweight='bold')

    ax1 = axes[0, 0]
    model.plot(forecast, ax=ax1)
    ax1.set_title('Traffic Forecast with Confidence Intervals')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Traffic (Passengers)')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    trend = forecast[['ds', 'trend']].set_index('ds')
    trend.plot(ax=ax2, color='red', linewidth=2)
    ax2.set_title('Trend Component')
    ax2.set_ylabel('Trend')
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    if 'yearly' in forecast.columns:
        yearly = forecast[['ds', 'yearly']].set_index('ds')
        yearly.plot(ax=ax3, color='green', linewidth=2)
    ax3.set_title('Yearly Seasonality')
    ax3.set_ylabel('Seasonal Effect')
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    historical_forecast = forecast[forecast['ds'] <= df_prophet['ds'].max()]
    residuals = df_prophet.set_index('ds')['y'] - historical_forecast.set_index('ds')['yhat']
    residuals.plot(ax=ax4, color='purple', marker='o')
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax4.set_title('Residuals (Actual - Predicted)')
    ax4.set_ylabel('Residuals')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

def plot_components(model, forecast):
    """Plot Prophet internal seasonalities."""
    fig = model.plot_components(forecast, figsize=(12, 8))
    plt.tight_layout()

def append_forecast_summary(all_summary_df, forecast, airport_name, last_train_date):
    """
    Appends forecast and growth metrics for one airport to an existing DataFrame.
    """
    future_forecast = forecast[forecast['ds'] > last_train_date].copy()
    future_forecast['ds'] = pd.to_datetime(future_forecast['ds'])

    if len(future_forecast) == 0:
        return all_summary_df  # Nothing to append

    # Add period column (S1/S2)
    future_forecast['period'] = future_forecast['ds'].apply(lambda ds: 'S1' if ds.month == 6 else 'S2')

    # Compute growth metrics
    first_forecast = future_forecast.iloc[0]['yhat']
    last_forecast = future_forecast.iloc[-1]['yhat']
    total_growth = ((last_forecast - first_forecast) / first_forecast) * 100

    min_ds = future_forecast['ds'].min()
    max_ds = future_forecast['ds'].max()
    years = (max_ds - min_ds).days / 365.25
    annual_growth = (((last_forecast / first_forecast) ** (1 / years)) - 1) * 100 if years > 0 else 0

    # Prepare DataFrame to append
    future_forecast['Airport'] = airport_name
    future_forecast['First_Forecast'] = first_forecast
    future_forecast['Last_Forecast'] = last_forecast
    future_forecast['Total_Growth_%'] = total_growth
    future_forecast['Annual_Growth_%'] = annual_growth

    # Select relevant columns
    cols = ['Airport', 'ds', 'period', 'yhat', 'yhat_lower', 'yhat_upper', 'First_Forecast', 'Last_Forecast', 'Total_Growth_%', 'Annual_Growth_%']
    future_forecast = future_forecast[cols]

    # Append to existing DataFrame
    all_summary_df = pd.concat([all_summary_df, future_forecast], ignore_index=True)
    return all_summary_df

# ----------------------------
# Main pipeline
# ----------------------------

def run_prophet():

    # Load data
    df = pd.read_csv("./cleaned_data/cleaned_trafic_ma_long.csv")

    airports = ['agadir', 'casablanca', 'fes_saiss', 'marrakech', 'rabat_sale']
    all_forecasts = {}
    all_evaluations = {}
    all_airports_summary = pd.DataFrame()

    for airport in airports:
        try:
            df_prophet = prepare_data_for_prophet(df, airport)
            if len(df_prophet) < 4:
                continue

            model = create_prophet_model(df_prophet, airport)
            forecast, _ = make_forecast(model, periods=4)

            all_forecasts[airport] = {
                'model': model,
                'forecast': forecast,
                'data': df_prophet
            }

            if len(df_prophet) >= 6:
                evaluation = evaluate_model(df_prophet, forecast)
                if evaluation:
                    all_evaluations[airport] = evaluation
            all_airports_summary = append_forecast_summary(all_airports_summary, forecast, airport, df_prophet['ds'].max())
            
        except Exception as e:
            with open("./docs/outputs/errors.log", "a") as log_file:
                log_file.write(f"Error processing {airport}: {str(e)}\n")
            continue
        
        all_airports_summary.to_csv("./docs/outputs/forecast_summary.csv")
        
    # Save combined final forecast plot
    if len(all_forecasts) > 1:
        plt.figure(figsize=(15, 10))
        colors = plt.cm.Set3(np.linspace(0, 1, len(all_forecasts)))
        for i, (airport, data) in enumerate(all_forecasts.items()):
            forecast = data['forecast']
            df_prophet = data['data']
            plt.plot(df_prophet['ds'], df_prophet['y'], 'o-', color=colors[i], label=f'{airport} (Historical)', alpha=0.7)
            future_data = forecast[forecast['ds'] > df_prophet['ds'].max()]
            plt.plot(future_data['ds'], future_data['yhat'], '--', color=colors[i], label=f'{airport} (Forecast)', linewidth=2)
            plt.fill_between(future_data['ds'], future_data['yhat_lower'], future_data['yhat_upper'], alpha=0.2, color=colors[i])
        plt.title('All Airports - Traffic Forecast Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Traffic (Passengers)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("./docs/outputs/final_forecast.png", dpi=300, bbox_inches="tight")
        plt.close()

    # Save evaluation metrics to CSV
    if all_evaluations:
        eval_df = pd.DataFrame(all_evaluations).T
        eval_df.to_csv("./docs/outputs/model_metrics.csv", index=True)

    print("Forecasting complete. Results saved in 'outputs/' folder.")
