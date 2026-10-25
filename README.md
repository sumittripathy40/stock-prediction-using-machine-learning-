# Tesla Stock Price Forecasting with Prophet

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sumittripathy40/stock-prediction-using-machine-learning-/blob/main/Learner_Notebook_prophet.ipynb)

A Jupyter notebook that explores historical Tesla (TSLA) prices and uses Facebook Prophet to forecast the closing price 30 days beyond the available data.

## What the notebook does

- Loads and summarizes the included TSLA historical stock data.
- Visualizes closing prices and trading volume with Plotly Express, including line, area, bar, and box charts.
- Prepares the data for Prophet by mapping `Date` to `ds` and `Close` to the target `y`.
- Fits Prophet with its default model configuration and generates a 30-day forecast.
- Plots the forecast and Prophet components, then exports the forecast as `forecast.csv` from Google Colab.

## Repository files

| File | Description |
| --- | --- |
| [`Learner_Notebook_prophet.ipynb`](Learner_Notebook_prophet.ipynb) | Notebook for data exploration, visualization, forecasting, and export. |
| [`TSLA(12).csv`](https://github.com/sumittripathy40/stock-prediction-using-machine-learning-/blob/main/TSLA%2812%29.csv) | Historical daily TSLA data: 252 rows dated July 12, 2021 through July 11, 2022. Columns: `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, and `Volume`. |

The forecast uses only the `Date` and `Close` columns.

## Setup and run

1. Open `Learner_Notebook_prophet.ipynb` in Google Colab using the badge above.
2. Make the CSV available to the notebook as `TSLA.csv`. The file in this repository is named `TSLA(12).csv`, so either rename it after uploading or change the notebook's `read_csv` line to:
   ```python
   df = pd.read_csv("TSLA(12).csv")
   ```
3. Use a Python environment with `pandas`, `plotly`, and the notebook's `fbprophet` import available. The repository does not include pinned dependency versions.
4. Run the notebook cells in order. In Colab, the final cell downloads the generated `forecast.csv`.

## Notes

- The included CSV is a static historical sample; the notebook does not fetch live market data.
- The notebook contains saved output from a different date range than the checked-in CSV. Rerun the cells with the included data to refresh the displayed results.
- The notebook does not include a train/test split, backtest, or forecast accuracy metrics.
