# Hospital Bed Demand Forecasting 🏥

This project combines Poisson distribution with machine learning to predict hospital bed demand. It provides both statistical and ML-based forecasting approaches to help hospitals better prepare for incoming patients.

## Features

- 📊 Poisson distribution modeling for admission predictions
- 🤖 Machine learning enhancement (GBM and XGBoost)
- 📈 Interactive visualizations
- 📱 Web-based dashboard using Streamlit
- 📄 Downloadable prediction reports

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd hospital-bed-forecasting
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Download the dataset:
```bash
python download_data.py
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Open your browser and navigate to `http://localhost:8501`

## Deploy On Render

Use a Web Service and ensure Render starts Streamlit (not `python app.py`).

- Build Command:
```bash
pip install -r requirements.txt
```
- Start Command:
```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

This repository also includes `render.yaml` with the same settings.

## Project Structure

```
hospital-bed-forecasting/
├── src/
│   ├── data_loader.py      # Data loading and preprocessing
│   ├── poisson_model.py    # Poisson distribution implementation
│   └── ml_model.py         # Machine learning model implementation
├── data/                   # Data directory
├── app.py                  # Streamlit dashboard
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

## Models

### 1. Poisson Distribution
- Uses historical admission data to model random arrival patterns
- Provides probability distribution for different admission numbers
- Calculates confidence intervals for predictions

### 2. Machine Learning Enhancement
- Gradient Boosting Machine (GBM)
- XGBoost
- Features include:
  - Day of week
  - Month
  - Weekend indicator
  - Rolling statistics

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hospital admission dataset from Kaggle
- Streamlit for the web interface
- scikit-learn and XGBoost for machine learning models "# Hospita_lBed_Recommendation" 
