"""Streamlit Documentation page with API examples and guides.

This page provides comprehensive documentation for using the API, understanding the model, and
deploying the system.
"""

import streamlit as st

st.set_page_config(page_title="Documentation", page_icon="📚", layout="wide")

st.title("📚 Documentation")

st.markdown("Comprehensive guides and API documentation for the Rossmann Forecasting Platform.")

# Tab layout for different documentation sections
tab1, tab2, tab3, tab4 = st.tabs(["API Reference", "Model Architecture", "Features", "Deployment"])

with tab1:
    st.header("🔌 API Reference")

    st.markdown(
        """
        The FastAPI server provides REST endpoints for model predictions and management.

        **Base URL**: `http://localhost:8000`
        """
    )

    # Health Check
    st.subheader("Health Check")
    st.code(
        """
# GET /health
curl http://localhost:8000/health
        """,
        language="bash",
    )

    st.code(
        """
{
  "status": "healthy",
  "timestamp": "2025-01-19T10:30:00",
  "model_loaded": true,
  "model_version": "7"
}
        """,
        language="json",
    )

    st.divider()

    # Single Prediction
    st.subheader("Single Prediction")
    st.code(
        """
# POST /predict
curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "features": [{
      "Store": 1,
      "DayOfWeek": 1,
      "Open": 1,
      "Promo": 0,
      "StateHoliday": "0",
      "SchoolHoliday": 0,
      "StoreType": "a",
      "Assortment": "a",
      "CompetitionDistance": 1000.0,
      "CompetitionOpenSinceMonth": 1,
      "CompetitionOpenSinceYear": 2015,
      "Promo2": 0,
      "Promo2SinceWeek": null,
      "Promo2SinceYear": null,
      "PromoInterval": null,
      "Year": 2015,
      "Month": 7,
      "Week": 28,
      "Day": 1
    }],
    "model_stage": "Production"
  }'
        """,
        language="bash",
    )

    st.code(
        """
{
  "predictions": [6234.52],
  "model_version": "7",
  "timestamp": "2025-01-19T10:35:00",
  "count": 1
}
        """,
        language="json",
    )

    st.divider()

    # Batch Prediction
    st.subheader("Batch Prediction")
    st.markdown("Send multiple feature sets in a single request:")

    st.code(
        """
{
  "features": [
    {...},  # First prediction
    {...},  # Second prediction
    {...}   # Third prediction
  ],
  "model_stage": "Production"
}
        """,
        language="json",
    )

    st.divider()

    # Python Client Example
    st.subheader("Python Client Example")
    st.code(
        """
import requests
import pandas as pd

# API endpoint
API_URL = "http://localhost:8000"

# Prepare features
features = {
    "Store": 1,
    "DayOfWeek": 1,
    "Open": 1,
    "Promo": 0,
    "StateHoliday": "0",
    "SchoolHoliday": 0,
    "StoreType": "a",
    "Assortment": "a",
    "CompetitionDistance": 1000.0,
    "CompetitionOpenSinceMonth": 1,
    "CompetitionOpenSinceYear": 2015,
    "Promo2": 0,
    "Promo2SinceWeek": None,
    "Promo2SinceYear": None,
    "PromoInterval": None,
    "Year": 2015,
    "Month": 7,
    "Week": 28,
    "Day": 1
}

# Make prediction request
response = requests.post(
    f"{API_URL}/predict",
    json={"features": [features], "model_stage": "Production"}
)

# Get prediction
result = response.json()
prediction = result["predictions"][0]
print(f"Predicted Sales: ${prediction:,.2f}")
        """,
        language="python",
    )

with tab2:
    st.header("🏗️ Model Architecture")

    st.markdown(
        """
        The Rossmann forecasting system uses **one LightGBM model** (gradient-boosted
        decision trees) with fixed hyperparameters from `config/params.yaml`. Keeping the
        model simple keeps the focus on the system around it.
        """
    )

    st.subheader("Training Pipeline")
    st.code(
        """
1. Data Loading
   ↓
2. Feature Engineering (12 features)
   ↓
3. Train/Holdout Split (last 6 weeks held out)
   ↓
4. LightGBM Training (fixed hyperparameters)
   ↓
5. Model Registration to MLflow
   ↓
6. Validation & Promotion
        """,
        language="text",
    )

    st.subheader("Performance Metrics")
    st.markdown(
        """
        **Primary Metric**: RMSPE (Root Mean Square Percentage Error)

        - **Promotion threshold**: RMSPE at or below 0.20 (`validation.max_rmspe`)
        - **Current model**: about 0.16 on the 6-week holdout

        **Validation Strategy**:
        - Time-based train/holdout split (6-week holdout)
        - No data leakage from future to past
        - No features computed from past sales, so serving matches training
        """
    )

with tab3:
    st.header("📊 Feature Engineering")

    st.markdown(
        """
        The model uses 12 features in four groups (listed in `config/params.yaml`):
        """
    )

    feature_col1, feature_col2 = st.columns(2)

    with feature_col1:
        st.markdown(
            """
            **When**
            - DayOfWeek (1=Monday, 7=Sunday)
            - Month
            - Week (of year)
            - Year

            **Promotions**
            - Promo (promotion that day)
            - Promo2 (store is in the ongoing coupon program)
            - Promo2Active (coupon program running this month)
            """
        )

    with feature_col2:
        st.markdown(
            """
            **Store**
            - StoreType (a, b, c, d)
            - Assortment (a, b, c)
            - CompetitionDistance (meters to nearest competitor)

            **Holidays**
            - StateHoliday (0, a, b, c)
            - SchoolHoliday (0, 1)
            """
        )

    st.divider()

    st.subheader("Feature Requirements")
    st.markdown(
        """
        All predictions require these features:

        | Feature | Type | Range | Description |
        |---------|------|-------|-------------|
        | Store | int | 100 store IDs | One of the stores in data/raw/store.csv |
        | DayOfWeek | int | 1-7 | Day of week (1=Monday) |
        | Open | int | 0-1 | Store open flag |
        | Promo | int | 0-1 | Daily promotion active |
        | StateHoliday | str | 0,a,b,c | State holiday type |
        | SchoolHoliday | int | 0-1 | School holiday flag |
        | StoreType | str | a,b,c,d | Store format |
        | Assortment | str | a,b,c | Product assortment |
        | Year | int | — | Prediction year |
        | Month | int | 1-12 | Prediction month |
        | Week | int | 1-53 | Week of year |
        | Day | int | 1-31 | Day of month |

        **Optional** (can be null): CompetitionDistance, CompetitionOpenSinceMonth/Year,
        Promo2SinceWeek/Year, PromoInterval
        """
    )

with tab4:
    st.header("🚀 Deployment Guide")

    st.subheader("Prerequisites")
    st.code(
        """
# Install dependencies
uv pip install -e ".[dev]"

# Ensure MLflow tracking server is running
mlflow ui --port 5000

# Ensure a model is registered in MLflow Registry
        """,
        language="bash",
    )

    st.divider()

    st.subheader("Starting the FastAPI Server")
    st.code(
        """
# Navigate to project root
cd /path/to/rossmann-forecasting

# Start FastAPI server
bash scripts/launch_api.sh

# Or run directly
cd deployment/api
python main.py

# Server will start at http://localhost:8000
# API docs available at http://localhost:8000/docs
        """,
        language="bash",
    )

    st.divider()

    st.subheader("Starting the Streamlit Dashboard")
    st.code(
        """
# Navigate to project root
cd /path/to/rossmann-forecasting

# Start Streamlit dashboard
bash scripts/launch_streamlit.sh

# Or run directly
cd deployment/streamlit
streamlit run Home.py

# Dashboard will start at http://localhost:8501
        """,
        language="bash",
    )

    st.divider()

    st.subheader("Production Considerations")
    st.markdown(
        """
        **For Production Deployment**:

        1. **API Server**:
           - Use Gunicorn/Uvicorn with multiple workers
           - Set up reverse proxy (Nginx)
           - Configure CORS for specific origins
           - Enable HTTPS/TLS
           - Set up monitoring and logging

        2. **MLflow**:
           - Use dedicated MLflow tracking server
           - Configure remote artifact storage (S3, Azure Blob)
           - Set up authentication and access control
           - Regular model versioning and cleanup

        3. **Streamlit**:
           - Configure authentication (Streamlit Cloud or custom)
           - Set up environment-specific configs
           - Enable caching for model loading
           - Monitor resource usage

        4. **Infrastructure**:
           - Docker containerization (Phase 4)
           - Kubernetes orchestration (Phase 4)
           - Auto-scaling policies
           - Health checks and circuit breakers
        """
    )

st.divider()

# Additional Resources
st.subheader("📖 Additional Resources")

resource_col1, resource_col2 = st.columns(2)

with resource_col1:
    st.markdown(
        """
        **Documentation**
        - [Model Training Guide](../docs/modelops/training.md)
        - [Model Registry Guide](../docs/modelops/registry.md)
        - [Validation Workflow](../docs/modelops/validation.md)
        """
    )

with resource_col2:
    st.markdown(
        """
        **External Links**
        - [FastAPI Documentation](https://fastapi.tiangolo.com/)
        - [Streamlit Documentation](https://docs.streamlit.io/)
        - [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html)
        - [Pydantic Documentation](https://docs.pydantic.dev/)
        """
    )
