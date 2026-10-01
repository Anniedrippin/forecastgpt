"""
Seed documents for the Qdrant knowledge base.

These are short, hand-written explanations of forecasting concepts. They
get embedded once at startup (idempotent) so that conceptual questions like
"what's the difference between SARIMA and XGBoost?" can be answered via
retrieval instead of relying purely on the LLM's own (unverified) training
knowledge.
"""
from __future__ import annotations

from app.core.logging_config import get_logger
from app.services.qdrant_service import QdrantService

logger = get_logger(__name__)

KNOWLEDGE_DOCS = [
    {
        "text": (
            "Naive forecasting simply repeats the last observed value for every "
            "future period. It is the simplest possible baseline and is "
            "surprisingly hard to beat for series that are close to a random "
            "walk (e.g. many financial series). Always compare other models "
            "against it."
        ),
        "doc_type": "method_doc",
        "topic": "naive",
    },
    {
        "text": (
            "Moving-average forecasting repeats the mean of the last N "
            "observations for every future period. It smooths out noise and "
            "works well for series with no strong trend or seasonality, but "
            "it lags behind a real trend because it only looks backward."
        ),
        "doc_type": "method_doc",
        "topic": "moving_average",
    },
    {
        "text": (
            "SARIMA (Seasonal AutoRegressive Integrated Moving Average) is a "
            "classical statistical model that explicitly models trend through "
            "differencing and seasonality through seasonal AR/MA terms. It "
            "works well on series with a clear, stable seasonal pattern and "
            "produces statistically grounded confidence intervals. It needs "
            "at least two full seasonal cycles of history to estimate "
            "seasonality reliably, and struggles with irregular, changing "
            "patterns or many external drivers."
        ),
        "doc_type": "method_doc",
        "topic": "sarima",
    },
    {
        "text": (
            "XGBoost forecasting reframes the time series as a supervised "
            "learning problem using lag features (past values), rolling "
            "statistics, and calendar features (day of week, month, etc). It "
            "can capture non-linear relationships and interactions between "
            "features that SARIMA cannot, and works well when there are "
            "multiple seasonal patterns or covariates. Its downside is that "
            "multi-step forecasts are generated recursively, so small errors "
            "can compound over a long horizon, and it does not naturally "
            "produce calibrated confidence intervals."
        ),
        "doc_type": "method_doc",
        "topic": "xgboost",
    },
    {
        "text": (
            "MAE (Mean Absolute Error) measures the average absolute "
            "difference between predicted and actual values, in the same "
            "units as the data. RMSE (Root Mean Squared Error) does the same "
            "but squares errors before averaging, which penalizes large "
            "errors more heavily than MAE. MAPE (Mean Absolute Percentage "
            "Error) expresses error as a percentage of the actual value, "
            "which makes it easy to compare across series with different "
            "scales, but it is undefined when actual values are zero."
        ),
        "doc_type": "method_doc",
        "topic": "evaluation_metrics",
    },
    {
        "text": (
            "Time-based train/validation splitting holds out the most recent "
            "N periods of a time series as the validation set instead of "
            "randomly sampling rows. This avoids leaking future information "
            "into training and reflects how the model will actually be used: "
            "to predict values it hasn't seen yet, in chronological order."
        ),
        "doc_type": "method_doc",
        "topic": "time_based_splitting",
    },
    {
        "text": (
            "Anomaly detection in time series can be done with simple "
            "statistical methods like rolling z-scores (flagging points far "
            "from a local rolling mean/standard deviation) or with "
            "unsupervised machine learning methods like Isolation Forest, "
            "which isolates points that are easy to separate from the rest "
            "of the data in feature space. Z-score methods are fast and "
            "explainable; Isolation Forest can catch more subtle, contextual "
            "anomalies at the cost of being harder to interpret."
        ),
        "doc_type": "method_doc",
        "topic": "anomaly_detection",
    },
    {
        "text": (
            "What-if (scenario) analysis in this app applies a hypothetical "
            "percentage adjustment to the recent trend or baseline and "
            "recomputes a forecast, to answer questions like 'what would the "
            "forecast look like if demand grew 10% faster?'. This is a "
            "simplified sensitivity analysis, not a causal model -- it does "
            "not know *why* demand would change, only how the numbers shift "
            "under the stated assumption."
        ),
        "doc_type": "method_doc",
        "topic": "what_if_analysis",
    },
]


def seed_knowledge_base(qdrant: QdrantService) -> None:
    """Idempotently seed the knowledge base. We check a marker payload field
    to avoid duplicating documents on every backend restart."""
    existing = qdrant.search("forecasting method", top_k=1, doc_type="method_doc")
    if existing:
        logger.info("Knowledge base already seeded, skipping.")
        return
    logger.info(f"Seeding knowledge base with {len(KNOWLEDGE_DOCS)} documents...")
    for doc in KNOWLEDGE_DOCS:
        qdrant.add_document(doc["text"], {"doc_type": doc["doc_type"], "topic": doc["topic"]})
    logger.info("Knowledge base seeding complete.")
