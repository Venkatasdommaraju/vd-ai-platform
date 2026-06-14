# Projects

Agentic AI for REMS (Pfizer, 2024-2025): Built a LangGraph multi-agent workflow for the REMS regulatory platform. Agents handle document retrieval, policy analysis, and compliance checking. Deployed to production on AWS with LangSmith observability and Elasticsearch vector search.

RAG Pipeline on AWS Bedrock: Designed a production RAG system that indexes 10,000+ regulatory documents, handles hybrid search (BM25 + vector), and generates grounded responses with citations. Built with Python, FastAPI, Bedrock Embeddings, and OpenSearch.

Snowflake Lakehouse Migration: Led migration of 6 legacy Oracle/SQL Server sources to Snowflake, including dbt models, Airflow DAGs, and BI dashboards. Reduced data latency from 24 hours to 15 minutes.

IoT Sensor Anomaly Detection: ML service that predicts environmental anomalies from sensor data using scikit-learn logistic regression. Deployed as FastAPI microservice with REST API and interactive web UI.

Agentic Microservice Demo: Production-shaped LangGraph service demonstrating planner-retriever-analyst-critic multi-agent loop with pluggable LLM backends and TF-IDF retrieval.

BIOVIA Pipeline Pilot Integration: Python APIs connecting BIOVIA scientific data workflows to downstream Snowflake and reporting systems in a GxP-validated environment.
