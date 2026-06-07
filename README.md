# 🌍 GeoAI Manouba: Integrated LULC Analysis & CA-Markov Prediction Platform

[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Backend-black?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![GeoPandas](https://img.shields.io/badge/GeoPandas-Spatial-green?style=for-the-badge&logo=python&logoColor=white)](https://geopandas.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**A comprehensive geospatial decision-support system for Land Use/Land Cover (LULC) monitoring, change detection, and Cellular Automata-Markov Chain (CA-Markov) predictive modeling for the Manouba region, Tunisia.**

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Project Structure](#️-project-structure)
- [Backend Architecture](#️-backend-architecture)
- [Mathematical Model](#-mathematical-model-ca-markov)
- [API Endpoints](#-api-endpoints)
- [Author & Contact](#-author--contact)

---

## 🌟 Overview

This platform bridges the gap between raw satellite data and actionable spatial intelligence. It ingests historical LULC classifications (1985 and 2025), computes spatial transition matrices, and utilizes a **Cellular Automata-Markov Chain (CA-Markov)** model to predict future land cover scenarios (2030, 2035, 2040). 

The system also features an interactive **Scenario Planning Game**, allowing users to simulate environmental interventions (e.g., reforestation, urban expansion) and observe their impact on future spatial distributions in real-time.

---

## 🗂️ Project Structure

``` text 
LULC_ANALYSIS_PLATFORM/
│
├── server.py                         # Main Flask API Backend (v4.0)
├── server_get_prediction_geojs.py    # Standalone CA-Markov Prediction Script
│
├── capstone_web_with_maps.html       # Frontend: Visualization & Statistics Dashboard
├── manouba_advnced_ca_markov.html    # Frontend: Prediction & Scenario Game Interface
│
├── lulc_1985.geojson                 # Historical LULC Data (1985) [Excluded: >100MB]
├── lulc_2025.geojson                 # Historical LULC Data (2025) [Excluded: >100MB]
├── cd_1985_2025.geojson              # Change Detection Map (1985-2025) [Excluded: >100MB]
│
├── pred_2030.geojson                 # Generated Prediction (2030) [Excluded: >100MB]
├── pred_2035.geojson                 # Generated Prediction (2035) [Excluded: >100MB]
├── pred_2040.geojson                 # Generated Prediction (2040) [Excluded: >100MB]
│
├── requirements.txt                  # Python Dependencies
├── .gitignore                        # Git Configuration
└── README.md                         # Project Documentation
````

Note on Data Files: Due to GitHub's 100MB file size limit, the large .geojson spatial data files are not included in this repository. To run the backend locally, you will need to generate your own GeoJSON files using Landsat imagery and place them in the root directory with the exact names listed above.

---

##⚙️ Backend Architecture
The backend is split into two Python modules to separate real-time API serving from heavy spatial processing:
server.py (Flask API v4.0): Powers the frontend dashboards. It loads the 1985 and 2025 GeoJSONs, computes the real 40-year transition matrix based on pixel area changes, and handles scenario interventions (e.g., simulating a new dam, forest, or residential area) by dynamically recalculating the transition probabilities.
server_get_prediction_geojs.py (Standalone Engine): A heavy-processing script used to pre-generate the prediction maps. It uses scipy.spatial.cKDTree to map polygon neighbors within a 45m radius and applies Cellular Automata (CA) smoothing (majority voting over 2 passes) to eliminate isolated pixels and create realistic, contiguous land cover patches.

---

##🧮 Mathematical Model (CA-Markov)
The prediction engine uses a hybrid approach to ensure both statistical accuracy and spatial realism:
Markov Chain Transition: Calculates the probability of land cover transitioning from one class to another based on historical changes (1985 → 2025).
Annual Matrix Derivation: Uses eigenvalue decomposition to convert the 40-year transition matrix into an annual transition rate.
Cellular Automata (CA) Smoothing: Pure Markov chains ignore spatial context. To fix this, the standalone script applies a spatial filter where a polygon's predicted class is adjusted based on the majority class of its spatial neighbors, preventing "salt-and-pepper" artifacts.

---

##📡 API Endpoints
The Flask server (server.py) exposes three main endpoints:
GET /api/health — Checks server status, loaded data, and baseline 1985 percentages.
POST /api/generate_predictions — Generates full spatial GeoJSON predictions for selected years (2030, 2035, 2040) based on user interventions. Returns the GeoJSONs, updated statistics, and an environmental health score.
POST /api/predict — Returns a lightweight statistical trajectory (percentages over time) for fast chart rendering without heavy geometry data.

---

##👤 Author & Contact
Roua Remadi | Geomatics Engineering Student @ MSE
📧 rouaremadi5@gmail.com
💼 LinkedIn: linkedin.com/in/rouaremadi
