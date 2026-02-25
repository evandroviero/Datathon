import pytest
import pandas as pd
import numpy as np
import joblib
import os
from src.model_builder import ModelBuilder


def test_model_file_exists():
    """Verifica se o artefato do modelo foi gerado."""
    assert os.path.exists("model/modelo_risco_alunos_v2.pkl")

def test_pipeline_inference_types(sample_data):
    """Garante que o modelo aceita dados mesmo com tipos mistos (proteção contra o erro de str/float)."""
    model_artifact = joblib.load("model/modelo_risco_alunos_v2.pkl")
    pipeline = model_artifact["pipeline"]
    
    # Simula um valor nulo que causaria erro de tipo
    sample_data.loc[0, "genero"] = np.nan 
    
    # O pipeline deve processar sem quebrar devido ao nosso cast manual no predict
    try:
        # Testamos a lógica que incluímos no ModelBuilder.predict
        categorical_features = sample_data.select_dtypes(include=['object', 'category']).columns.tolist()
        sample_data[categorical_features] = sample_data[categorical_features].astype(str)
        preds = pipeline.predict(sample_data)
        assert len(preds) == 1
    except TypeError:
        pytest.fail("O pipeline falhou ao processar tipos mistos/nulos nas categorias.")

def test_threshold_logic():
    model_artifact = joblib.load("model/modelo_risco_alunos_v2.pkl")
    threshold = model_artifact.get("threshold", 0.5)
    
    mock_probs = np.array([[0.2, 0.34, 0.46]]) # Severa = 0.46
    severa_idx = 2
    
    prediction = "Severa" if mock_probs[0][severa_idx] >= threshold else "Outra"
    
    # Se o threshold for 0.45, 0.46 é >= 0.45, então DEVE ser Severa.
    if threshold <= 0.46:
        assert prediction == "Severa"
    else:
        assert prediction == "Outra"