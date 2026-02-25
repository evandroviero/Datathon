import pytest
import pandas as pd

@pytest.fixture
def sample_data():
    """Fixture centralizada para fornecer dados de teste para o Modelo e a API."""
    return pd.DataFrame([{
        "idade": 20, "inde": 5.0, "ian": 5.0, "ida": 5.0, 
        "ieg": 5.0, "iaa": 5.0, "ips": 5.0, "ipv": 5.0, 
        "matem": 5.0, "portug": 5.0, "no_av": 1, 
        "genero": "M", "instituicao_padronizada": "Escola A", 
        "fase": "1", "rec_psicologia_padronizada": "Nao",
        "classe_defas": "Moderada" # Coluna alvo para alguns testes se necessário
    }])