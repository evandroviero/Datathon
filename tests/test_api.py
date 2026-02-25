from fastapi.testclient import TestClient
from api.app import app # Importe sua instância do FastAPI

client = TestClient(app)

def test_api_predict_endpoint(sample_data):
    payload = sample_data.to_dict(orient="records")[0]
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    json_data = response.json()
    assert "prediction" in json_data