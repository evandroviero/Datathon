import streamlit as st
import requests
import pandas as pd

# -----------------------
# Configuração da Página
# -----------------------
st.set_page_config(
    page_title="Predição de Defasagem Escolar",
    page_icon="🎓",
    layout="wide"
)

# -----------------------
# Estilo customizado
# -----------------------
st.markdown("""
    <style>
        .main {
            background-color: #f4f6f9;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            height: 45px;
            width: 100%;
        }
        .risk-high {
            color: red;
            font-size: 24px;
            font-weight: bold;
        }
        .risk-low {
            color: green;
            font-size: 24px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------
# Título
# -----------------------
st.title("🎓 Sistema de Predição de Defasagem Escolar")
st.markdown("Avalie o risco de defasagem com base nos indicadores acadêmicos e comportamentais.")

# -----------------------
# Sidebar - Dados do Aluno
# -----------------------
st.sidebar.header("📋 Informações do Aluno")

ano_nascimento = st.sidebar.number_input("Ano de Nascimento", 2000, 2015, 2005)
idade = 2024 - ano_nascimento
genero = st.sidebar.selectbox("Gênero", ["Menina", "Menino"])

mapa_instituicao = {
    # Pública
    "Pública": "Publica",
    "Escola Pública": "Publica",

    # Privada
    "Privada": "Privada",

    # Privada com parcerias / bolsa
    "Privada - Programa de Apadrinhamento": "Privada_Parceria",
    "Privada - Programa de apadrinhamento": "Privada_Parceria",
    "Privada *Parcerias com Bolsa 100%": "Privada_Parceria",
    "Privada - Pagamento por *Empresa Parceira": "Privada_Parceria",

    # Redes específicas
    "Rede Decisão": "Rede_Decisao",

    # Situação acadêmica posterior
    "Concluiu o 3º EM": "Universitario",
    "Bolsista Universitário *Formado (a)": "Universitario",

    # Outros
    "Escola JP II": "Outros",
    "Nenhuma das opções acima": "Outros",
}
instituicao = st.sidebar.selectbox("Instituição", mapa_instituicao.keys())
instituicao = mapa_instituicao[instituicao]

mapa_psicologia = {
    # Avaliado sem risco
    "Sem limitações": "Sem_Risco",

    # Em processo de avaliação
    "Requer avaliação": "Em_Avaliacao",
    "Não avaliado": "Em_Avaliacao",

    # Casos com atenção psicológica
    "Não atendido": "Risco_Psicologico",

    # Avaliado e não indicado
    "Não indicado": "Nao_Indicado"
}

rec_psicologia = st.sidebar.selectbox(
    "Rec Psicologia",
    options=list(mapa_psicologia.keys())
)
rec_psicologia = mapa_psicologia[rec_psicologia]

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Indicadores")

INDE = st.sidebar.slider("INDE", 0.0, 10.0, 5.7)
IAN = st.sidebar.slider("IAN", 0.0, 10.0, 5.0)
IDA = st.sidebar.slider("IDA", 0.0, 10.0, 4.0)
IEG = st.sidebar.slider("IEG", 0.0, 10.0, 4.1)
IAA = st.sidebar.slider("IAA", 0.0, 10.0, 8.3)
IPS = st.sidebar.slider("IPS", 0.0, 10.0, 5.6)
IPV = st.sidebar.slider("IPV", 0.0, 10.0, 7.2)

Matem = st.sidebar.slider("Nota Matemática", 0.0, 10.0, 2.7)
Portug = st.sidebar.slider("Nota Português", 0.0, 10.0, 3.5)
n_av = st.sidebar.slider("Número de Avaliações", 0.0, 10.0, 4.0)

# -----------------------
# Botão de Predição
# -----------------------
st.markdown("## 📈 Resultado da Análise")

if st.button("🔍 Analisar Risco de Defasagem"):

    payload = {
        "idade": idade,
        "inde": INDE,
        "ian": IAN,
        "ida": IDA,
        "ieg": IEG,
        "iaa": IAA,
        "ips": IPS,
        "ipv": IPV,
        "matem": Matem,
        "portug": Portug,
        "no_av": n_av,
        "genero": genero,
        "instituicao_padronizada": instituicao,
        "rec_psicologia_padronizada": rec_psicologia,
        "classe_defas": ""
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=payload
        )

        result = response.json()

        col1 = st.columns(1)


        if "Severa" in result.get("prediction"):
            st.markdown(
                '<p class="risk-high">⚠ Alto Risco de Defasagem</p>',
                unsafe_allow_html=True
            )
        elif "Moderada" in result.get("prediction"):
            st.markdown(
                '<p class="risk-medium">⚠ Moderado Risco de Defasagem</p>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<p class="risk-low">✅ Baixo Risco de Defasagem</p>',
                unsafe_allow_html=True
            )

    except Exception as e:
        st.error(f"Erro ao conectar com API: {e}")

# -----------------------
# Footer
# -----------------------
st.markdown("---")
st.markdown("Desenvolvido para suporte à tomada de decisão pedagógica 📚")