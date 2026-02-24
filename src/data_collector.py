import pandas as pd
import os
from pathlib import Path
from sqlalchemy import create_engine
import re
import unicodedata

class DataCollector:
    """
    Classe responsável por coletar, limpar e unificar dados do Datathon.
    """
    TARGET_COLUMNS = [
        "RA", "Fase", "Ano nasc", "Gênero", "Instituição de ensino",
        "Pedra", "INDE", "Cg", "Cf", "Ct", "Nº Av", "IAA", "IEG", "IPS",
        "Rec Psicologia", "IDA", "Matem", "Portug", "Inglês", "Indicado",
        "Atingiu PV", "IPV", "IAN", "Destaque IEG", "Destaque IDA",
        "Destaque IPV", "Defas", "ano_base"
    ]

    def __init__(self, file_path: str | Path = "data/raw/BASE DE DADOS PEDE 2024 - DATATHON.xlsx"):
        self.file_path = Path(file_path)
        self.db_url =  os.getenv("DATABASE_URL", "sqlite:///data/processed/dados_datathon.db")
        self.table_name = "pede_participants"
        self.engine = create_engine(self.db_url, echo=False)

    def _process_year(self, year: int, sheet_name: str, rename_map: dict) -> pd.DataFrame:
        """Método interno genérico para extrair e padronizar o dataframe de um ano específico."""
        df = pd.read_excel(self.file_path, sheet_name=sheet_name)
        df = df.rename(columns=rename_map)
        if "Data de Nasc" in df.columns:
            df["Ano nasc"] = pd.to_datetime(df["Data de Nasc"], errors="coerce").dt.year
        df["ano_base"] = year
        return df[self.TARGET_COLUMNS].copy()

    def collect_all_data(self, ) -> pd.DataFrame:
        """Coleta, padroniza e concatena os dados de 2022, 2023 e 2024."""
        configs = {
            2022: {
                "sheet_name": "PEDE2022",
                "rename_map": {
                    "INDE 22": "INDE", 
                    "Pedra 22": "Pedra"
                }
            },
            2023: {
                "sheet_name": "PEDE2023",
                "rename_map": {
                    "INDE 2023": "INDE",
                    "Pedra 23": "Pedra", "Mat": "Matem", 
                    "Por": "Portug", "Ing": "Inglês", "Defasagem": "Defas"
                }
            },
            2024: {
                "sheet_name": "PEDE2024",
                "rename_map": {
                    "INDE 2024": "INDE", "Pedra 2024": "Pedra", 
                    "Mat": "Matem", "Por": "Portug", 
                    "Ing": "Inglês", "Defasagem": "Defas"
                }
            }
        }
        
        dataframes = [
            self._process_year(year=yr, sheet_name=cfg["sheet_name"], rename_map=cfg["rename_map"])
            for yr, cfg in configs.items()
        ]

        dfs = pd.concat(dataframes, ignore_index=True)
        return self.format_dataframe(dfs)
    
    def instituicao(self, df: pd.DataFrame) -> None:
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

        df["Instituicao_padronizada"] = (
            df["Instituição de ensino"]
            .map(mapa_instituicao)
            .fillna("Nao_informado")
        )
        df["Instituicao_padronizada"].value_counts(dropna=False)
        return df
    
    def psicologia(self, df: pd.DataFrame) -> None:
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

        df["Rec_Psicologia_padronizada"] = (
            df["Rec Psicologia"]
            .map(mapa_psicologia)
            .fillna("Nao_Informado")
        )
        df["Rec_Psicologia_padronizada"].value_counts(dropna=False)
        return df
    
    def classificar_defas(self, x):
        if x >= 0:
            return "Em fase"
        elif x >= -2:
            return "Moderada"
        else:
            return "Severa"
        
    def idade(self, df: pd.DataFrame) -> None:
        df["idade"] = df["ano_base"] - df["Ano nasc"]
        return df

    
    def format_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self.instituicao(df)
        df = self.psicologia(df)
        df = self.idade(df)
        df["classe_defas"] = df["Defas"].apply(self.classificar_defas)
        df["Nº Av"] = df["Nº Av"].fillna(0).astype(int)
        return self.save_dataframe(df)

    def save_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Salva um DataFrame do Pandas em uma tabela SQL.
        """
        df.columns = [self.normalize_column_name(col) for col in df.columns]
        df.to_sql(self.table_name, if_exists="replace", index=False, con=self.engine)
        return df

    def normalize_column_name(self, column_name: str) -> str:
        # Remove accents
        normalized = (
            unicodedata.normalize("NFKD", column_name)
            .encode("ascii", "ignore")
            .decode("utf-8")
        )

        # Convert to lowercase
        normalized = normalized.lower()

        # Replace whitespace with underscore
        normalized = re.sub(r"\s+", "_", normalized)

        # Remove non-alphanumeric characters except underscore
        normalized = re.sub(r"[^a-z0-9_]", "", normalized)

        # Collapse multiple underscores
        normalized = re.sub(r"_+", "_", normalized)

        # Strip leading/trailing underscores
        normalized = normalized.strip("_")

        return normalized