import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from typing import Optional

class DatabaseLoader:
    """
    Classe responsável por gerenciar a conexão com o banco de dados SQL 
    e realizar o carregamento (Load) dos dados de forma segura.
    """
    
    def __init__(self, db_url: Optional[str] = None):
        """
        Inicializa o gerenciador de banco de dados.
        
        Se a URL não for fornecida diretamente, tenta buscar nas variáveis de 
        ambiente (arquivo .env). Se não encontrar, usa um SQLite local como fallback.
        """
        self.db_url = db_url or os.getenv("DATABASE_URL", "sqlite:///dados_datathon.db")
        
        # O engine gerencia o pool de conexões com o banco de dados
        self.engine = create_engine(self.db_url, echo=False)

    def save_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = "append") -> bool:
        """
        Salva um DataFrame do Pandas em uma tabela SQL utilizando transações seguras.
        
        Parâmetros:
            df (pd.DataFrame): O dataframe com os dados limpos a serem salvos.
            table_name (str): Nome da tabela de destino no banco de dados.
            if_exists (str): Comportamento se a tabela já existir ('fail', 'replace', 'append').
        
        Retorna:
            bool: True se a operação foi bem-sucedida, False caso contrário.
        """
        if df is None or df.empty:
            print("⚠️ O DataFrame está vazio ou é nulo. Nenhuma ação foi realizada no banco.")
            return False

        try:
            # O .begin() abre uma transação (transaction). 
            # Se ocorrer um erro no meio do insert, ele faz um rollback automático.
            with self.engine.begin() as connection:
                df.to_sql(
                    name=table_name,
                    con=connection,
                    if_exists=if_exists,
                    index=False,      # Evita salvar o índice numérico do Pandas como uma coluna
                    chunksize=1000    # Insere de 1000 em 1000 linhas para não estourar a memória RAM
                )
            
            print(f"✅ Sucesso: {len(df)} registros foram salvos na tabela '{table_name}'.")
            return True
            
        except SQLAlchemyError as db_error:
            print(f"❌ Erro de Banco de Dados ao salvar a tabela '{table_name}': {db_error}")
            return False
            
        except Exception as generic_error:
            print(f"❌ Erro inesperado durante o salvamento: {generic_error}")
            return False