import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve
import numpy as np

class ModelBuilder:
    """
    Classe responsável por criar pipelines de Machine Learning, 
    treinar o modelo e realizar predições.
    """
    
    def __init__(self, df: pd.DataFrame, target_col: str = "classe_defas"):
        """
        Inicializa o builder recebendo os dados completos e a coluna alvo (Y).
        """
        
        self.target_col = target_col
        self.pipeline = None
        self.file_path_model = "model/modelo_risco_alunos_v2.pkl"
        self.columns = ["idade", "inde", "ian", "ida", "ieg", "iaa", "ips", "ipv", "matem", "portug", "no_av", "genero", "instituicao_padronizada", "fase", "rec_psicologia_padronizada"]
        self.df = df[self.columns + [self.target_col]]
        # Onde guardaremos nossos dados separados
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4
        self.best_threshold = None

    def prepare_data(self, test_size: float = 0.2, random_state: int = 42):
        """Separa X (features) e Y (target), e divide em treino e teste."""
        
        # 1. Criação de X e Y
        X = self.df[self.columns]
        y = self.df[self.target_col]
        
        # 2. Separação de Treino e Teste
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"📊 Dados divididos: {len(self.X_train)} linhas de treino e {len(self.X_test)} de teste.")

    def _build_pipeline(self):
        """Método interno para construir o pré-processamento e o modelo."""
        
        features = self.X_train.drop(columns=["RA"], errors="ignore")
        numeric_features = features.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = features.select_dtypes(include=['object', 'category']).columns.tolist()

        categorical_columns = self.X_train.select_dtypes(include="object").columns

        self.X_train[categorical_columns] = (
            self.X_train[categorical_columns]
                .astype(str)
        )

        # Transformador para variáveis Numéricas (Trata nulos com a média e padroniza a escala)
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        # Transformador para variáveis Categóricas (Trata nulos como 'Faltante' e aplica One-Hot)
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Faltante')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Junta os transformadores no ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ], 
            remainder='drop' # Ignora colunas que não especificamos (ex: RA)
        )

        # Pipeline Final: Pré-processamento + Algoritmo (Estimador)

        
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', RandomForestClassifier(random_state=42, n_estimators=300))
        ])

    def train(self):
        """Constrói o pipeline e treina (fit) o modelo."""
        if self.X_train is None:
            raise ValueError("Você precisa chamar 'prepare_data()' antes de treinar.")
            
        print("⚙️ Construindo o pipeline e treinando o modelo...")
        self._build_pipeline()
        
        for col in self.X_train.columns:
            types = self.X_train[col].map(type).unique()
            if len(types) > 1:
                print(f"{col} -> {types}")


        # O .fit() aqui executa os transformadores E treina o modelo de uma vez!
        self.pipeline.fit(self.X_train, self.y_train)
        print("✅ Modelo treinado com sucesso!")
        self.evaluate_model(self.X_test, self.y_test)
        self.save_model()


    def save_model(self):
        if self.pipeline is None:
            raise ValueError("O modelo ainda não foi treinado.")

        model_artifact = {
            "pipeline": self.pipeline,
            "threshold": self.best_threshold
        }

        joblib.dump(model_artifact, self.file_path_model)
        print(f"💾 Modelo e threshold salvos em: {self.file_path_model}")

    def predict(self, new_data: pd.DataFrame) -> list:
        """Realiza predições em novos dados."""
        modelo = joblib.load(self.file_path_model)
        
        return modelo.predict(new_data)

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series):
        y_proba = self.pipeline.predict_proba(X_test)
        classes = self.pipeline.classes_
        severa_index = list(classes).index("Severa")

        # Probabilidade apenas da classe Severa
        y_scores = y_proba[:, severa_index]

        # Criar variável binária: 1 se Severa, 0 caso contrário
        y_true_binary = (y_test == "Severa").astype(int)

        # =============================
        # 2️⃣ Precision-Recall Curve
        # =============================
        precision, recall, thresholds = precision_recall_curve(
            y_true_binary,
            y_scores
        )

        # Remover último ponto (precision_recall_curve retorna 1 valor extra)
        precision = precision[:-1]
        recall = recall[:-1]

        # Calcular F1 para cada threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

        best_index = np.argmax(f1_scores)
        best_threshold = thresholds[best_index]
        self.best_threshold = best_threshold

        print("🏆 Melhor threshold encontrado para 'Severa'")
        print(f"Threshold: {best_threshold:.3f}")
        print(f"Precision: {precision[best_index]:.3f}")
        print(f"Recall: {recall[best_index]:.3f}")
        print(f"F1-score: {f1_scores[best_index]:.3f}")

        # =============================
        # 3️⃣ Aplicar threshold otimizado
        # =============================
        y_pred = []

        for probs in y_proba:
            if probs[severa_index] > best_threshold:
                y_pred.append("Severa")
            else:
                y_pred.append(classes[np.argmax(probs)])

        # =============================
        # 4️⃣ Relatório atualizado
        # =============================
        print("\n--- Relatório de Classificação (Threshold Ajustado) ---")
        print(classification_report(y_test, y_pred))

        # =============================
        # 5️⃣ Matriz de Confusão
        # =============================
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(cmap="Blues", ax=ax)
        plt.title("Matriz de Confusão - Threshold Otimizado")
        plt.show()

        # =============================
        # 6️⃣ Curva Precision-Recall
        # =============================
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision)
        plt.scatter(
            recall[best_index],
            precision[best_index],
            marker="o",
            label="Melhor Threshold",
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve - Classe Severa")
        plt.legend()
        plt.show()

    def predict(self, new_data: pd.DataFrame) -> list:
        model_artifact = joblib.load(self.file_path_model)

        pipeline = model_artifact["pipeline"]
        threshold = model_artifact["threshold"]

        y_proba = pipeline.predict_proba(new_data)
        classes = pipeline.classes_
        severa_index = list(classes).index("Severa")

        predictions = []

        for probs in y_proba:
            if probs[severa_index] > threshold:
                predictions.append("Severa")
            else:
                predictions.append(classes[probs.argmax()])
        return predictions