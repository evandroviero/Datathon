import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, make_scorer, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

class ModelBuilder:
    def __init__(self, df: pd.DataFrame, target_col: str = "classe_defas"):
        self.target_col = target_col
        self.pipeline = None
        self.file_path_model = "model/modelo_risco_alunos_v2.pkl"
        self.columns = ["idade", "inde", "ian", "ida", "ieg", "iaa", "ips", "ipv", "matem", "portug", "no_av", "genero", "instituicao_padronizada", "fase", "rec_psicologia_padronizada"]
        self.df = df[self.columns + [self.target_col]].copy()
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4
        self.best_threshold = None

    def prepare_data(self, test_size: float = 0.2, random_state: int = 42):
        X = self.df[self.columns]
        y = self.df[self.target_col]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"📊 Treino: {len(self.X_train)} | Teste: {len(self.X_test)}")

    def _build_pipeline(self):
        numeric_features = self.X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = self.X_train.select_dtypes(include=['object', 'category']).columns.tolist()

        self.X_train[categorical_features] = self.X_train[categorical_features].astype(str)
        if self.X_test is not None:
            self.X_test[categorical_features] = self.X_test[categorical_features].astype(str)

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            # Agora o imputer trata a string 'nan' se ela existir, ou valores vazios
            ('imputer', SimpleImputer(strategy='constant', fill_value='Faltante')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42, k_neighbors=min(3, self.y_train.value_counts().min() - 1))), 
            ('classifier', RandomForestClassifier(
                random_state=42, 
                n_estimators=300, 
                class_weight='balanced_subsample',
                max_depth=10
            ))
        ])

    def save_model(self):
        if self.pipeline is None:
            raise ValueError("O modelo ainda não foi treinado.")

        model_artifact = {
            "pipeline": self.pipeline,
            "threshold": self.best_threshold
        }

        joblib.dump(model_artifact, self.file_path_model)
        print(f"💾 Modelo e threshold salvos em: {self.file_path_model}")

    def run_cross_validation(self):
        print("\n🔍 Iniciando Validação Cruzada...")
 
        cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=42) # n_splits menor devido à classe Severa
        
        scoring = {'f1_macro': 'f1_macro', 'accuracy': 'accuracy'}
        
        results = cross_validate(self.pipeline, self.X_train, self.y_train, cv=cv, scoring=scoring)
        
        print(f"Mean F1-Macro: {results['test_f1_macro'].mean():.3f}")
        print(f"Mean Accuracy: {results['test_accuracy'].mean():.3f}")
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series):
        y_proba = self.pipeline.predict_proba(X_test)
        classes = self.pipeline.classes_
        severa_index = list(classes).index("Severa")

        y_scores = y_proba[:, severa_index]

        y_true_binary = (y_test == "Severa").astype(int)

        precision, recall, thresholds = precision_recall_curve(
            y_true_binary,
            y_scores
        )

        precision = precision[:-1]
        recall = recall[:-1]

        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

        best_index = np.argmax(f1_scores)
        best_threshold = thresholds[best_index]
        self.best_threshold = best_threshold

        print("🏆 Melhor threshold encontrado para 'Severa'")
        print(f"Threshold: {best_threshold:.3f}")
        print(f"Precision: {precision[best_index]:.3f}")
        print(f"Recall: {recall[best_index]:.3f}")
        print(f"F1-score: {f1_scores[best_index]:.3f}")

        y_pred = []

        for probs in y_proba:
            if probs[severa_index] > best_threshold:
                y_pred.append("Severa")
            else:
                y_pred.append(classes[np.argmax(probs)])

        print("\n--- Relatório de Classificação (Threshold Ajustado) ---")
        print(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred, labels=classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(cmap="Blues", ax=ax)
        plt.title("Matriz de Confusão - Threshold Otimizado")
        plt.show()

        # plt.figure(figsize=(8, 6))
        # plt.plot(recall, precision)
        # plt.scatter(
        #     recall[best_index],
        #     precision[best_index],
        #     marker="o",
        #     label="Melhor Threshold",
        # )
        # plt.xlabel("Recall")
        # plt.ylabel("Precision")
        # plt.title("Precision-Recall Curve - Classe Severa")
        # plt.legend()
        # plt.show()

    def plot_feature_importance(self):
        model = self.pipeline.named_steps['classifier']
        ohe_cols = self.pipeline.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out()
        num_cols = self.X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        all_features = num_cols + list(ohe_cols)
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[-15:] # Top 15

        plt.figure(figsize=(10, 6))
        plt.title("Top 15 Feature Importances (Verifique vazamentos!)")
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [all_features[i] for i in indices])
        plt.show()

    def export_error_analysis(self, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Gera um relatório detalhado dos erros para inspeção manual.
        """
        y_proba = self.pipeline.predict_proba(X_test)
        classes = self.pipeline.classes_
        severa_idx = list(classes).index("Severa")
        
        y_pred = []
        for probs in y_proba:
            if probs[severa_idx] >= self.best_threshold:
                y_pred.append("Severa")
            else:
                y_pred.append(classes[np.argmax(probs)])

        analysis_df = X_test.copy()
        analysis_df['Real'] = y_test.values
        analysis_df['Predito'] = y_pred
        analysis_df['Prob_Severa'] = y_proba[:, severa_idx]
        
        errors = analysis_df[analysis_df['Real'] != analysis_df['Predito']].copy()
        
        def categorize_error(row):
            if row['Real'] == 'Severa' and row['Predito'] != 'Severa':
                return 'Falso Negativo (Risco!)'
            if row['Real'] != 'Severa' and row['Predito'] == 'Severa':
                return 'Falso Positivo (Alarme Falso)'
            return 'Erro entre outras classes'

        errors['Tipo_Erro'] = errors.apply(categorize_error, axis=1)
        
        errors = errors.sort_values(by='Prob_Severa', ascending=False)
        
        print(f"❌ Total de erros encontrados: {len(errors)}")
        errors.to_csv("analise_de_erros_modelo.csv", index=False)
        print("💾 Relatório salvo como 'analise_de_erros_modelo.csv'")
        return errors


    def train(self):
        if self.X_train is None: self.prepare_data()
        self._build_pipeline()
        
        self.run_cross_validation()
    
        self.pipeline.fit(self.X_train, self.y_train)

        self.evaluate_model(self.X_test, self.y_test)
        self.export_error_analysis(self.X_test, self.y_test)
        self.save_model()