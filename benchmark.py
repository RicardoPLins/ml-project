import warnings

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, log_loss, roc_auc_score
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend sem GUI
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import openml
from typing import Dict, List, Tuple, Any
import json
from pathlib import Path
import os
import gc
import traceback
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from pytorch_tabnet.tab_model import TabNetClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
from autogluon.tabular import TabularPredictor
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

# Auto-sklearn (opcional)
AUTOSKLEARN_AVAILABLE = False
try:
    import autosklearn.classification as _askl
    from autosklearn.classification import AutoSklearnClassifier
    AUTOSKLEARN_AVAILABLE = True
except Exception:
    AUTOSKLEARN_AVAILABLE = False

# Configurações
np.random.seed(42)
plt.style.use('seaborn-v0_8-darkgrid')

# Diretórios (ajustado para Linux/VSCode)
RESULTS_DIR = "./results"  # Diretório local ao invés de /content
os.makedirs(RESULTS_DIR, exist_ok=True)
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

print("="*80)
print("Bibliotecas importadas com sucesso!")
print(f"Resultados serão salvos em: {os.path.abspath(RESULTS_DIR)}")
print("="*80 + "\n")


class MLBenchmark:
    """Benchmark ML - Versão corrigida para Linux/VSCode"""
    
    def __init__(self, n_datasets: int = 30, test_size: float = 0.3, 
                 random_state: int = 16, cv_folds: int = 5):
        self.n_datasets = n_datasets
        self.test_size = test_size
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.results = []
        self.datasets_info = []
        
        print("="*80)
        print("Benchmark ML Inicializado")
        print("="*80)
        print(f"Datasets: {n_datasets}")
        print(f"Split: {int((1-test_size)*100)}/{int(test_size)*100}")
        print(f"CV Folds: {cv_folds}")
        print(f"Random State: {random_state}")
        print(f"Modelos: TabNet, LightGBM, CatBoost, XGBoost, AutoGluon" + 
              (", Auto-sklearn" if AUTOSKLEARN_AVAILABLE else ""))
        print(f"Resultados: {os.path.abspath(RESULTS_DIR)}")
        print("="*80 + "\n")

    def get_openml_datasets(self) -> List[int]:
        """Retorna IDs dos datasets OpenML-CC18"""
        print("Buscando datasets do OpenML-CC18...")
        cc18_ids = [
            3, 6, 11, 12, 14, 15, 16, 18, 22, 23,
            28, 29, 31, 32, 37, 44, 46, 50, 54, 151,
            182, 188, 273, 293, 300, 307, 458, 469, 554, 1049
        ]
        selected = cc18_ids[:self.n_datasets]
        print(f"✓ {len(selected)} datasets selecionados\n")
        return selected

    def load_and_preprocess_dataset(self, dataset_id: int) -> Tuple:
        """Carrega e pré-processa dataset"""
        try:
            dataset = openml.datasets.get_dataset(dataset_id)
            X, y, _, _ = dataset.get_data(
                dataset_format="array",
                target=dataset.default_target_attribute
            )

            X = np.array(X)
            y = np.array(y)

            info = {
                'id': dataset_id,
                'name': dataset.name,
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'n_classes': len(np.unique(y))
            }

            le = LabelEncoder()
            y = le.fit_transform(y)

            if np.isnan(X).any():
                imputer = SimpleImputer(strategy='mean')
                X = imputer.fit_transform(X)

            print(f"  Dataset {dataset_id} ({dataset.name}): "
                  f"{info['n_samples']}×{info['n_features']}, {info['n_classes']} classes")
            return X, y, info

        except Exception as e:
            print(f"  ✗ Erro: {e}")
            return None, None, None

    def calculate_gmean(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula G-Mean"""
        cm = confusion_matrix(y_true, y_pred)
        sensitivities = []
        for i in range(len(cm)):
            if cm[i].sum() > 0:
                sensitivities.append(cm[i, i] / cm[i].sum())
        if not sensitivities:
            return 0.0
        return np.prod(sensitivities) ** (1.0 / len(sensitivities))

    def evaluate_model(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Avalia modelo e retorna métricas"""
        y_pred = model.predict(X)
        
        try:
            y_proba = model.predict_proba(X)
        except:
            n_classes = len(np.unique(y))
            y_proba = np.zeros((len(y_pred), n_classes))
            for i, p in enumerate(y_pred):
                if int(p) < n_classes:
                    y_proba[i, int(p)] = 1.0

        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'gmean': self.calculate_gmean(y, y_pred)
        }
        
        n_classes = len(np.unique(y))
        try:
            if n_classes == 2:
                if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                    metrics['auc_ovo'] = roc_auc_score(y, y_proba[:, 1])
                else:
                    metrics['auc_ovo'] = roc_auc_score(y, y_proba)
            else:
                metrics['auc_ovo'] = roc_auc_score(y, y_proba, multi_class='ovo', average='macro')
        except:
            metrics['auc_ovo'] = 0.0

        try:
            metrics['cross_entropy'] = log_loss(y, y_proba)
        except:
            metrics['cross_entropy'] = np.inf

        return metrics

    def _random_search(self, estimator, param_dist, X_train, y_train, n_iter=12):
        """RandomizedSearchCV com StratifiedKFold"""
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        rs = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            random_state=self.random_state,
            verbose=0
        )
        t0 = time.perf_counter()
        rs.fit(X_train, y_train)
        tune_time = time.perf_counter() - t0
        return rs.best_estimator_, tune_time, rs

    def train_tabnet(self, X_train, y_train, X_test, y_test):
        """Treina TabNet"""
        start = time.perf_counter()
        
        param_grid = [
            {'n_d': 8, 'n_a': 8, 'n_steps': 3, 'gamma': 1.3},
            {'n_d': 16, 'n_a': 16, 'n_steps': 5, 'gamma': 1.5},
            {'n_d': 32, 'n_a': 32, 'n_steps': 4, 'gamma': 1.2},
        ]
        
        best_score, best_model, tune_time = -np.inf, None, 0.0
        
        for params in param_grid:
            t0 = time.perf_counter()
            model = TabNetClassifier(**params, seed=self.random_state, verbose=0)
            try:
                model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                         max_epochs=100, patience=20, batch_size=256, virtual_batch_size=128)
            except Exception as e:
                print(f"      TabNet fit error: {e}")
            
            tune_time += (time.perf_counter() - t0)
            score = accuracy_score(y_test, model.predict(X_test))
            
            if score > best_score:
                best_score, best_model = score, model

        train_metrics = self.evaluate_model(best_model, X_train, y_train)
        test_metrics = self.evaluate_model(best_model, X_test, y_test)
        total_time = time.perf_counter() - start

        return best_model, {'train': train_metrics, 'test': test_metrics, 'tune_time': tune_time}, total_time, {}

    def train_lightgbm(self, X_train, y_train, X_test, y_test):
        """Treina LightGBM"""
        start = time.perf_counter()
        
        param_dist = {
            'num_leaves': [20, 31, 50, 80],
            'learning_rate': [0.01, 0.05, 0.1, 0.15],
            'n_estimators': [50, 100, 200, 300]
        }
        
        try:
            best_model, tune_time, _ = self._random_search(
                lgb.LGBMClassifier(random_state=self.random_state, verbose=-1),
                param_dist, X_train, y_train, 12
            )
        except:
            best_model = lgb.LGBMClassifier(random_state=self.random_state, verbose=-1).fit(X_train, y_train)
            tune_time = 0.0

        train_metrics = self.evaluate_model(best_model, X_train, y_train)
        test_metrics = self.evaluate_model(best_model, X_test, y_test)
        total_time = time.perf_counter() - start

        return best_model, {'train': train_metrics, 'test': test_metrics, 'tune_time': tune_time}, total_time, {}

    def train_catboost(self, X_train, y_train, X_test, y_test):
        """Treina CatBoost"""
        start = time.perf_counter()
        
        param_dist = {
            'depth': [4, 6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'iterations': [100, 200, 400]
        }
        
        try:
            best_model, tune_time, _ = self._random_search(
                CatBoostClassifier(random_state=self.random_state, verbose=0),
                param_dist, X_train, y_train, 12
            )
        except:
            best_model = CatBoostClassifier(random_state=self.random_state, verbose=0).fit(X_train, y_train)
            tune_time = 0.0

        train_metrics = self.evaluate_model(best_model, X_train, y_train)
        test_metrics = self.evaluate_model(best_model, X_test, y_test)
        total_time = time.perf_counter() - start

        return best_model, {'train': train_metrics, 'test': test_metrics, 'tune_time': tune_time}, total_time, {}

    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Treina XGBoost"""
        start = time.perf_counter()
        
        param_dist = {
            'max_depth': [3, 4, 6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'n_estimators': [50, 100, 200, 300]
        }
        
        try:
            best_model, tune_time, _ = self._random_search(
                xgb.XGBClassifier(eval_metric='logloss', random_state=self.random_state),
                param_dist, X_train, y_train, 12
            )
        except:
            best_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', 
                                          random_state=self.random_state).fit(X_train, y_train)
            tune_time = 0.0

        train_metrics = self.evaluate_model(best_model, X_train, y_train)
        test_metrics = self.evaluate_model(best_model, X_test, y_test)
        total_time = time.perf_counter() - start

        return best_model, {'train': train_metrics, 'test': test_metrics, 'tune_time': tune_time}, total_time, {}

    def train_autogluon(self, X_train, y_train, X_test, y_test):
        """Treina AutoGluon"""
        start = time.perf_counter()
        
        train_data = pd.DataFrame(X_train)
        train_data['target'] = y_train
        test_data = pd.DataFrame(X_test)

        try:
            predictor = TabularPredictor(label='target', eval_metric='accuracy', verbosity=0).fit(
                train_data, time_limit=120, presets='best_quality'
            )
            
            y_pred_test = predictor.predict(test_data).values
            y_proba_test = predictor.predict_proba(test_data).values
            y_pred_train = predictor.predict(train_data.drop(columns=['target'])).values
            y_proba_train = predictor.predict_proba(train_data.drop(columns=['target'])).values
            
            n_classes = len(np.unique(y_train))
            
            # Treino
            train_metrics = {
                'accuracy': accuracy_score(y_train, y_pred_train),
                'gmean': self.calculate_gmean(y_train, y_pred_train)
            }
            try:
                if n_classes == 2 and y_proba_train.shape[1] == 2:
                    train_metrics['auc_ovo'] = roc_auc_score(y_train, y_proba_train[:, 1])
                else:
                    train_metrics['auc_ovo'] = roc_auc_score(y_train, y_proba_train, multi_class='ovo', average='macro')
            except:
                train_metrics['auc_ovo'] = 0.0
            
            try:
                train_metrics['cross_entropy'] = log_loss(y_train, y_proba_train)
            except:
                train_metrics['cross_entropy'] = np.inf
            
            # Teste
            test_metrics = {
                'accuracy': accuracy_score(y_test, y_pred_test),
                'gmean': self.calculate_gmean(y_test, y_pred_test)
            }
            try:
                if n_classes == 2 and y_proba_test.shape[1] == 2:
                    test_metrics['auc_ovo'] = roc_auc_score(y_test, y_proba_test[:, 1])
                else:
                    test_metrics['auc_ovo'] = roc_auc_score(y_test, y_proba_test, multi_class='ovo', average='macro')
            except:
                test_metrics['auc_ovo'] = 0.0
            
            try:
                test_metrics['cross_entropy'] = log_loss(y_test, y_proba_test)
            except:
                test_metrics['cross_entropy'] = np.inf
                
        except Exception as e:
            print(f"      AutoGluon error: {e}")
            train_metrics = {'accuracy': 0.0, 'gmean': 0.0, 'auc_ovo': 0.0, 'cross_entropy': np.inf}
            test_metrics = {'accuracy': 0.0, 'gmean': 0.0, 'auc_ovo': 0.0, 'cross_entropy': np.inf}
            predictor = None

        total_time = time.perf_counter() - start
        return predictor, {'train': train_metrics, 'test': test_metrics, 'tune_time': 0.0}, total_time, {}

    def train_autosklearn(self, X_train, y_train, X_test, y_test):
        """Treina Auto-sklearn"""
        if not AUTOSKLEARN_AVAILABLE:
            dummy = {
                'train': {'accuracy': 0.0, 'gmean': 0.0, 'auc_ovo': 0.0, 'cross_entropy': np.inf},
                'test': {'accuracy': 0.0, 'gmean': 0.0, 'auc_ovo': 0.0, 'cross_entropy': np.inf},
                'tune_time': 0.0
            }
            return None, dummy, 0.0, {}

        start = time.perf_counter()
        
        try:
            automl = AutoSklearnClassifier(time_left_for_this_task=120, per_run_time_limit=30)
            automl.fit(X_train, y_train)
            
            train_metrics = self.evaluate_model(automl, X_train, y_train)
            test_metrics = self.evaluate_model(automl, X_test, y_test)
            total_time = time.perf_counter() - start
            
            return automl, {'train': train_metrics, 'test': test_metrics, 'tune_time': total_time}, total_time, {}
        except Exception as e:
            print(f"      Auto-sklearn error: {e}")
            dummy = {
                'train': {'accuracy': 0.0, 'gmean': 0.0, 'auc_ovo': 0.0, 'cross_entropy': np.inf},
                'test': {'accuracy': 0.0, 'gmean': 0.0, 'auc_ovo': 0.0, 'cross_entropy': np.inf},
                'tune_time': 0.0
            }
            return None, dummy, 0.0, {}

    def run_benchmark(self):
        """Executa benchmark completo"""
        dataset_ids = self.get_openml_datasets()

        models_config = {
            'TabNet': self.train_tabnet,
            'LightGBM': self.train_lightgbm,
            'CatBoost': self.train_catboost,
            'XGBoost': self.train_xgboost,
            'AutoGluon': self.train_autogluon
        }
        
        if AUTOSKLEARN_AVAILABLE:
            models_config['Auto-sklearn'] = self.train_autosklearn

        print(f"\n{'='*80}")
        print(f"INICIANDO: {len(models_config)} modelos × {len(dataset_ids)} datasets")
        print(f"{'='*80}\n")

        for idx, dataset_id in enumerate(dataset_ids, 1):
            print(f"\n[{idx}/{len(dataset_ids)}] Dataset {dataset_id}")
            print("-"*80)

            X, y, info = self.load_and_preprocess_dataset(dataset_id)
            if X is None:
                continue

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
            )

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            for model_name, train_func in models_config.items():
                print(f"  → {model_name}...", end=" ", flush=True)
                
                try:
                    model, metrics, total_time, _ = train_func(X_train, y_train, X_test, y_test)

                    result = {
                        'dataset_id': dataset_id,
                        'dataset_name': info['name'],
                        'model': model_name,
                        'train_accuracy': metrics['train']['accuracy'],
                        'train_auc_ovo': metrics['train']['auc_ovo'],
                        'train_gmean': metrics['train']['gmean'],
                        'train_cross_entropy': metrics['train']['cross_entropy'],
                        'accuracy': metrics['test']['accuracy'],
                        'auc_ovo': metrics['test']['auc_ovo'],
                        'gmean': metrics['test']['gmean'],
                        'cross_entropy': metrics['test']['cross_entropy'],
                        'time_seconds': total_time,
                        'tune_time_seconds': metrics.get('tune_time', 0.0)
                    }

                    self.results.append(result)
                    print(f"✓ ACC={metrics['test']['accuracy']:.3f} ({total_time:.1f}s)")

                except Exception as e:
                    print(f"✗ {str(e)[:40]}")

            self.datasets_info.append(info)
            gc.collect()

        print(f"\n{'='*80}")
        print(f"✓ CONCLUÍDO: {len(self.results)} resultados")
        print(f"{'='*80}\n")

    def generate_results_dataframe(self) -> pd.DataFrame:
        """Gera DataFrame e salva CSV"""
        df = pd.DataFrame(self.results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(RESULTS_DIR, f"benchmark_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        print(f"✓ Resultados salvos: {csv_path}")
        return df

    def calculate_average_ranks(self, df: pd.DataFrame, metric: str) -> pd.Series:
        """Calcula rankings médios"""
        rankings = []
        for dataset_id in df['dataset_id'].unique():
            subset = df[df['dataset_id'] == dataset_id]
            if metric in ['cross_entropy', 'train_cross_entropy']:
                ranks = subset[metric].rank(ascending=True)
            else:
                ranks = subset[metric].rank(ascending=False)
            rankings.append(ranks)
        
        all_ranks = pd.concat(rankings)
        return all_ranks.groupby(df['model']).mean().sort_values()

    def friedman_nemenyi_test(self, df: pd.DataFrame, metric: str):
        """Testes de Friedman e Nemenyi"""
        print(f"\n{'='*80}")
        print(f"Friedman + Nemenyi: {metric.upper()}")
        print(f"{'='*80}\n")

        pivot = df.pivot(index='dataset_id', columns='model', values=metric).dropna()

        try:
            stat, p = friedmanchisquare(*[pivot[col] for col in pivot.columns])
            print(f"Friedman: stat={stat:.4f}, p={p:.6f}")

            if p < 0.05:
                print("  ✓ Diferença significativa\n")
                nemenyi = sp.posthoc_nemenyi_friedman(pivot)
                print(nemenyi)
                
                nem_path = os.path.join(RESULTS_DIR, f"nemenyi_{metric}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                nemenyi.to_csv(nem_path)
                print(f"  ✓ Salvo: {nem_path}")
            else:
                print("  ✗ Sem diferença significativa")
        except Exception as e:
            print(f"Erro: {e}")

    def plot_results(self, df: pd.DataFrame):
        """Gera visualizações"""
        metrics = ['accuracy', 'auc_ovo', 'gmean', 'cross_entropy', 'time_seconds']
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        axes = axes.ravel()

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            df.boxplot(column=metric, by='model', ax=ax)
            ax.set_title(metric.replace("_", " ").title(), fontsize=13, weight='bold')
            ax.set_xlabel('Modelo', fontsize=11)
            ax.set_ylabel(metric.replace("_", " ").title(), fontsize=11)
            plt.sca(ax)
            plt.xticks(rotation=45, ha='right')

        ax = axes[5]
        ranks = self.calculate_average_ranks(df, 'accuracy')
        ranks.plot(kind='barh', ax=ax, color='skyblue', edgecolor='navy')
        ax.set_title('Ranking Médio (Accuracy)', fontsize=13, weight='bold')
        ax.set_xlabel('Rank', fontsize=11)
        ax.invert_yaxis()

        plt.tight_layout()
        out_path = os.path.join(RESULTS_DIR, 'benchmark_plots.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Gráficos: {out_path}")

    def generate_report(self, df: pd.DataFrame):
        """Gera relatório final"""
        print(f"\n{'='*80}")
        print("RELATÓRIO FINAL")
        print(f"{'='*80}\n")

        print("Estatísticas por Modelo:\n")
        summary = df.groupby('model').agg({
            'accuracy': ['mean', 'std', 'min', 'max'],
            'auc_ovo': ['mean', 'std'],
            'gmean': ['mean', 'std'],
            'cross_entropy': ['mean', 'std'],
            'time_seconds': ['mean', 'std']
        }).round(4)
        print(summary)

        print(f"\n{'-'*80}")
        print("Rankings Médios:\n")
        for metric in ['accuracy', 'auc_ovo', 'gmean', 'cross_entropy']:
            print(f"\n{metric.upper()}:")
            ranks = self.calculate_average_ranks(df, metric)
            for model, rank in ranks.items():
                print(f"  {model:20s}: {rank:.2f}")

        self.friedman_nemenyi_test(df, 'accuracy')
        self.friedman_nemenyi_test(df, 'auc_ovo')

        print(f"\n{'='*80}")
        try:
            best = df.groupby('model')['accuracy'].mean().idxmax()
            best_acc = df.groupby('model')['accuracy'].mean().max()
            print(f"🏆 MELHOR: {best} (ACC={best_acc:.4f})")
        except:
            print("Melhor modelo: N/A")
        print(f"{'='*80}\n")


# ============================================================================
# EXECUÇÃO
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("BENCHMARK ML - LINUX/VSCODE")
    print("="*80 + "\n")
    
    # Teste rápido: 1 dataset
    benchmark = MLBenchmark(n_datasets=1, test_size=0.30, random_state=16)
    
    # Completo: 30 datasets
    # benchmark = MLBenchmark(n_datasets=30, test_size=0.30, random_state=42)
    
    print("Iniciando benchmark...\n")
    benchmark.run_benchmark()
    
    print("\nGerando análises...")
    results_df = benchmark.generate_results_dataframe()
    
    # Salvar latest
    latest_path = os.path.join(RESULTS_DIR, 'benchmark_latest.csv')
    results_df.to_csv(latest_path, index=False)
    print(f"✓ Latest: {latest_path}")
    
    print("\nPrimeiras linhas:")
    print(results_df.head(10))
    
    print("\nGerando visualizações...")
    benchmark.plot_results(results_df)
    
    print("\nGerando relatório...")
    benchmark.generate_report(results_df)
    
    print("\n" + "="*80)
    print("✅ BENCHMARK CONCLUÍDO!")
    print("="*80)
    print(f"\nArquivos gerados em: {os.path.abspath(RESULTS_DIR)}")
    print("  • benchmark_<timestamp>.csv")
    print("  • benchmark_latest.csv")
    print("  • benchmark_plots.png")
    print("  • nemenyi_*.csv")
    print("="*80)