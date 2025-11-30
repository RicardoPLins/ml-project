#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark completo:
- Modelos: LightGBM, CatBoost, XGBoost, TabNet, AutoGluon, AutoSklearn (se disponíveis)
- Datasets: 30 menores do OpenML CC18
- Split: 70% train / 30% test (seed)
- Hyperparam tuning: RandomizedSearchCV (StratifiedKFold)
- Métricas: AUC OVO, Accuracy, G-Mean, Cross-Entropy, Tempo (tune+train+predict)
- Estatística: Friedman + Nemenyi (Demšar)
- Saídas: CSV, Excel com abas, PNGs (boxplots, ranking)
"""
import warnings
warnings.filterwarnings("ignore")

import time
import os
from pathlib import Path
from datetime import datetime
import gc
import json

import numpy as np
import pandas as pd
import joblib
import openml

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, confusion_matrix

from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

# Try imports for models (some may be absent)
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CAT_AVAILABLE = True
except Exception:
    CAT_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True
except Exception:
    TABNET_AVAILABLE = False

try:
    from autogluon.tabular import TabularPredictor
    AUTOGLUON_AVAILABLE = True
except Exception:
    AUTOGLUON_AVAILABLE = False

try:
    from autosklearn.classification import AutoSklearnClassifier
    AUTOSKLEARN_AVAILABLE = True
except Exception:
    AUTOSKLEARN_AVAILABLE = False

# Directories
RESULTS_DIR = Path("results")
MODELS_DIR = RESULTS_DIR / "models"
PLOTS_DIR = RESULTS_DIR / "plots"
for d in (RESULTS_DIR, MODELS_DIR, PLOTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Dataset IDs (30) - OpenML CC18 (example list)
OPENML_CC18_IDS_30 = [
    3,6,11,12,14,15,16,18,22,23,
    28,29,31,32,37,44,46,50,54,151,
    182,188,273,293,300,307,458,469,554,1049
]

# Utility metric functions
def calculate_gmean(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sensitivities = []
    for i in range(len(cm)):
        denom = cm[i].sum()
        if denom > 0:
            sensitivities.append(cm[i, i] / denom)
    if not sensitivities:
        return 0.0
    return float(np.prod(sensitivities) ** (1.0 / len(sensitivities)))

def safe_auc_ovo(y_true, y_proba):
    try:
        if y_proba is None:
            return 0.0
        # If binary and shape (n,) or (n,2) treat accordingly
        if y_proba.ndim == 1:
            return float(roc_auc_score(y_true, y_proba))
        if y_proba.ndim == 2 and y_proba.shape[1] == 2:
            return float(roc_auc_score(y_true, y_proba[:,1]))
        return float(roc_auc_score(y_true, y_proba, multi_class='ovo', average='macro'))
    except Exception:
        return 0.0

def evaluate_preds(y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    gmean = calculate_gmean(y_true, y_pred)
    auc = safe_auc_ovo(y_true, y_proba)
    try:
        ce = float(log_loss(y_true, y_proba))
    except Exception:
        ce = float("inf")
    return {'accuracy': float(acc), 'gmean': float(gmean), 'auc_ovo': float(auc), 'cross_entropy': float(ce)}

# Benchmark class
class FullBenchmark:
    def __init__(self, dataset_ids=None, n_datasets=30, seed=42, test_size=0.30, cv_folds=5, n_iter_search=12):
        self.seed = seed
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.n_iter_search = n_iter_search
        self.dataset_ids = (dataset_ids if dataset_ids is not None else OPENML_CC18_IDS_30[:n_datasets])
        self.results = []  # dicts
        self.datasets_info = []

    def load_and_preprocess(self, dataset_id):
        try:
            ds = openml.datasets.get_dataset(dataset_id)
            X, y, _, _ = ds.get_data(dataset_format="array", target=ds.default_target_attribute)
            X = np.array(X)
            y = np.array(y)
            # encode labels
            le = LabelEncoder()
            y = le.fit_transform(y)
            # impute if needed
            if np.isnan(X).any():
                imputer = SimpleImputer(strategy='mean')
                X = imputer.fit_transform(X)
            info = {'id': dataset_id, 'name': ds.name, 'n_samples': X.shape[0], 'n_features': X.shape[1], 'n_classes': len(np.unique(y))}
            return X, y, info
        except Exception as e:
            print(f"  ✗ Erro carregando {dataset_id}: {e}")
            return None, None, None

    def _random_search(self, estimator, param_dist, X_train, y_train, n_iter=None):
        if n_iter is None:
            n_iter = self.n_iter_search
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
        rs = RandomizedSearchCV(estimator=estimator, param_distributions=param_dist, n_iter=n_iter,
                                cv=cv, scoring='accuracy', n_jobs=-1, random_state=self.seed, verbose=0)
        t0 = time.perf_counter()
        rs.fit(X_train, y_train)
        tune_time = time.perf_counter() - t0
        return rs.best_estimator_, tune_time, rs

    # TRAIN functions
    def train_lightgbm(self, X_train, y_train, X_test, y_test):
        if not LGB_AVAILABLE:
            return None, self._dummy(), 0.0
        param_dist = {'num_leaves':[20,31,50,80], 'learning_rate':[0.01,0.05,0.1], 'n_estimators':[50,100,200]}
        try:
            best, tune_time, _ = self._random_search(lgb.LGBMClassifier(random_state=self.seed), param_dist, X_train, y_train)
        except Exception:
            t0 = time.perf_counter(); best = lgb.LGBMClassifier(random_state=self.seed).fit(X_train, y_train); tune_time = time.perf_counter()-t0
        t0 = time.perf_counter()
        y_pred_train = best.predict(X_train); y_proba_train = best.predict_proba(X_train) if hasattr(best,"predict_proba") else None
        y_pred_test = best.predict(X_test); y_proba_test = best.predict_proba(X_test) if hasattr(best,"predict_proba") else None
        tp_time = time.perf_counter() - t0
        train_m = evaluate_preds(y_train, y_pred_train, y_proba_train)
        test_m = evaluate_preds(y_test, y_pred_test, y_proba_test)
        return best, {'train':train_m, 'test':test_m, 'tune_time':tune_time}, tune_time+tp_time

    def train_catboost(self, X_train, y_train, X_test, y_test):
        if not CAT_AVAILABLE:
            return None, self._dummy(), 0.0
        param_dist = {'depth':[4,6,8], 'learning_rate':[0.01,0.05,0.1], 'iterations':[100,200]}
        try:
            best, tune_time, _ = self._random_search(CatBoostClassifier(verbose=0, random_state=self.seed), param_dist, X_train, y_train)
        except Exception:
            t0 = time.perf_counter(); best = CatBoostClassifier(verbose=0, random_state=self.seed).fit(X_train, y_train); tune_time = time.perf_counter()-t0
        t0 = time.perf_counter()
        y_pred_train = best.predict(X_train); y_proba_train = best.predict_proba(X_train) if hasattr(best,"predict_proba") else None
        y_pred_test = best.predict(X_test); y_proba_test = best.predict_proba(X_test) if hasattr(best,"predict_proba") else None
        tp_time = time.perf_counter() - t0
        train_m = evaluate_preds(y_train, y_pred_train, y_proba_train)
        test_m = evaluate_preds(y_test, y_pred_test, y_proba_test)
        return best, {'train':train_m, 'test':test_m, 'tune_time':tune_time}, tune_time+tp_time

    def train_xgboost(self, X_train, y_train, X_test, y_test):
        if not XGB_AVAILABLE:
            return None, self._dummy(), 0.0
        param_dist = {'max_depth':[3,4,6], 'learning_rate':[0.01,0.05,0.1], 'n_estimators':[50,100,200]}
        try:
            best, tune_time, _ = self._random_search(xgb.XGBClassifier(eval_metric='logloss', random_state=self.seed), param_dist, X_train, y_train)
        except Exception:
            t0 = time.perf_counter(); best = xgb.XGBClassifier(eval_metric='logloss', random_state=self.seed).fit(X_train, y_train); tune_time = time.perf_counter()-t0
        t0 = time.perf_counter()
        y_pred_train = best.predict(X_train); y_proba_train = best.predict_proba(X_train) if hasattr(best,"predict_proba") else None
        y_pred_test = best.predict(X_test); y_proba_test = best.predict_proba(X_test) if hasattr(best,"predict_proba") else None
        tp_time = time.perf_counter() - t0
        train_m = evaluate_preds(y_train, y_pred_train, y_proba_train)
        test_m = evaluate_preds(y_test, y_pred_test, y_proba_test)
        return best, {'train':train_m, 'test':test_m, 'tune_time':tune_time}, tune_time+tp_time

    def train_tabnet(self, X_train, y_train, X_test, y_test):
        if not TABNET_AVAILABLE:
            return None, self._dummy(), 0.0
        param_grid = [
            {'n_d':8,'n_a':8,'n_steps':3,'gamma':1.3},
            {'n_d':16,'n_a':16,'n_steps':5,'gamma':1.5},
            {'n_d':32,'n_a':32,'n_steps':4,'gamma':1.2},
        ]
        best_score = -np.inf; best_model = None; tune_time = 0.0
        for p in param_grid:
            t0 = time.perf_counter()
            model = TabNetClassifier(**p, seed=self.seed, verbose=0)
            try:
                model.fit(X_train, y_train, eval_set=[(X_test, y_test)], max_epochs=100, patience=20, batch_size=256, virtual_batch_size=128)
            except Exception as e:
                print(f"    TabNet fit error: {e}")
            tune_time += time.perf_counter() - t0
            try:
                score = accuracy_score(y_test, model.predict(X_test))
            except Exception:
                score = -np.inf
            if score > best_score:
                best_score, best_model = score, model
        if best_model is None:
            return None, self._dummy(), tune_time
        t0 = time.perf_counter()
        y_pred_train = best_model.predict(X_train); y_proba_train = best_model.predict_proba(X_train) if hasattr(best_model,"predict_proba") else None
        y_pred_test = best_model.predict(X_test); y_proba_test = best_model.predict_proba(X_test) if hasattr(best_model,"predict_proba") else None
        tp_time = time.perf_counter() - t0
        train_m = evaluate_preds(y_train, y_pred_train, y_proba_train)
        test_m = evaluate_preds(y_test, y_pred_test, y_proba_test)
        return best_model, {'train':train_m, 'test':test_m, 'tune_time':tune_time}, tune_time+tp_time

    def train_autogluon(self, X_train, y_train, X_test, y_test, time_limit=120):
        if not AUTOGLUON_AVAILABLE:
            return None, self._dummy(), 0.0
        cols = [f"f{i}" for i in range(X_train.shape[1])]
        train_df = pd.DataFrame(X_train, columns=cols); train_df['target']=y_train
        test_df = pd.DataFrame(X_test, columns=cols); test_df['target']=y_test
        t0 = time.perf_counter()
        try:
            predictor = TabularPredictor(label='target', verbosity=0).fit(train_df, time_limit=time_limit, presets='best_quality')
            tune_time = time.perf_counter() - t0
            t1 = time.perf_counter()
            y_pred_train = predictor.predict(train_df.drop(columns=['target'])).values
            y_proba_train = predictor.predict_proba(train_df.drop(columns=['target'])).values
            y_pred_test = predictor.predict(test_df.drop(columns=['target'])).values
            y_proba_test = predictor.predict_proba(test_df.drop(columns=['target'])).values
            tp_time = time.perf_counter() - t1
            train_m = evaluate_preds(y_train, y_pred_train, y_proba_train)
            test_m = evaluate_preds(y_test, y_pred_test, y_proba_test)
            return predictor, {'train':train_m, 'test':test_m, 'tune_time':tune_time}, tune_time+tp_time
        except Exception as e:
            print(f"    AutoGluon error: {e}")
            return None, self._dummy(), 0.0

    def train_autosklearn(self, X_train, y_train, X_test, y_test, time_left=120):
        if not AUTOSKLEARN_AVAILABLE:
            return None, self._dummy(), 0.0
        t0 = time.perf_counter()
        try:
            automl = AutoSklearnClassifier(time_left_for_this_task=time_left, per_run_time_limit=max(10, int(time_left/4)), n_jobs=1, memory_limit=9000)
            automl.fit(X_train, y_train)
            tune_time = time.perf_counter() - t0
            t1 = time.perf_counter()
            y_pred_train = automl.predict(X_train); y_proba_train = automl.predict_proba(X_train) if hasattr(automl,"predict_proba") else None
            y_pred_test = automl.predict(X_test); y_proba_test = automl.predict_proba(X_test) if hasattr(automl,"predict_proba") else None
            tp_time = time.perf_counter() - t1
            train_m = evaluate_preds(y_train, y_pred_train, y_proba_train)
            test_m = evaluate_preds(y_test, y_pred_test, y_proba_test)
            return automl, {'train':train_m, 'test':test_m, 'tune_time':tune_time}, tune_time+tp_time
        except Exception as e:
            print(f"    AutoSklearn error: {e}")
            return None, self._dummy(), 0.0

    def _dummy(self):
        return {'train':{'accuracy':0.0,'gmean':0.0,'auc_ovo':0.0,'cross_entropy':float('inf')},
                'test': {'accuracy':0.0,'gmean':0.0,'auc_ovo':0.0,'cross_entropy':float('inf')},
                'tune_time':0.0}

    # Run benchmark
    def run(self):
        model_funcs = [
            ("LightGBM", self.train_lightgbm),
            ("CatBoost", self.train_catboost),
            ("XGBoost", self.train_xgboost),
            ("TabNet", self.train_tabnet),
            ("AutoGluon", self.train_autogluon),
        ]
        if AUTOSKLEARN_AVAILABLE:
            model_funcs.append(("AutoSklearn", self.train_autosklearn))

        for idx, dsid in enumerate(self.dataset_ids, start=1):
            print(f"\n[{idx}/{len(self.dataset_ids)}] Dataset {dsid}")
            X, y, info = self.load_and_preprocess(dsid)
            if X is None:
                continue
            self.datasets_info.append(info)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.seed, stratify=y)
            # scale
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            for model_name, func in model_funcs:
                print(f"  → {model_name}...", end=" ", flush=True)
                try:
                    if model_name == "AutoGluon":
                        model_obj, metrics, total_time = func(X_train, y_train, X_test, y_test, time_limit=120)
                    elif model_name == "AutoSklearn":
                        model_obj, metrics, total_time = func(X_train, y_train, X_test, y_test, time_left=120)
                    else:
                        model_obj, metrics, total_time = func(X_train, y_train, X_test, y_test)
                    row = {
                        'dataset_id': dsid,
                        'dataset_name': info['name'],
                        'model': model_name,
                        'train_accuracy': metrics['train']['accuracy'],
                        'train_auc_ovo': metrics['train']['auc_ovo'],
                        'train_gmean': metrics['train']['gmean'],
                        'train_cross_entropy': metrics['train']['cross_entropy'],
                        'test_accuracy': metrics['test']['accuracy'],
                        'test_auc_ovo': metrics['test']['auc_ovo'],
                        'test_gmean': metrics['test']['gmean'],
                        'test_cross_entropy': metrics['test']['cross_entropy'],
                        'time_seconds': float(total_time),
                        'tune_time_seconds': float(metrics.get('tune_time', 0.0))
                    }
                    self.results.append(row)
                    print(f"✓ ACC={row['test_accuracy']:.3f} (time={row['time_seconds']:.1f}s)")
                    # try save model
                    try:
                        model_fname = MODELS_DIR / f"{info['name']}_{model_name}_ds{dsid}.joblib"
                        joblib.dump(model_obj, model_fname)
                    except Exception:
                        pass
                except Exception as e:
                    print(f"✗ Erro {e}")
            # partial save
            self._save_partial()
            gc.collect()
        # final save + stats
        self._save_results()
        df = pd.DataFrame(self.results)
        if not df.empty:
            self.generate_reports_and_excel(df)
        print("\n✓ FINISHED")

    def _save_partial(self):
        if self.results:
            pd.DataFrame(self.results).to_csv(RESULTS_DIR / "benchmark_partial.csv", index=False)

    def _save_results(self):
        df = pd.DataFrame(self.results)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(RESULTS_DIR / f"benchmark_{ts}.csv", index=False)
        df.to_csv(RESULTS_DIR / "benchmark_latest.csv", index=False)
        print(f"✓ Results saved to {RESULTS_DIR}")

    # Stats & Excel
    def generate_reports_and_excel(self, df: pd.DataFrame):
        # basic summary
        summary = df.groupby('model').agg({
            'test_accuracy': ['mean','std','min','max'],
            'test_auc_ovo': ['mean','std'],
            'test_gmean': ['mean','std'],
            'test_cross_entropy': ['mean','std'],
            'time_seconds': ['mean','std']
        }).round(4)
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]

        # average ranks per dataset (Demšar): for accuracy
        def average_ranks(df, metric, higher_is_better=True):
            ranks = []
            for ds in df['dataset_id'].unique():
                sub = df[df['dataset_id']==ds]
                if higher_is_better:
                    r = sub[metric].rank(ascending=False, method='average')
                else:
                    r = sub[metric].rank(ascending=True, method='average')
                # align index to model names
                r.index = sub['model'].values
                ranks.append(r)
            allr = pd.concat(ranks, axis=1).T
            mean_r = allr.groupby(allr.index).mean().iloc[0] if False else allr.mean()
            # the above logic is messy in concat; simpler approach:
            # compute per-dataset ranks properly:
            rank_list = []
            for ds in df['dataset_id'].unique():
                sub = df[df['dataset_id']==ds].set_index('model')
                if higher_is_better:
                    r = sub[metric].rank(ascending=False, method='average')
                else:
                    r = sub[metric].rank(ascending=True, method='average')
                rank_list.append(r)
            rank_df = pd.concat(rank_list, axis=1).T
            mean_rank = rank_df.mean().sort_values()
            return mean_rank, rank_df

        # For accuracy (higher better)
        mean_rank_acc, rank_details_acc = average_ranks(df, 'test_accuracy', higher_is_better=True)

        # Friedman + Nemenyi for accuracy
        pivot_acc = df.pivot(index='dataset_id', columns='model', values='test_accuracy').dropna()
        friedman_stat, friedman_p = None, None
        nemenyi_matrix = None
        if pivot_acc.shape[1] >= 2 and pivot_acc.shape[0] >= 2:
            try:
                friedman_stat, friedman_p = friedmanchisquare(*[pivot_acc[col] for col in pivot_acc.columns])
            except Exception as e:
                friedman_stat, friedman_p = None, None
            try:
                nemenyi_matrix = sp.posthoc_nemenyi_friedman(pivot_acc)
            except Exception as e:
                nemenyi_matrix = None

        # Save Excel with multiple sheets
        excel_path = RESULTS_DIR / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='results_raw', index=False)
            summary.to_excel(writer, sheet_name='summary')
            mean_rank_acc.rename("mean_rank").to_frame().to_excel(writer, sheet_name='mean_ranks_accuracy')
            rank_details_acc.to_excel(writer, sheet_name='ranks_per_dataset_accuracy')
            if nemenyi_matrix is not None:
                nemenyi_matrix.to_excel(writer, sheet_name='nemenyi_matrix')
            # meta info
            info_df = pd.DataFrame([{
                'n_datasets': df['dataset_id'].nunique(),
                'models': ','.join(sorted(df['model'].unique())),
                'friedman_stat': friedman_stat,
                'friedman_p': friedman_p,
                'autosklearn_available': AUTOSKLEARN_AVAILABLE,
                'autogluon_available': AUTOGLUON_AVAILABLE,
                'tabnet_available': TABNET_AVAILABLE
            }])
            info_df.to_excel(writer, sheet_name='meta', index=False)
        print(f"✓ Excel report saved: {excel_path}")

        # Plots: boxplots and mean rank barh
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'figure.max_open_warning': 0})

        metrics_to_plot = ['test_accuracy','test_auc_ovo','test_gmean','test_cross_entropy','time_seconds']
        for metric in metrics_to_plot:
            try:
                fig, ax = plt.subplots(figsize=(10,6))
                df.boxplot(column=metric, by='model', ax=ax)
                ax.set_title(metric.replace('_',' ').title())
                ax.set_xlabel('Model')
                ax.set_ylabel(metric.replace('_',' ').title())
                plt.suptitle('')
                plt.xticks(rotation=45, ha='right')
                fn = PLOTS_DIR / f"boxplot_{metric}.png"
                plt.tight_layout()
                plt.savefig(fn, dpi=300, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                print(f"  ✗ plot {metric} failed: {e}")

        # mean rank plot
        try:
            fig, ax = plt.subplots(figsize=(8,6))
            mean_rank_acc.sort_values().plot(kind='barh', ax=ax)
            ax.set_title('Mean Rank (Accuracy) - lower is better')
            ax.set_xlabel('Mean Rank')
            plt.tight_layout()
            fn = PLOTS_DIR / "mean_rank_accuracy.png"
            plt.savefig(fn, dpi=300, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"  ✗ mean rank plot failed: {e}")

        # Save nemenyi csv separately if exists
        if nemenyi_matrix is not None:
            nm_path = RESULTS_DIR / "nemenyi_matrix.csv"
            nemenyi_matrix.to_csv(nm_path)
            print(f"✓ Nemenyi matrix saved: {nm_path}")
        else:
            print(f"Not nemnyi matrix")

        print("✓ Reports + plots generated")

# Run if called
if __name__ == "__main__":
    bm = FullBenchmark(n_datasets=10, seed=16, test_size=0.30, cv_folds=5, n_iter_search=12)
    bm.run()
