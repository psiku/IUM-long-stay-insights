import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler, ParameterGrid
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from typing import Dict, Any, List

from src.models.classifier_emb import BinaryClassifierEmbeddings
from src.models.train_model import train_classifier, evaluate_model
from src.data.data_loader import create_dataloader
from src.data.set_seed import set_seed
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, KFold


def randomized_search_embeddings(
    param_distributions: Dict[str, Any],
    n_iter: int,
    x_cat_train, x_num_train, y_train,
    x_cat_val, x_num_val, y_val,
    embedding_sizes: List[tuple],
    num_numeric: int,
    device: torch.device,
    num_epochs: int = 10
) -> pd.DataFrame:
    results = []
    sampler = list(ParameterSampler(param_distributions, n_iter=n_iter, random_state=42))

    for i, params in enumerate(sampler, 1):
        print(f"\n=== Trial {i}/{n_iter} — Testing params: {params}")
        set_seed(42, verbose=False)

        #dataloaders
        train_loader = create_dataloader(
            x_cat_train, y_train, x_num_train,
            batch_size=params['batch_size'], seed=42
        )
        val_loader = create_dataloader(
            x_cat_val, y_val, x_num_val,
            batch_size=params['batch_size'], seed=42
        )

        #model and criterions
        model = BinaryClassifierEmbeddings(
            num_numeric=num_numeric,
            embedding_sizes=embedding_sizes,
            output_size=2,
            hidden_sizes=params['hidden_sizes'],
            dropout=params['dropout']
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        if params['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'])

        # training
        train_classifier(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=num_epochs,
            device=device
        )

        # results
        preds, labels = evaluate_model(model, val_loader, device)
        val_acc = accuracy_score(labels, np.argmax(preds, axis=1))
        roc_auc = roc_auc_score(labels, preds[:, 1])
        avg_precision = average_precision_score(labels, preds[:, 1])

        result = {
            'lr': params['lr'],
            'batch_size': params['batch_size'],
            'optimizer': params['optimizer'],
            'dropout': params['dropout'],
            'hidden_sizes': tuple(params['hidden_sizes']),
            'val_acc': val_acc,
            'roc_auc': roc_auc,
            'avg_precision': avg_precision,
        }
        results.append(result)

    return pd.DataFrame(results)


def full_grid_search_embeddings(
    param_grid: Dict[str, Any],
    x_cat_train, x_num_train, y_train,
    x_cat_val, x_num_val, y_val,
    embedding_sizes: List[tuple],
    num_numeric: int,
    device: torch.device,
    num_epochs: int = 10
) -> pd.DataFrame:

    results = []
    grid = list(ParameterGrid(param_grid))

    for idx, params in enumerate(grid, 1):
        print(f"\n=== Grid {idx}/{len(grid)} — Testing params: {params}")
        set_seed(42, verbose=False)

        # dataloaders
        train_loader = create_dataloader(
            x_cat_train, y_train, x_num_train,
            batch_size=params['batch_size'], seed=42
        )
        val_loader = create_dataloader(
            x_cat_val, y_val, x_num_val,
            batch_size=params['batch_size'], seed=42
        )

        # model and criterion
        model = BinaryClassifierEmbeddings(
            num_numeric=num_numeric,
            embedding_sizes=embedding_sizes,
            output_size=2,
            hidden_sizes=params['hidden_sizes'],
            dropout=params['dropout']
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        if params['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'])

        # training
        train_classifier(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=num_epochs,
            device=device
        )

        # evaluation
        preds, labels = evaluate_model(model, val_loader, device)
        val_acc       = accuracy_score(labels, np.argmax(preds, axis=1))
        roc_auc       = roc_auc_score(labels, preds[:, 1])
        avg_precision = average_precision_score(labels, preds[:, 1])

        results.append({
            'lr':            params['lr'],
            'batch_size':    params['batch_size'],
            'optimizer':     params['optimizer'],
            'dropout':       params['dropout'],
            'hidden_sizes':  tuple(params['hidden_sizes']),
            'val_acc':       val_acc,
            'roc_auc':       roc_auc,
            'avg_precision': avg_precision,
        })

    return pd.DataFrame(results)


def random_search_xgb(
    param_distributions: dict,
    n_iter: int,
    x_train, y_train,
    x_test, y_test,
    cv: KFold = None,
    scoring: str = 'roc_auc',
    random_state: int = 42,
    verbose: int = 2,
    n_jobs: int = -1
):

    set_seed(random_state)

    if cv is None:
        cv = KFold(n_splits=3, shuffle=True, random_state=random_state)

    xgb = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=random_state
    )


    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        verbose=verbose,
        n_jobs=n_jobs,
        refit=True
    )

    search.fit(x_train, y_train)

    best = search.best_estimator_

    y_pred  = best.predict(x_test)
    y_proba = best.predict_proba(x_test)[:, 1]

    metrics = {
        'best_params':         search.best_params_,
        'best_cv_score':       search.best_score_,
        'test_accuracy':       accuracy_score(y_test, y_pred),
        'test_roc_auc':        roc_auc_score(y_test, y_proba),
        'test_avg_precision':  average_precision_score(y_test, y_proba),
    }

    return search, metrics