from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC

def get_models(config=None):
    models = {}
    if config is None:
        from src.utils.io import load_yaml
        config = load_yaml("config/config.yaml")
    enabled = config.get("models", {}).get("enabled", [])
    mcfg = config.get("models", {})

    if "logistic_regression" in enabled:
        c = mcfg.get("logistic_regression", {})
        models["logistic_regression"] = LogisticRegression(
            solver=c.get("solver", "liblinear"),
            max_iter=c.get("max_iter", 1000),
            class_weight=c.get("class_weight", "balanced"),
            random_state=config.get("project", {}).get("seed", 42),
        )
    if "decision_tree" in enabled:
        c = mcfg.get("decision_tree", {})
        models["decision_tree"] = DecisionTreeClassifier(
            max_depth=c.get("max_depth", 10),
            min_samples_split=c.get("min_samples_split", 5),
            class_weight=c.get("class_weight", "balanced"),
            random_state=config.get("project", {}).get("seed", 42),
        )
    if "random_forest" in enabled:
        c = mcfg.get("random_forest", {})
        models["random_forest"] = RandomForestClassifier(
            n_estimators=c.get("n_estimators", 200),
            max_depth=c.get("max_depth", 15),
            min_samples_split=c.get("min_samples_split", 5),
            class_weight=c.get("class_weight", "balanced"),
            n_jobs=c.get("n_jobs", -1),
            random_state=config.get("project", {}).get("seed", 42),
        )
    if "svm" in enabled:
        c = mcfg.get("svm", {})
        models["svm"] = SVC(
            kernel=c.get("kernel", "rbf"),
            C=c.get("C", 1.0),
            probability=c.get("probability", True),
            class_weight=c.get("class_weight", "balanced"),
            random_state=config.get("project", {}).get("seed", 42),
        )
    if "gradient_boosting" in enabled:
        c = mcfg.get("gradient_boosting", {})
        models["gradient_boosting"] = GradientBoostingClassifier(
            n_estimators=c.get("n_estimators", 200),
            learning_rate=c.get("learning_rate", 0.1),
            max_depth=c.get("max_depth", 5),
            random_state=config.get("project", {}).get("seed", 42),
        )
    # Ensembles need base estimators
    if "voting_ensemble" in enabled and models:
        c = mcfg.get("voting_ensemble", {})
        estimators = [(k, v) for k, v in models.items() if k not in ("voting_ensemble", "stacking_ensemble")]
        if estimators:
            models["voting_ensemble"] = VotingClassifier(estimators=estimators, voting=c.get("voting", "soft"), weights=c.get("weights"))
    if "stacking_ensemble" in enabled and models:
        c = mcfg.get("stacking_ensemble", {})
        estimators = [(k, v) for k, v in models.items() if k not in ("voting_ensemble", "stacking_ensemble")]
        if estimators:
            final_est = models.get("logistic_regression") or LogisticRegression(max_iter=1000)
            models["stacking_ensemble"] = StackingClassifier(estimators=estimators, final_estimator=final_est, cv=c.get("cv", 5), n_jobs=config.get("training", {}).get("n_jobs", -1))
    return models
