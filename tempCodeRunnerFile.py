    if 'XGBoost' in results:
        xgboost_model = models['XGBoost']
        xgboost_path = os.path.join("exported_models", "xgboost_model.pkl")
        joblib.dump(xgboost_model, xgboost_path, compress=3)
        print(f"XGBoost model exported to {xgboost_path}")