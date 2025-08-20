from mlflow.tracking import MlflowClient
import mlflow, os

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
exp_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "getaround_rental_price_predictor")
exp = mlflow.get_experiment_by_name(exp_name)

if exp is None:
    print(f"Experiment '{exp_name}' does not exist. Nothing to delete.")
else:
    MlflowClient().delete_experiment(exp.experiment_id)
    print(f"Experiment '{exp_name}' deleted permanently.")


