from mlflow.tracking import MlflowClient
import mlflow, os

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
client = MlflowClient()

# Get all experiments, including deleted ones
experiments = client.search_experiments(view_type=mlflow.entities.ViewType.ALL)

target_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "getaround_rental_price_predictor")

for exp in experiments:
    if exp.name == target_name:
        print(f"Found experiment: {exp.experiment_id} (lifecycle_stage={exp.lifecycle_stage})")
        client.delete_experiment(exp.experiment_id)
        print(f"Experiment '{target_name}' permanently deleted.")
        break
else:
    print(f"No experiment found with name '{target_name}'")