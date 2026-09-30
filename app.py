from flask import Flask, render_template, redirect, url_for
import subprocess
import mlflow
from mlflow.tracking import MlflowClient

app = Flask(__name__)

@app.route("/")
def index():
    runs_data = []
    try:
        client = MlflowClient()
        experiment = client.get_experiment_by_name("Iris_RandomForest_Experiment")
        if experiment:
            runs = client.search_runs(experiment.experiment_id)
            for r in runs:
                runs_data.append({
                    "run_id": r.info.run_id,
                    "status": r.info.status,
                    "accuracy": r.data.metrics.get("accuracy", "N/A"),
                    "n_estimators": r.data.params.get("n_estimators", "N/A"),
                    "max_depth": r.data.params.get("max_depth", "N/A")
                })
    except Exception as e:
        print("Error fetching mlflow runs:", e)

    return render_template("index.html", runs=runs_data)

@app.route("/train", methods=["POST"])
def train():
    subprocess.run(["python", "train_mlflow.py"], check=True)
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
