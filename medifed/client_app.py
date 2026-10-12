"""Flower ClientApp for one hospital in the MediFed FedAvg baseline.

Only model weights and scalar metrics leave this process. Images stay on the
hospital's local disk (NFR-01).
"""

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from medifed.config import (
    BATCH_SIZE,
    FL_LOCAL_EPOCHS,
    LEARNING_RATE,
    RUNS_DIR,
    SEED,
)
from medifed.data import make_loader
from medifed.federated import (
    append_round_row,
    local_update,
    partition_to_hospital,
    state_dict_to_cpu,
)
from medifed.model import build_model, get_device, set_seed
from medifed.train import evaluate as evaluate_loader

app = ClientApp()


def _hospital_and_run(context: Context) -> tuple[int, str]:
    """Read hospital id from the SuperNode partition-id and the run name from run config."""
    partition_id = int(context.node_config["partition-id"])
    hospital_id = partition_to_hospital(partition_id)
    run_name = str(context.run_config["run-name"])
    return hospital_id, run_name


def _rounds_csv(run_name: str, hospital_id: int):
    return RUNS_DIR / run_name / f"hospital{hospital_id}_rounds.csv"


def _load_received_model(msg: Message, device):
    """Build a non-pretrained model and load the weights sent by the server."""
    model = build_model(pretrained=False)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    return model.to(device)


def _server_round(msg: Message) -> int:
    """FedAvg injects the current round into the train/evaluate ConfigRecord as 'server-round'."""
    return int(msg.content["config"]["server-round"])


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Run FL_LOCAL_EPOCHS of local training on this hospital's train set."""
    hospital_id, run_name = _hospital_and_run(context)
    server_round = _server_round(msg)
    config = msg.content["config"]
    lr = float(config["lr"]) if "lr" in config else LEARNING_RATE

    device = get_device()
    set_seed(SEED + 1000 * hospital_id + server_round)

    model = _load_received_model(msg, device)
    train_loader = make_loader(hospital_id, "train", BATCH_SIZE, shuffle=True)
    metrics = local_update(model, train_loader, FL_LOCAL_EPOCHS, lr, device)

    num_examples = len(train_loader.dataset)
    append_round_row(
        _rounds_csv(run_name, hospital_id),
        {
            "round": server_round,
            "hospital": hospital_id,
            "split": "train",
            "loss": metrics["train_loss"],
            "accuracy": metrics["train_acc"],
            "num_samples": num_examples,
        },
    )

    return Message(
        content=RecordDict(
            {
                "arrays": ArrayRecord(state_dict_to_cpu(model)),
                "metrics": MetricRecord(
                    {
                        "train_loss": float(metrics["train_loss"]),
                        "train_acc": float(metrics["train_acc"]),
                        "num-examples": int(num_examples),
                    }
                ),
            }
        ),
        reply_to=msg,
    )


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Evaluate the received global model on this hospital's validation set."""
    hospital_id, run_name = _hospital_and_run(context)
    server_round = _server_round(msg)
    device = get_device()

    model = _load_received_model(msg, device)
    val_loader = make_loader(hospital_id, "val", BATCH_SIZE, shuffle=False)
    val_metrics = evaluate_loader(model, val_loader, device)

    append_round_row(
        _rounds_csv(run_name, hospital_id),
        {
            "round": server_round,
            "hospital": hospital_id,
            "split": "val",
            "loss": val_metrics["loss"],
            "accuracy": val_metrics["accuracy"],
            "num_samples": val_metrics["num_samples"],
        },
    )

    return Message(
        content=RecordDict(
            {
                "metrics": MetricRecord(
                    {
                        "eval_loss": float(val_metrics["loss"]),
                        "eval_acc": float(val_metrics["accuracy"]),
                        "num-examples": int(val_metrics["num_samples"]),
                    }
                ),
            }
        ),
        reply_to=msg,
    )
