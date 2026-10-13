"""Flower ServerApp for the MediFed FedAvg baseline.

The server holds no images. It builds the initial pretrained model, runs
weighted FedAvg on the complete state_dict, and writes the final global weights.
"""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from medifed.config import CHECKPOINT_DIR, LEARNING_RATE
from medifed.federated import validate_mode
from medifed.model import build_model, set_seed

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run plain FedAvg for num-server-rounds and save the final global weights."""
    num_rounds = int(context.run_config["num-server-rounds"])
    run_name = str(context.run_config["run-name"])
    seed = int(context.run_config["seed"])
    mode = str(context.run_config["mode"])
    if mode == "personalized":
        raise ValueError("personalized mode is added in Phase 4")
    validate_mode(mode, ("baseline",))

    set_seed(seed)
    model = build_model(pretrained=True)
    initial_arrays = ArrayRecord(model.state_dict())

    # Names verified from FedAvg.__init__ in flwr 1.36.0: min_train_nodes,
    # min_evaluate_nodes, min_available_nodes. Defaults are 2; all three hospitals
    # must take part in every round.
    strategy = FedAvg(
        fraction_train=1.0,
        fraction_evaluate=1.0,
        min_train_nodes=3,
        min_evaluate_nodes=3,
        min_available_nodes=3,
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=num_rounds,
        train_config=ConfigRecord({"lr": LEARNING_RATE}),
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"{run_name}_global.pt"
    torch.save(result.arrays.to_torch_state_dict(), ckpt_path)
    print(f"FEDERATED RUN FINISHED: {run_name}")
