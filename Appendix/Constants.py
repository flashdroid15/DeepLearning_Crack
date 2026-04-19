CLASS_TO_LABEL = {"Negative": 0, "Positive": 1}
LABEL_TO_CLASS = {label: name for name, label in CLASS_TO_LABEL.items()}
DISPLAY_NAMES = {0: "Uncracked", 1: "Cracked"}

# ==========================================================

SEED = 42
IMAGE_SIZE = 227
BATCH_SIZE = 64
NUM_WORKERS = 0  # safer for Windows notebooks; raise if multiprocessing works reliably on your machine
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
NUM_EPOCHS = 15
EARLY_STOPPING_PATIENCE = 4
MAX_GRAD_NORM = 1.0
USE_WEIGHTED_SAMPLER = True
RUN_HYPERPARAM_SEARCH = True
RUN_REFERENCE_BENCHMARK = False
USE_PRETRAINED_REFERENCE = True
SAVE_PLOTS = True

# IMAGENET_MEAN = (0.485, 0.456, 0.406)
# IMAGENET_STD = (0.229, 0.224, 0.225)

# ==========================================================

BASE_CONFIG: dict[str, str | float | bool] = {
    "model_name": "custom_cnn",
    "learning_rate": 7e-4,
    "weight_decay": 1e-4,
    "dropout": 0.35,
    "use_weighted_sampler": USE_WEIGHTED_SAMPLER,
}

HYPERPARAM_CANDIDATES: list[dict[str, float | bool]] = [
    {"learning_rate": 1e-3, "weight_decay": 1e-4, "dropout": 0.25, "use_weighted_sampler": True},
    {"learning_rate": 7e-4, "weight_decay": 1e-4, "dropout": 0.35, "use_weighted_sampler": True},
    {"learning_rate": 5e-4, "weight_decay": 5e-4, "dropout": 0.45, "use_weighted_sampler": True},
]

TUNING_TRAIN_FRACTION = 0.75
TUNING_VAL_FRACTION = 0.75
TUNING_EPOCHS = 3