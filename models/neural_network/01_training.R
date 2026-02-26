# ══════════════════════════════════════════════════════════
# Neural Network (nnet) — Training & Hyperparameter Tuning
#
# Steps:
#   1. Load training and testing data
#   2. Select features and prepare data
#   3. Train nnet with cross-validation
#   4. Evaluate on test set
#   5. Save trained model
#
# Author: Aman Arora
# ══════════════════════════════════════════════════════════

# ── Setup: Find repo root ────────────────────────────────
# This script lives in: models/neural_network/01_training.R
# Repo root is 3 levels up
get_repo_root <- function() {
  script_dir <- dirname(sys.frame(1)$ofile)
  # If running interactively, use current working directory
  if (is.null(script_dir) || script_dir == "") {
    script_dir <- getwd()
  }
  normalizePath(file.path(script_dir, "..", ".."), mustWork = FALSE)
}

REPO_ROOT <- get_repo_root()
cat("Repo root:", REPO_ROOT, "\n")

# ── Paths (relative to repo root) ────────────────────────
DATA_DIR    <- file.path(REPO_ROOT, "data", "raw")
RESULTS_DIR <- file.path(REPO_ROOT, "results", "neural_network")
FIGURES_DIR <- file.path(REPO_ROOT, "figures")

# Create output directories
dir.create(RESULTS_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FIGURES_DIR, recursive = TRUE, showWarnings = FALSE)

# Input files
TRAIN_DATA  <- file.path(DATA_DIR, "inventory", "train_data_70k.csv")
TEST_DATA   <- file.path(DATA_DIR, "inventory", "test_data_30k.csv")

# Output files
MODEL_PATH  <- file.path(RESULTS_DIR, "nnet_model_4dtsts.rds")
RESULTS_TXT <- file.path(RESULTS_DIR, "nnet_training_summary.txt")

# ── Parameters ────────────────────────────────────────────
SELECTED_FEATURES <- c("CID", "HAND", "Dist2stream_N",
                       "River_Slope", "Discharge_N")
TARGET_COL        <- "CID"
CV_FOLDS          <- 10
MAX_ITER           <- 500
RANDOM_SEED        <- 42

# ══════════════════════════════════════════════════════════
# 1. Load Libraries
# ══════════════════════════════════════════════════════════
load_libraries <- function(libraries) {
  for (lib in libraries) {
    if (!require(lib, character.only = TRUE)) {
      install.packages(lib, dependencies = TRUE)
      library(lib, character.only = TRUE)
    }
  }
}

libraries <- c(
  "raster", "dplyr", "caret", "pROC",
  "ggplot2", "doParallel"
)

load_libraries(libraries)
cat("✅ Libraries loaded\n")

# ══════════════════════════════════════════════════════════
# 2. Load and Prepare Data
# ══════════════════════════════════════════════════════════
cat("\n1. Loading data...\n")

df_train <- read.csv(TRAIN_DATA, header = TRUE)
df_test  <- read.csv(TEST_DATA, header = TRUE)

# Select features
data_train <- df_train[, SELECTED_FEATURES]
data_test  <- df_test[, SELECTED_FEATURES]

# Remove NAs
data_train <- na.omit(data_train)
data_test  <- na.omit(data_test)

# Convert target to factor
data_train[[TARGET_COL]] <- as.factor(data_train[[TARGET_COL]])
data_test[[TARGET_COL]]  <- as.factor(data_test[[TARGET_COL]])

cat("   Training samples:", nrow(data_train), "\n")
cat("   Testing samples: ", nrow(data_test), "\n")
cat("   Features:        ", paste(setdiff(SELECTED_FEATURES, TARGET_COL),
                                  collapse = ", "), "\n")

str(data_train)
str(data_test)

# ══════════════════════════════════════════════════════════
# 3. Train Neural Network
# ══════════════════════════════════════════════════════════
cat("\n2. Training Neural Network...\n")

set.seed(RANDOM_SEED)

# Define cross-validation
train_control <- trainControl(
  method = "cv",
  number = CV_FOLDS,
  savePredictions = TRUE,
  classProbs = TRUE
)

# Define hyperparameter grid
tune_grid <- expand.grid(
  size  = c(5, 10),
  decay = c(0.1, 0.5, 0.9)
)

# Train nnet model
nnet_model <- train(
  CID ~ .,
  data = data_train,
  method = "nnet",
  tuneGrid = tune_grid,
  trControl = train_control,
  trace = FALSE,
  linout = FALSE,
  maxit = MAX_ITER
)

cat("\n   Best parameters:\n")
print(nnet_model$bestTune)
cat("   Best Accuracy:", max(nnet_model$results$Accuracy), "\n")

# ══════════════════════════════════════════════════════════
# 4. Evaluate on Test Set
# ══════════════════════════════════════════════════════════
cat("\n3. Evaluating on test set...\n")

# Predict probabilities
p1_nnet <- predict(nnet_model,
                   data_test[, -which(names(data_test) == TARGET_COL)],
                   type = "prob")

# Predict classes
class_predictions <- predict(nnet_model,
                             data_test[, -which(names(data_test) == TARGET_COL)],
                             type = "raw")

# Confusion matrix
cm <- confusionMatrix(
  as.factor(class_predictions),
  as.factor(data_test[[TARGET_COL]])
)

cat("\n   Confusion Matrix:\n")
print(cm)

# ══════════════════════════════════════════════════════════
# 5. Save Model and Results
# ══════════════════════════════════════════════════════════
cat("\n4. Saving model and results...\n")

# Save model
saveRDS(nnet_model, file = MODEL_PATH)
cat("   ✅ Model saved to:", MODEL_PATH, "\n")

# Save summary
sink(RESULTS_TXT)
cat("Neural Network Training Summary\n")
cat("================================\n\n")
cat("Date:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")
cat("Parameters:\n")
cat("  CV Folds:", CV_FOLDS, "\n")
cat("  Max Iterations:", MAX_ITER, "\n")
cat("  Random Seed:", RANDOM_SEED, "\n\n")
cat("Best Hyperparameters:\n")
print(nnet_model$bestTune)
cat("\nAll Results:\n")
print(nnet_model$results)
cat("\nConfusion Matrix (Test Set):\n")
print(cm)
sink()

cat("   ✅ Summary saved to:", RESULTS_TXT, "\n")
cat("\nDone!\n")