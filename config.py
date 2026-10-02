"""
Central configuration for the Dst forecasting project.

Every script imports its settings from here, so you only ever need to change
a number in ONE place (e.g. the look-back window or the learning rate).
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Folders
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"              # downloaded OMNI2 yearly files
DATA_PROCESSED = ROOT / "data" / "processed"  # windows ready for training
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
MODELS = RESULTS / "models"

# --------------------------------------------------------------------------
# Data source: NASA OMNI2 hourly data (one plain-text file per year)
# --------------------------------------------------------------------------
OMNI_URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat"
START_YEAR = 1995   # from 1995 on, solar-wind coverage (Wind, then ACE) is good
END_YEAR = 2025

# Chronological split (inclusive year ranges). NEVER split time series randomly:
# neighbouring hours are almost identical, so a random split leaks information.
TRAIN_YEARS = (1995, 2014)   # solar cycle 23 + rise/max of cycle 24
VAL_YEARS = (2015, 2018)     # declining phase of cycle 24 (used for early stopping)
TEST_YEARS = (2019, 2025)    # cycle 25, including the May 2024 superstorm

# --------------------------------------------------------------------------
# Inputs and targets
# --------------------------------------------------------------------------
# B   : |B| magnitude of the interplanetary magnetic field (IMF) [nT]
# By  : IMF y-component, GSM [nT]
# Bz  : IMF z-component, GSM [nT]  (southward Bz drives dayside reconnection)
# V   : solar wind bulk speed [km/s]
# n   : proton density [cm^-3]
# P   : solar wind dynamic (flow) pressure [nPa]
# VBs : rectified dawn-dusk electric field V*Bs [mV/m] (Bs = -Bz if Bz<0, else 0)
# Dst : the storm index itself (its past values are an input, too)
FEATURES = ["B", "By", "Bz", "V", "n", "P", "VBs", "Dst"]
TARGET = "Dst"

LOOKBACK = 24                    # hours of history the model sees
HORIZONS = [1, 2, 3, 4, 5, 6]    # forecast Dst this many hours ahead
MAX_GAP_HOURS = 3                # linearly fill data gaps up to this length

STORM_THRESHOLD = -50.0          # nT; Dst below this = (at least) moderate storm

# --------------------------------------------------------------------------
# LSTM hyper-parameters
# --------------------------------------------------------------------------
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 60
PATIENCE = 8          # early stopping: stop after this many epochs without improvement
SEED = 42

# Ridge regression baseline
RIDGE_ALPHA = 1.0
