"""Central configuration for the market random process simulator."""

DEFAULT_INITIAL_PRICE = 100.0
DEFAULT_N_DAYS = 500
DEFAULT_N_BUYERS = 200
DEFAULT_N_SELLERS = 200
DEFAULT_PRICE_SIGMA = 0.03
DEFAULT_MAX_QTY = 100.0
DEFAULT_SEED = 42

DEFAULT_N_AGENTS = 500
DEFAULT_INITIAL_CASH_PER_AGENT = 20_000.0
DEFAULT_INITIAL_SHARES_PER_AGENT = 200.0
DEFAULT_AGENT_PARTICIPATION_RATE = 0.8

EXP3_DEPTH_LEVELS = [20, 50, 100, 200, 500, 1000, 5000]
EXP3_N_RUNS = 100

EXP2_WEALTH_MULTIPLIERS = [0.5, 1, 2, 5, 10]
EXP2_N_RUNS = 100
EXP2_PATHS_PER_MULTIPLIER = 20
EXP3_PATHS_PER_DEPTH = 20

ABLATION_CONSTRAINT_TYPES = {
    "buyer_cash_only": {
        "buyer_cash_constraint": True,
        "seller_inventory_constraint": False,
        "label": "Buyer cash only",
    },
    "seller_inventory_only": {
        "buyer_cash_constraint": False,
        "seller_inventory_constraint": True,
        "label": "Seller inventory only",
    },
    "both_constraints": {
        "buyer_cash_constraint": True,
        "seller_inventory_constraint": True,
        "label": "Both constraints",
    },
}

SENSITIVITY_PRICE_SIGMAS = [0.01, 0.03, 0.05, 0.08]
SENSITIVITY_MAX_QTYS = [50.0, 100.0, 200.0]
SENSITIVITY_N_DAYS = [250, 500, 1000]

BENCHMARK_N_RUNS = 100
EXTREME_CASE_N_RUNS = 80

EXP4_INFLOW_TREATMENT = 10.0
EXP4_MAIN_REGIME = "loose"
EXP4_INITIAL_CASH_REGIMES = {
    "tight": {
        "initial_cash_per_agent": 200.0,
        "initial_shares_per_agent": DEFAULT_INITIAL_SHARES_PER_AGENT,
        "label": "Tight cash",
    },
    "loose": {
        "initial_cash_per_agent": DEFAULT_INITIAL_CASH_PER_AGENT,
        "initial_shares_per_agent": DEFAULT_INITIAL_SHARES_PER_AGENT,
        "label": "Loose cash",
    },
}
EXP4_N_RUNS = 100
EXP4_PATHS_FOR_FIGURE = 50

OUTPUT_DIR = "outputs"
