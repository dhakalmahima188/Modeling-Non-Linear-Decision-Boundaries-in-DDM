MODEL_NAMES = [ "exponential"  ]

DT = 0.002
MAX_TIME = 3.0

FLASH_PAIRS = [
    (0.55, 0.45), (0.60, 0.40), (0.65, 0.35),
    (0.70, 0.30), (0.75, 0.25), (0.80, 0.20),
]


COHERENCES = tuple(round(p1 - p2, 2) for p1, p2 in FLASH_PAIRS)

N_TRIALS_PER_DIFF = 60

DEADLINE_TIME = 1.35

CUE = 0 # 0 speed, 1 Accuracy

HAS_DEADLINE = 1

SHAPE_PRIOR_RANGES = {
    "exponential": ("lam",),
    "sigmoid":     ("k_sig", "t_mid"),
}