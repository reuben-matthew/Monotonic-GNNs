import yaml
from enum import Enum
from pathlib import Path

class EncoderType(Enum):
    CANONICAL = "canonical"
    ICLR22 = "iclr22"

class AggregationType(Enum):
    MAX = "max"
    SUM = "sum"

class ExperimentConfig:

    data_dir: Path  # Folder with the specific dataset with all training, validation, and test data.
    exp_dir: Path  # Folder where all experiment folders are stored (we will create a new experiment folder in here)
    encoding_scheme: EncoderType  # Encoding/decoding scheme (canonical or iclr22)
    num_layers: int  #  Depth of the model (1,2,3)
    agg_functions: list  # Aggregation functions per layer (max|sum), length of list must equal num_layers
    derivation_threshold: float # Model threshold for derivation; must be between 0 and 1
    use_dummies: bool  # Use dummy nodes during training (this is a training optimisation that sometimes helps)
    clamping: float  # Clamp weights whose absolute value is smaller than this to 0. [CURRENTLY UNSUPPORTED]
    non_negative_weights: bool  # Use only non-negative weights in the model's matrices.

    # TODO: factor out boilerplate (use utils.check or define a similar one)
    def __init__(self, config_path: str):

        with open(config_path) as f:
            data = yaml.safe_load(f)

        data_path = Path(data["data_dir"])
        if not data_path.is_dir():
            raise ValueError(f"data path is not an existing folder: {data_path}")
        self.data_dir = data_path

        exp_path = Path(data["exp_dir"])
        if not exp_path.is_dir():
            raise ValueError(f"experiment path is not an existing folder: {exp_path}")
        self.exp_dir = exp_path

        try:
            self.encoding_scheme = EncoderType(data["encoding_scheme"])
        except ValueError:
            valid = " or ".join(f'"{e.value}"' for e in EncoderType)
            raise ValueError(f"encoder type not valid: please choose {valid}")

        try:
            self.num_layers = int(data["num_layers"])
        except (ValueError, TypeError):
            raise ValueError(f"num_layers must be an integer, instead gor {data['num_layers']!r}")
        if self.num_layers not in [1, 2, 3]:
            raise ValueError(f"num_layers must be 1, 2, or 3, instead got {self.num_layers}")

        agg_functions = data["aggregations"]
        if not isinstance(agg_functions, list) or len(agg_functions) != self.num_layers:
            raise ValueError(f"aggregations must be a list with an entry for each layer {self.num_layers}, instead got {agg_functions!r}")
        self.agg_functions = []
        for agg in agg_functions:
            try:
                self.agg_functions.append(AggregationType(agg))
            except ValueError:
                valid = " or ".join(f'"{a_type.value}"' for a_type in AggregationType)
                raise ValueError(f"aggregation type not valid: please one of {valid}")

        try:
            self.derivation_threshold = float(data["derivation_threshold"])
        except ValueError:
            raise ValueError(f"threshold value must be a float, got {data['derivation_threshold']!r}")
        if not 0 <= self.derivation_threshold <= 1:
            raise ValueError(f"threshold value must be between 0 and 1, got {self.derivation_threshold!r}")

        try:
            self.clamping = float(data["clamping"])
        except ValueError:
            raise ValueError(f"clamping value must be a float, got {data['clamping']!r}")
        if self.clamping < 0:
            raise ValueError(f"clamping value must be non-negative, got {self.clamping!r}")

        self.use_dummies = data["use_dummy_constants"]
        self.non_negative_weights = data["non_negative_weights"]

