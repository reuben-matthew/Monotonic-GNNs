import torch
from torch_geometric.data import Data, DataLoader
import argparse
from src.utils.data_parser import parse
from src.encodings.canonical import CanonicalEncoderDecoder
from src.encodings.noncanonical.identity import IdentityEncoderDecoder
from src.encodings.noncanonical.iclr22 import ICLREncoderDecoder
from src.config.config import EncoderType, ExperimentConfig
from src.model.gnn_transformation import apply_gnn_transformation
from src.utils.utils import check, load_predicates, TYPE_PRED
from src.run.train import train
from src.run.compute_metrics import compute_metrics, best_f1_threshold
from src.rule_extraction.fact_explanation import FactExplainer
from src.model.gnn_architectures import GNN
from src.model.cd_graph import CDGraph, TraceCollector
from datetime import datetime
from pathlib import Path
import shutil

parser = argparse.ArgumentParser()
# Mandatory argument
parser.add_argument("config_file", help='Path of the configuration file that controls the experiment.')
# Optional arguments
parser.add_argument("--load-model", help='Use existing model')
parser.add_argument( "--minimal", action="store_true", help="Minimise explanatory rules" )


if __name__ == "__main__":

    args = parser.parse_args()

    # TODO: handle case where no input argument is passed
    cfg = ExperimentConfig(check(args.config_file, "Configuration"))
    dd = cfg.data_dir # useful abbreviation
    ef = cfg.exp_dir / f"{dd.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}" # Experiment folder
    ef.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config_file, ef) # Copy configuration into experiment folder

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # TODO: this idea does not work; it tries to recreate a new folder, so it forgets the old model
    # Training (or Model Loading, if training is skipped)
    if args.load_model:
        print("Loading encoder and model from file...")
        load_ef = Path(args.load_model)
        # TODO: read the load_config and assert the important bits (e.g. aggregation) match.
        internal_encoder = CanonicalEncoderDecoder(check(load_ef/ 'internal_encoder.tsv', "Internal encoding"))
        if cfg.encoding_scheme == EncoderType.ICLR22:
            external_encoder = ICLREncoderDecoder(check(load_ef / 'external_encoder.tsv', "External encoding"))
        else:
            external_encoder = IdentityEncoderDecoder(check(load_ef / 'external_encoder.tsv', "External encoding"))
        model = torch.load(check(load_ef / "model.pt", "Model"), weights_only=False).to(device)
    else:
        print("Training...")

        # Set-up Encoder-Decoder
        dbp, dup = load_predicates(check(dd / "predicates.csv", "Predicates"))
        if cfg.encoding_scheme == EncoderType.ICLR22:
            external_encoder = ICLREncoderDecoder(load_from_document=None, unary_predicates=dup, binary_predicates=dbp)
        else:
            # Default external encoding is the identity, encoding each fact as itself.
            external_encoder = IdentityEncoderDecoder(load_from_document=None, unary_predicates=dup, binary_predicates=dbp)
        internal_encoder = CanonicalEncoderDecoder(load_from_document=None,
                                                   unary_predicates=external_encoder.canonical_unary_predicates,
                                                   binary_predicates=external_encoder.canonical_binary_predicates)
        external_encoder.save_to_file(ef / 'external_encoder.tsv')
        internal_encoder.save_to_file(ef / 'internal_encoder.tsv')

        # Load & encode training data
        # TODO: sanity check - warn if the training data contains any predicates out of the signature.
        graph_dataset = parse(check(dd / "train_graph.tsv", "Training graph"))
        cd_dataset = external_encoder.encode_dataset(graph_dataset, use_dummy_constants=cfg.use_dummies)
        cd_graph = internal_encoder.encode_dataset(cd_dataset)
        train_examples_dataset = parse(check(dd / "train_pos.tsv", "Training positive examples"))
        cd_train_examples = external_encoder.encode_dataset(train_examples_dataset)
        # Negative training examples for ADNI are used and only when the target predicate is set, not needed for WN18RRv1
        cd_train_neg_examples = []
        if (dd / "train_neg.tsv").exists():
            cd_train_neg_examples = external_encoder.encode_dataset(parse(dd / "train_neg.tsv"))

        model = GNN(feature_dimension=cd_graph.delta, num_edge_colours=cd_graph.col_size,
                    num_layers=cfg.num_layers, aggregations=cfg.agg_functions).to(device)

        # TODO: try this: use a non-uniform encoding where much like in the code of the original ICLR paper, we encode
        #  the query facts into the cd_graph that we use. This is expressive enough to support transitivity
        train(cfg=cfg, device=device, internal_encoder=internal_encoder, model=model, cd_graph=cd_graph,
              train_examples=cd_train_examples, train_negatives=cd_train_neg_examples, experiment_folder=ef)

    # Validation
    print("Validating...")
    valid_graph_dataset = parse(check(dd / "valid_graph.tsv", "Validation graph"))
    predictions = apply_gnn_transformation(valid_graph_dataset, external_encoder, internal_encoder, model,
                                                 cfg.derivation_threshold, device) # Ignore activations, don't save.
    compute_metrics(predictions, dd/"valid_pos.tsv", dd/"valid_neg.tsv", ef/"valid_metrics.txt")
    best_threshold, best_f1 = best_f1_threshold(ef / "valid_metrics.txt")
    print(f"Threshold selected on validation: {best_threshold} (validation F1 {best_f1:.4f})")

    # Test
    print("Testing...")
    test_graph_dataset = parse(check(dd / "test_graph.tsv", "Test graph"))
    trace = TraceCollector()
    predictions = apply_gnn_transformation(test_graph_dataset, external_encoder, internal_encoder, model,
                                                 cfg.derivation_threshold, device, trace_collector=trace)
    compute_metrics(predictions, dd/"test_pos.tsv", dd/"test_neg.tsv", ef/"test_metrics.txt")

    #Log best threshold from validation set
    with open(ef / "test_at_valid_threshold.txt", "w") as f:
        f.write("Threshold\tPrecision\tRecall\tAccuracy\tF1 Score\n")
        for line in open(ef / "test_metrics.txt"):
            parts = line.strip().split("\t")
            if len(parts) == 5 and parts[0] != "Threshold" and float(parts[0]) == best_threshold:
                f.write(line)
                break
    
    # Save output
    derivations_file = ef / "predicted_triples.tsv"
    derivations_file_scored = ef / "predicted_triples_scored.tsv"
    to_print = []
    for (s, p, o) in predictions:
        to_print.append((predictions[s, p, o], (s, p, o)))
    to_print = sorted(to_print, reverse=True)  # Print from the fact with the highest score to the least
    with open(derivations_file, 'w') as output:
        for (score, (s, p, o)) in to_print:
            output.write("{}\t{}\t{}\n".format(s, p, o))
    with open(derivations_file_scored, 'w') as output2:
        for (score, (s, p, o)) in to_print:
            output2.write("{}\t{}\t{}\t{}\n".format(s, p, o, score))
    output.close()

    # Explanation
    print("Computing prediction explanations...")
    explanations_file = ef / "explanations.txt"
    with open(explanations_file, 'w') as output:
        explain = [f for f in predictions if cfg.target_predicate is None or (f[1] == TYPE_PRED and f[2] == cfg.target_predicate)]
        explain = sorted(explain, key=lambda f: -predictions[f]) # We want highest scoring fact firstcd
        for fact in explain[:1]:  # TODO: replace magic number with parameter
            explainer = FactExplainer(device, fact, model, cfg.derivation_threshold, trace, external_encoder,
                                      internal_encoder, args.minimal)
            rule = explainer.rule
            output.write("{}\n".format(fact))
            output.write(rule + '\n')
    output.close()