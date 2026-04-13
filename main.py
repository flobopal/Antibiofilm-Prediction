import argparse
from pathlib import Path
import torch
import pandas as pd

from script.utils.molecular_descriptors import get_descriptors
from script.utils.data_load import data_load
from model.full_model import FullModel

def parse_descriptors(args: argparse.Namespace):
    get_descriptors(args.input, args.smiles_column, args.output)

def parse_data(args: argparse.Namespace):
    return data_load(
        args.input,
        features_start= args.features_start,
        features_end= args.features_end,
        normalizer_path=args.normalizer_path,
        normalizer_start=args.normalizer_start,
        normalizer_end=args.normalizer_end,
        organism_encoder_path=args.organism_encoder_path,
        organism_column=args.organism_column,
    )

def parse_prediction(args: argparse.Namespace):
    model = FullModel.load(args.model_checkpoint_path)
    Xd, Xp, _ = parse_data(args)
    model.eval()
    with torch.no_grad():
        y_pred = model.forward(Xd, Xp)
    df = pd.read_csv(args.input)
    df['prediction'] = y_pred
    df.to_csv(args.output)
    

parser = argparse.ArgumentParser(
    prog="Antibiofilm Prediction",
    description="""Implementation of the model for the paper
        \"Species-Context Aware Quantitative Prediction of the ¡
        Antibiofilm Activity of Small Molecules\"""",
    epilog="Please cite us: [Cite will be included here]"
)

subparsers = parser.add_subparsers(
    help="Commands"
)

descriptors_parser = subparsers.add_parser(
    "descriptors",
    help="Script to calculate RDKit molecular descriptors"
)

descriptors_parser.set_defaults(
    func=parse_descriptors
)

descriptors_parser.add_argument(
    "input",
    help="Route to the csv file with the smiles",
)

descriptors_parser.add_argument(
    "smiles_column",
    help="Name of the column containing molecular smiles"
)

descriptors_parser.add_argument(
    "output",
    help="Route to the output csv file",
)

predict_parser = subparsers.add_parser(
    "prediction",
    help="Script to perform predictions. See documentation about how to prepare your data"
)

predict_parser.set_defaults(
    func = parse_prediction
)

predict_parser.add_argument(
    "input",
    help="Route to the csv file with the smiles",
)

predict_parser.add_argument(
    "organism_column",
    help="Name of the column containging output variable"
)

predict_parser.add_argument(
    "output",
    help="Name of the output csv file"
)

predict_parser.add_argument(
    "--features_start",
    type=int,
    default=3,
    help="Index of the first features column"
)

predict_parser.add_argument(
    "--features_end",
    type=int,
    default=None,
    help="Index of the first column that does not contain features"
)

predict_parser.add_argument(
    "--normalizer_start",
    type=int,
    default=766
)

predict_parser.add_argument(
    "--normalizer_end",
    type=int,
    default=2000
)

predict_parser.add_argument(
    "--organism_encode_path",
    default=Path("antibiofilm checkpoint/encoder.pkl"),
    help="path to the organism encoder"
)

predict_parser.add_argument(
    "--normalizer_path",
    default=Path("antibiofilm checkpoint/normalizer.pkl"),
    help="path to the normalizer"

)

predict_parser.add_argument(
    "--model_checkpoint_path",
    default=Path("antibiofilm checkpoint/enantibiofilm_model.pth"),
    help="path to the model checkpoint"
)

args = parser.parse_args()

args.func(args)