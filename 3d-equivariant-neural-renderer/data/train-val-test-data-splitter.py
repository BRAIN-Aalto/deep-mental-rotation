import argparse
import logging
from pathlib import Path

from sklearn.model_selection import train_test_split

# set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s %(name)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)

# set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "path_to_data",
    default=Path(__file__).absolute().parent,
    type=lambda p: Path(p).absolute(),
    help="path to data to be split"
)
parser.add_argument(
    "--split",
    default=[80, 10, 10],
    type=int,
    nargs=3,
    help="train, validation, and test proportions of the data to include in each split"
    )
parser.add_argument(
    "--seed",
    default=12345,
    type=int,
    help="random seed"
)
parser.add_argument(
    "--save_dir",
    default=Path(__file__).absolute().parent,
    type=lambda p: Path(p).absolute(),
    help="path to the saving directory for data"
    )


args = parser.parse_args()
### load shapes from the file ###
with open(args.path_to_data, "r", encoding="utf-8") as reader:
    shapes = reader.read().splitlines()


### split the generated shapes in train, validation, and test sets ###

train_size, valid_size, test_size = args.split

shapes_train, shapes_val_test = train_test_split(
    shapes,
    train_size=train_size/100,
    random_state=args.seed
)
shapes_val, shapes_test = train_test_split(
    shapes_val_test,
    train_size=valid_size/(valid_size + test_size),
    random_state=args.seed
)

### save splits with shape strings into files ####
args.save_dir.mkdir(parents=True, exist_ok=True)


def save_to_txt(data: list[str], fname: str) -> None:
    """
    """
    with open(args.save_dir / (fname + ".txt"), 'w', encoding="utf-8") as writer:
        writer.writelines(map(lambda shape: shape + "\n", data))
        logger.info(f"File saved to {args.save_dir / (fname + '.txt')}")


for split, name in zip((shapes_train, shapes_val, shapes_test), ("train", "val", "test")):
    save_to_txt(split, name)







