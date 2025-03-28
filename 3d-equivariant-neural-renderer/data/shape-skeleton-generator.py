import logging
import argparse
from pathlib import Path
import math

import numpy as np

from shaperenderer.geometry import ShapeGenerator
from shaperenderer.utils import xrotation, yrotation, zrotation


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s %(name)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)


parser = argparse.ArgumentParser()
parser.add_argument("--seed",  default=12345, type=int, help="random seed")
parser.add_argument(
    "--save_dir",
    default=Path(__file__).absolute().parent,
    type=lambda p: Path(p).absolute(),
    help="path to the directory to save data"
)
args = parser.parse_args()

### generate all possible shapes ###
generator = ShapeGenerator(random_state=args.seed)

n_unique_shapes = 1_000_000
shapes = []  # list of unique shapes

for _ in range(n_unique_shapes):
    shape = generator.generate()
    shapes.append(str(shape))

shapes = set(shapes)
logger.info(f"Total number of shapes generated: {len(shapes)}")


def reverse_shape(shape: str) -> str:
    """Generates the same input shape by tracing it backwards starting with an opposite endpoint
    """
    def find_opp_direction(char):
        """Helper function

        Returns a reverse direction of the input direction
        """
        idx = directions.index(char) + 3
        idx = idx if idx < 6 else idx - 6

        return directions[idx]

    directions = list('rufldb')

    return "".join(map(find_opp_direction, shape[::-1]))


def change_basis(source, target, reference, transformed):
    """
    """
    if reference.index(target) == transformed.index(source):
        return transformed

    matrix = np.zeros((4, 4))
    matrix[-1, -1] = 1

    for i in range(3):
        d = transformed[i]
        axis_index, direction = (reference.index(d), 1) if reference.index(
            d) < 3 else (reference.index(d) - 3, -1)
        matrix[axis_index, i] = direction * 1

    how_much_to_rotate = 90 if abs(transformed.index(
        source) - reference.index(target)) % 3 else 180

    if how_much_to_rotate == 180:
        if target == "u" or target == "d":
            axis_to_rotate_around = 3
        elif target == "b" or target == "f":
            axis_to_rotate_around = 2
        elif target == "r" or target == "l":
            axis_to_rotate_around = 3

    else:
        vec1, vec2 = np.zeros((2, 3))
        vec1[transformed.index(source) % 3] = 1.
        vec2[reference.index(target) % 3] = 1.

        axis_to_rotate_around = np.flatnonzero(np.cross(vec1, vec2))[0] + 1

    if axis_to_rotate_around == 1:
        rotate = xrotation
    elif axis_to_rotate_around == 2:
        rotate = yrotation
    elif axis_to_rotate_around == 3:
        rotate = zrotation

    while True:
        rotated = (matrix @ rotate(math.radians(how_much_to_rotate))
                   ).astype(np.int8)

        order, sign = np.hstack(
            list(map(np.flatnonzero, rotated.T))), rotated.sum(axis=0)
        order = [order[i] + 3 if sign[i] < 0 else order[i] for i in range(3)]
        order += [i - 3 if i >= 3 else i + 3 for i in order]

        target_index = reference.index(target)
        source_index = order[target_index]

        if source_index == reference.index(source):
            break
        else:
            how_much_to_rotate *= -1

    return "".join([reference[i] for i in order])


def transform_shape(shape, target_1, target_2, reference):
    """
    """
    source_1 = shape[0]
    # transformed basis in which source points at target
    transformed = change_basis(source_1, target_1, reference, reference)

    if target_2:
        bend_index = np.flatnonzero(np.array(
            [reference.index(char) for char in shape]) - reference.index(source_1))[0]
        source_2 = shape[bend_index]

        transformed = change_basis(source_2, target_2, reference, transformed)

    return "".join([reference[transformed.index(char)] for char in shape])


unique_shapes = []  # a set of shapes for uniquely-identified objects, i.e. none of the objects can be rotated into any other objects in the set
for shape in shapes:
    same_shape = transform_shape(reverse_shape(shape), target_1="u", target_2="b", reference="rufldb")

    if same_shape not in unique_shapes:
        unique_shapes.append(shape)

try:
    # remove shape of a perfect loop, there are two of them in the set above
    unique_shapes.remove('uubbbddff')
except:
    pass

logger.info(f"Total number of unique shapes generated: {len(unique_shapes)}")


### give more random look to all the shapes ###
# note: for now each shape has its 1st elbow fixed in orientation, i.e. up and then backwards
rng = np.random.default_rng(seed=1234)

shapes_rotated = []
for shape in unique_shapes:
    # draw a direction to rotate into
    target_direction = rng.choice(list("rufldb"))
    if target_direction == "u":
        shapes_rotated.append(shape)
    else:
        shapes_rotated.append(
            transform_shape(shape, target_1=target_direction, target_2=None, reference="rufldb"))

# save generated shape strings to a file
args.save_dir.mkdir(parents=True, exist_ok=True)

with open(args.save_dir / "shapes.txt", 'w', encoding="utf-8") as writer:
    writer.writelines(map(lambda shape: shape + "\n", shapes_rotated))

logger.info(
    f"A file with shape strings successfully saved to {args.save_dir / 'shapes.txt'}"
)
