import sys

from simulator.world import World
from simulator.game import Game, Orientation
from data_sources import load_original_data, write_hoare_triples
from data_sources import HOC4_DEV_PATH, HOC18_DEV_PATH, HOC4_FULL_PATH, HOC18_FULL_PATH,\
    HOARE_TRIPLES_DEV_DIR, HOARE_TRIPLES_FULL_DIR


DATASET = "dev"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        DATASET = sys.argv[1]
    else:
        raise Exception("Usage Error: python3 script_name.py <data | dev_data>")

if DATASET == "dev":
    hoc4_data = load_original_data(HOC4_DEV_PATH)
    hoc18_data = load_original_data(HOC18_DEV_PATH)
    hoare_triple_path = HOARE_TRIPLES_DEV_DIR
elif DATASET == "full":
    hoc4_data = load_original_data(HOC4_FULL_PATH)
    hoc18_data = load_original_data(HOC18_FULL_PATH)
    hoare_triple_path = HOARE_TRIPLES_FULL_DIR
else:
    raise Exception('Improper argument %s passed. Use "dev" or "full".')


def create_small_world():
    small_world = World(8, 8)

    # Top Row
    small_world.set_bomb(2, 3)

    small_world.set_square(3, 3)
    small_world.set_square(4, 3)

    # second Row
    small_world.set_square(2, 4)
    small_world.set_square(3, 4)

    return small_world


def create_large_world():
    large_world = World(8, 8)

    # Set the first row
    large_world.set_square(2, 2)

    # Set the second row to all squares
    large_world.set_row(3)
    large_world.set_none(7, 3)

    # Set the third row
    large_world.set_square(3, 4)
    large_world.set_square(6, 4)

    # Set the fourth row
    large_world.set_square(3, 5)
    large_world.set_square(6, 5)

    # Set the fifth row
    large_world.set_square(5, 6)
    large_world.set_square(6, 6)

    return large_world


def create_small_game(ast):
    small_world = create_small_world()
    small_startCoords = (2, 4)
    small_endCoords = (4, 3)
    small_game = Game(small_world, small_startCoords, small_endCoords, Orientation.right, ast)
    return small_game


def create_large_game(ast):
    large_world = create_large_world()
    large_startCoords = (5, 6)
    large_endCoords = (0, 3)
    large_game = Game(large_world, large_startCoords, large_endCoords, Orientation.right, ast)
    return large_game


def batch_processor(processor_name, batch_size):
    total_games = 0
    total_successful = 0
    hoare_triples = []
    batch_number = 0
    while True:
        program_batch = yield
        if program_batch is None:
            break
        for program in program_batch:
            small_game = create_small_game(program)
            small_game.run()
            hoare_triples += small_game.get_records()

            if len(hoare_triples) > batch_size:
                write_hoare_triples(hoare_triple_path + processor_name + "/", hoare_triples[:batch_size], batch_number)
                hoare_triples = hoare_triples[batch_size:]

            total_games += 1
            total_successful += 1 if small_game.successful else 0
        batch_number += 0

    write_hoare_triples(hoare_triple_path + processor_name + "/", hoare_triples, batch_number)
    yield total_games, total_successful


def generate_datasets(processor_name, program_batches, batch_size=1000):
    # list of hoc4 json files
    processor = batch_processor(processor_name, batch_size)
    # initialize the generator
    processor.send(None)
    for program_batch in program_batches:
        processor.send(program_batch)

    total_games, total_successful = processor.send(None)

    print("============Running a set of games=============")
    print("The total number of programs: ", total_games)
    print("The total number of successes: ", total_successful)
    print("The percentage that is successful: ", total_successful / total_games)


if __name__ == "__main__":

    generate_datasets("hoc4", hoc4_data, batch_size=200)
    generate_datasets("hoc18", hoc18_data, batch_size=1000)
