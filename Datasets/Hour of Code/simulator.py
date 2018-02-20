import json
import glob
import sys

from world import World, Orientation
from writer import generate_condition, write_hoare_triple
from game import Game

import pprint

pp = pprint.PrettyPrinter(indent=4)

debug_mode = False

################################## GENERATE THE TWO WORLD INSTANCES #############################
def create_small_world():
	small_world = World(8,8)

	## Top Row
	small_world.set_bomb(2, 3)

	small_world.set_square(3, 3)
	small_world.set_square(4, 3)

	## second Row
	small_world.set_square(2, 4)
	small_world.set_square(3, 4)

	return small_world


small_world = create_small_world()
small_startCoords = (2,4)
small_endCoords = (4,3)

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

large_world = create_large_world()
large_startCoords = (5, 6)
large_endCoords = (0, 3)


def create_small_game(ast):
	small_game = Game(small_world, small_startCoords, small_endCoords, Orientation.right, ast)
	return small_game

def create_large_game(ast):
	large_game = Game(large_world, large_startCoords, large_endCoords, Orientation.right , ast)
	return large_game


################################## PROCESS THE THE TWO HOC DATASETS #############################
def process_datasets():
	## list of hoc4 json files
	hoc4_paths = glob.glob("./hoc4/asts/*.json")

	total_games = 0
	total_successful = 0

	hoare_path = "./hoare_triples/"
	dataset_count = 0

	################################## PROCESS THE HOC 4 DATASET #################################
	for program_path in hoc4_paths:
		ast = json.load(open(program_path))
		small_game = create_small_game(ast)
		precond = generate_condition(*small_game.extract_state)
		small_game.run()
		total_games += 1
		total_successful += 1 if small_game.successful else 0
		post_cond = generate_condition(*small_game.extract_state)
		write_hoare_triple(ast, precond, post_cond, hoare_path + "hoc4/" + dataset_count + ".json")
		dataset_count +=1



	print("============Running the small games=============")
	print("The total number of programs: ", total_games)
	print("The total number of successes: ", total_successful)
	print("The percentage that is successful: ", total_successful/total_games)


	## list of hoc18 json files
	hoc18_paths = glob.glob("./hoc18/asts/*.json")

	total_games = 0
	total_successful = 0

	dataset_count = 0

	################################## PROCESS THE HOC 18 DATASET #################################

	for program_path in hoc18_paths:
		ast = json.load(open(program_path))
		large_game = create_large_game(ast)
		precond = generate_condition(*large_game.extract_state)
		large_game.run()
		total_games += 1
		total_successful += 1 if large_game.successful else 0
		post_cond = generate_condition(*large_game.extract_state)
		write_hoare_triple(ast, precond, post_cond, hoare_path + "hoc18/" + dataset_count + ".json")
		dataset_count += 1

	print("============Running the large games=============")
	print("The total number of programs: ", total_games)
	print("The total number of successes: ", total_successful)
	print("The percentage that is successful: ", total_successful/total_games)


def process_file(file_name, small_world=True):
	ast = json.load(open(file_name))
	if small_world:
		world = create_small_game(ast)
	else:
		world = create_large_game(ast)

	world.run()
	print(world.successful)

if __name__ == "__main__":	
	if len(sys.argv) > 1:
		file_name = sys.argv[1]
	else:
		file_name = None

	if file_name is None:
		process_datasets()
	else:
		process_file(file_name)


