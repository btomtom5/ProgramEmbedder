from enum import Enum
import json
import glob
import sys

import pprint

pp = pprint.PrettyPrinter(indent=4)

debug_mode = False

class MovingException(Exception):
	pass

class OrientationException(Exception):
	pass

class MaxDepthException(Exception):
	pass

def log(*statement):
	if debug_mode:
		print(*statement)


class World:
	def __init__(self, width, height):
		self._world = [[None for i in range(width)] for j in range(height)]
		self.height = height
		self.width = width

	def print_world(self):
		print(self._world)

	def set_square(self, x_coord, y_coord):
		self._world[y_coord][x_coord] = Square()

	def set_bomb(self, x_coord, y_coord):
		self._world[y_coord][x_coord] = Square(isBomb=True)

	def set_row(self, y_coord):
		self._world[y_coord] = [Square() for i in range(self.width)]

	def get_top_square(self, x_coord, y_coord):
		self.__check_world_bounds()

		if y_coord - 1 < 0:
			return None
		else:
			return self._world[y_coord - 1][x_coord]

	def get_bottom_square(self, x_coord, y_coord):
		self.__check_world_bounds()
		
		if y_coord + 1 >= self.height:
			return None
		else:
			return self._world[y_coord + 1][x_coord]

	def get_right_square(self, x_coord, y_coord):
		self.__check_world_bounds()
		
		if x_coord + 1 >= self.width:
			return None
		else:
			return self._world[y_coord][x_coord + 1]

	def get_left_square(self, x_coord, y_coord):
		self.__check_world_bounds()

		if x_coord - 1 < 0:
			return None
		else:
			return self._world[y_coord][x_coord - 1]

	def setSquare(self, x_coord, y_coord, square):
		self.__check_world_bounds()

		self._world[y_coord][x_coord] = square

	
	def checkCoords(self, x_coord, y_coord):
		try:
			square = self._world[y_coord][x_coord]
		except IndexError:
			square = None

		return not (square is None or square.isBomb)


	def __check_world_bounds(self):
		assert x_coord >= 0, 'x_coord cannot be negative'
		assert y_coord >= 0, 'y_coord cannot be negative'
		assert y_coord <= self.height, 'y_coord exceeds the world height'
		assert x_coord <= self.width, 'x_coord exceeds the world width'
		return


class Square:
	def __init__(self, isBomb=False):
		self.isBomb = isBomb


	def __repr__(self):
		if self.isBomb:
			return "Bomb"
		else:
			return "Square"


Orientation = Enum('Orientation', 'left right top bottom')

class Game:
	def __init__(self, world, startCoords, endCoords, orientation, ast):
		self._world = world
		self._start_loc = startCoords
		self._end_loc = endCoords
		self._curr_loc = startCoords
		self._orientation = orientation
		self._ast = ast
		self._max_depth = 100
		self._curr_depth = 0
		self.successful = None
		self.failure = False

	def _isPlaceholder(self, ast):
		return ast["type"] == "program" or ast["type"] == "statementList" or ast["type"] == "maze_turn" or ast["type"] == "DO" or ast["type"] == "ELSE"

	def run(self):
		try:
			self._ast_traverser(self._ast)
		except (MovingException, MaxDepthException):
			self.successful = False
		else:
			self.successful = self._curr_loc == self._end_loc

	def _check_max_depth(self):
		self._curr_depth += 1
		if self._curr_depth > self._max_depth:
			raise MaxDepthException

	def _ast_traverser(self, ast):
		self._check_max_depth()
		if self._isPlaceholder(ast):
			if "children" in ast:
				for sub_ast in ast["children"]:
					try:
						self._ast_traverser(sub_ast)
					except MovingException:
						raise

		elif ast["type"] == "turnLeft":
			self.turn_left()
			log("turning left: ", self._curr_loc, ":", self._orientation)

		elif ast["type"] == "turnRight":
			self.turn_right()
			log("turning right: ", self._curr_loc, ":", self._orientation)

		elif ast["type"] == "maze_moveForward":
			log("moving forward: ", self._curr_loc, ":", self._orientation)
			if not self.move_forward():
				self.failure = True
				raise MovingException('Cannot move forward from here')

		# hoc18 commands
		elif ast["type"] == "maze_ifElse":

			conditional_ast = ast["children"][0]
			do_block = ast["children"][1]
			else_block = ast["children"][2]

			if conditional_ast["type"] == "isPathForward":
				try:
					new_x, new_y = self.get_forward(*self._curr_loc)
					self._world.checkCoords(new_x, new_y)
					self._ast_traverser(do_block)
				except AssertionError:
					self._ast_traverser(else_block)
			elif conditional_ast["type"] == "isPathLeft":
				try:
					new_x, new_y = self.get_left(*self._curr_loc)
					self._world.checkCoords(new_x, new_y)
					self._ast_traverser(do_block)
				except AssertionError:
					self._ast_traverser(else_block)
			elif conditional_ast["type"] == "isPathRight":
				try:
					new_x, new_y = self.get_right(*self._curr_loc)
					self._world.checkCoords(new_x, new_y)
					self._ast_traverser(do_block)
				except AssertionError:
					self._ast_traverser(else_block)

		elif ast["type"] == "maze_forever":
			sub_ast = ast["children"][0]
			assert len(ast["children"]) == 1, "len of maze_forever has more than one child"
			while self._curr_loc != self._end_loc:
				self._ast_traverser(sub_ast)


	def get_left(self, x, y):
		if self._orientation == Orientation.left:
			return x, y + 1

		elif self._orientation == Orientation.right:
			return x, y - 1

		elif self._orientation == Orientation.top:
			return x - 1, y

		elif self._orientation == Orientation.bottom:
			return x + 1, y

	def get_right(self, x, y):
		if self._orientation == Orientation.left:
			return x, y - 1

		elif self._orientation == Orientation.right:
			return x, y + 1

		elif self._orientation == Orientation.top:
			return x + 1, y

		elif self._orientation == Orientation.bottom:
			return x - 1, y


	def get_forward(self, x, y):
		if self._orientation == Orientation.left:
			return x - 1, y

		elif self._orientation == Orientation.right:
			return x + 1, y

		elif self._orientation == Orientation.top:
			return x, y - 1

		elif self._orientation == Orientation.bottom:
			return x, y + 1

		else:
			raise OrientationException('Orientation object must be provided to method')

	def move_forward(self):

		x,y = self._curr_loc
		
		new_x, new_y = self.get_forward(x, y)

		self._curr_loc = new_x, new_y

		return self._world.checkCoords(*self._curr_loc)

	def turn_left(self):
		if self._orientation == Orientation.left:
			self._orientation = Orientation.bottom

		elif self._orientation == Orientation.right:
			self._orientation = Orientation.top

		elif self._orientation == Orientation.top:
			self._orientation = Orientation.left

		elif self._orientation == Orientation.bottom:
			self._orientation = Orientation.right

		else:
			raise OrientationException('Orientation object must be provided to method')


	def turn_right(self):
		if self._orientation == Orientation.left:
			self._orientation = Orientation.top

		elif self._orientation == Orientation.right:
			self._orientation = Orientation.bottom

		elif self._orientation == Orientation.top:
			self._orientation = Orientation.right

		elif self._orientation == Orientation.bottom:
			self._orientation = Orientation.left

		else:
			raise OrientationException('Orientation object must be provided to method')


CodeType = Enum("CodeType", "ELSE maze_moveForward maze_turn maze_ifElse maze_forever turnLeft turnRight isPathForward isPathRight isPathLeft DO")


def create_small_world():
	small_world = World(3,2)

	## Top Row
	small_world.set_bomb(0, 0)

	small_world.set_square(1, 0)
	small_world.set_square(2, 0)

	## second Row
	small_world.set_square(0, 1)
	small_world.set_square(1, 1)

	return small_world


small_world = create_small_world()
small_startCoords = (0,1)
small_endCoords = (2,0)

def create_large_world():
	large_world = World(7, 5)

	# Set the first row
	large_world.set_square(2, 0)

	# Set the second row to all squares
	large_world.set_row(1)
	
	# Set the third row
	large_world.set_square(3, 2)
	large_world.set_square(6, 2)

	# Set the fourth row
	large_world.set_square(3, 3)
	large_world.set_square(6, 3)

	# Set the fifth row
	large_world.set_square(5, 4)
	large_world.set_square(6, 4)

	return large_world

large_world = create_large_world()
large_startCoords = (5, 4)
large_endCoords = (0, 1)


def create_small_game(ast):
	small_game = Game(small_world, small_startCoords, small_endCoords, Orientation.right, ast)
	return small_game

def create_large_game(ast):
	large_game = Game(large_world, large_startCoords, large_endCoords, Orientation.right , ast)
	return large_game

def process_datasets():
	## list of hoc4 json files
	hoc4_paths = glob.glob("./hoc4/asts/*.json")

	total_games = 0
	total_successful = 0

	for program_path in hoc4_paths:
		small_game = create_small_game(json.load(open(program_path)))
		small_game.run()
		total_games += 1
		total_successful += 1 if small_game.successful else 0



	print("============Running the small games=============")
	print("The total number of programs: ", total_games)
	print("The total number of successes: ", total_successful)
	print("The percentage that is successful: ", total_successful/total_games)


	## list of hoc18 json files
	hoc18_paths = glob.glob("./hoc18/asts/*.json")

	total_games = 0
	total_successful = 0

	for program_path in hoc18_paths:
		large_game = create_large_game(json.load(open(program_path)))
		large_game.run()
		total_games += 1
		total_successful += 1 if large_game.successful else 0

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


