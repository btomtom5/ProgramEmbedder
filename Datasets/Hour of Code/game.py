from enum import Enum

class MovingException(Exception):
	pass

class OrientationException(Exception):
	pass

class MaxDepthException(Exception):
	pass

def log(*statement):
	if debug_mode:
		print(*statement)

class Orientation(Enum):
	left = 0
	right = 1
	top = 2
	bottom = 3


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

	def extract_state(self):
		return self._curr_loc, self.successful, self.failure, self._orientation, self._world, self._start_loc, self._end_loc


CodeType = Enum("CodeType", "ELSE maze_moveForward maze_turn maze_ifElse maze_forever turnLeft turnRight isPathForward isPathRight isPathLeft DO")