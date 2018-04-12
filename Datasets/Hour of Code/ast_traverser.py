import json
from record import Record
from enum import Enum

global_records = set()

total_skipped = 0


class AstTraverser:

	def __init__(self, ast, game, max_depth=100):
		self._max_depth = max_depth
		self._curr_depth = 0
		self._records = []
		self._game = game

	def _check_max_depth(self):
		self._curr_depth += 1
		if self._curr_depth > self._max_depth:
			raise MaxDepthException

	def traverse(self):
		global total_skipped
		self._check_max_depth()
		precond = list(self._game.extract_state())
		ast_copy = dict(ast)

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

		postcond = list(self._game.extract_state())

		if json.dumps(ast) not in global_records:
			self._records.append(Record(precond, postcond, ast_copy))
		else:
			total_skipped += 1

		global_records.add(json.dumps(ast) + str(precond))

	def write_records(self, path):
		for i, record in enumerate(self._records):
			record.write_hoare_triple(path + "_ast" + str(i) + ".json")

def _isPlaceholder(ast):
	return (ast["type"] == "program" or ast["type"] == "statementList" or 
			ast["type"] == "maze_turn" or ast["type"] == "DO" or 
			ast["type"] == "ELSE")

CodeType = Enum("CodeType", "ELSE maze_moveForward maze_turn maze_ifElse maze_forever turnLeft turnRight isPathForward isPathRight isPathLeft DO")

