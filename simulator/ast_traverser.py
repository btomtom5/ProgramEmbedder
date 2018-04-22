import json
from simulator.record import Record
from enum import Enum
from simulator.util import log, is_placeholder

from simulator.game import MovingException, MaxDepthException

global_records = set()

total_skipped = 0


class AstTraverser:

    def __init__(self, ast, game, max_depth=100):
        self._max_depth = max_depth
        self._curr_depth = 0
        self._records = []
        self._game = game
        self._ast = ast

    def _check_max_depth(self):
        self._curr_depth += 1
        if self._curr_depth > self._max_depth:
            raise MaxDepthException

    def _traverser_helper(self, ast):
        global total_skipped

        self._check_max_depth()
        precond = list(self._game.extract_state())
        ast_copy = dict(ast)

        if is_placeholder(ast):
            if "children" in ast:
                for sub_ast in ast["children"]:
                    try:
                        self._traverser_helper(sub_ast)
                    except MovingException:
                        raise

        elif ast["type"] == "turnLeft":
            self._game.turn_left()
            log("turning left: ", self._game.curr_loc, ":", self._game.orientation)

        elif ast["type"] == "turnRight":
            self._game.turn_right()
            log("turning right: ", self._game.curr_loc, ":", self._game.orientation)

        elif ast["type"] == "maze_moveForward":
            log("moving forward: ", self._game.curr_loc, ":", self._game.orientation)
            if not self._game.move_forward():
                self.failure = True
                raise MovingException('Cannot move forward from here')

        # hoc18 commands
        elif ast["type"] == "maze_ifElse":

            conditional_ast = ast["children"][0]
            do_block = ast["children"][1]
            else_block = ast["children"][2]

            if conditional_ast["type"] == "isPathForward":
                try:
                    new_x, new_y = self._game.get_forward(*self._game.curr_loc)
                    self._game.world.checkCoords(new_x, new_y)
                    self._traverser_helper(do_block)
                except AssertionError:
                    self._traverser_helper(else_block)
            elif conditional_ast["type"] == "isPathLeft":
                try:
                    new_x, new_y = self._game.get_left(*self._game.curr_loc)
                    self._game.world.checkCoords(new_x, new_y)
                    self._traverser_helper(do_block)
                except AssertionError:
                    self._traverser_helper(else_block)
            elif conditional_ast["type"] == "isPathRight":
                try:
                    new_x, new_y = self._game.get_right(*self._game.curr_loc)
                    self._game.world.checkCoords(new_x, new_y)
                    self._traverser_helper(do_block)
                except AssertionError:
                    self._traverser_helper(else_block)

        elif ast["type"] == "maze_forever":
            sub_ast = ast["children"][0]
            assert len(ast["children"]) == 1, "len of maze_forever has more than one child"
            while self._game.curr_loc != self._game.end_loc:
                self._traverser_helper(sub_ast)

        postcond = list(self._game.extract_state())

        if json.dumps(ast) + str(precond) not in global_records:
            self._records.append(Record(precond, postcond, ast_copy))
        else:
            total_skipped += 1

        global_records.add(json.dumps(ast) + str(precond))

    def traverse(self):
        self._traverser_helper(self._ast)

    def get_records(self):
        return [record.get_hoare_triple() for record in self._records]


CodeType = Enum("CodeType", "ELSE maze_moveForward maze_turn maze_ifElse maze_forever turnLeft turnRight isPathForward isPathRight isPathLeft DO")

