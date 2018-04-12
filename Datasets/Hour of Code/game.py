from enum import Enum
from ast_traverser import AstTraverser


class MovingException(Exception):
    pass


class OrientationException(Exception):
    pass


class MaxDepthException(Exception):
    pass


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
        self._ast_traverser = AstTraverser(ast, self)

        self.successful = None
        self.failure = False

    def run(self):
        try:
            self._ast_traverser.traverse()
        except (MovingException, MaxDepthException):
            self.successful = False
        else:
            self.successful = self._curr_loc == self._end_loc

    # print("total skipped: {}, total length: {}".format(total_skipped, len(global_records)))

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

        x, y = self._curr_loc

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
        return self._curr_loc, self.successful, self.failure, self._orientation, self._world._world, self._start_loc, self._end_loc

    def write_records(self, path):
        self._ast_traverser.write_records(path)
