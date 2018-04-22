from enum import Enum


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
        # Import here to avoid circular imports
        from simulator.ast_traverser import AstTraverser

        self.world = world
        self._start_loc = startCoords
        self.end_loc = endCoords
        self.curr_loc = startCoords
        self.orientation = orientation
        self._ast_traverser = AstTraverser(ast, self)

        self.successful = None
        self.failure = False

    def run(self):
        try:
            self._ast_traverser.traverse()
        except (MovingException, MaxDepthException):
            self.successful = False
        else:
            self.successful = self.curr_loc == self.end_loc

    # print("total skipped: {}, total length: {}".format(total_skipped, len(global_records)))

    def get_left(self, x, y):
        if self.orientation == Orientation.left:
            return x, y + 1

        elif self.orientation == Orientation.right:
            return x, y - 1

        elif self.orientation == Orientation.top:
            return x - 1, y

        elif self.orientation == Orientation.bottom:
            return x + 1, y

    def get_right(self, x, y):
        if self.orientation == Orientation.left:
            return x, y - 1

        elif self.orientation == Orientation.right:
            return x, y + 1

        elif self.orientation == Orientation.top:
            return x + 1, y

        elif self.orientation == Orientation.bottom:
            return x - 1, y

    def get_forward(self, x, y):
        if self.orientation == Orientation.left:
            return x - 1, y

        elif self.orientation == Orientation.right:
            return x + 1, y

        elif self.orientation == Orientation.top:
            return x, y - 1

        elif self.orientation == Orientation.bottom:
            return x, y + 1

        else:
            raise OrientationException('Orientation object must be provided to method')

    def move_forward(self):

        x, y = self.curr_loc

        new_x, new_y = self.get_forward(x, y)

        self.curr_loc = new_x, new_y

        return self.world.checkCoords(*self.curr_loc)

    def turn_left(self):
        if self.orientation == Orientation.left:
            self.orientation = Orientation.bottom

        elif self.orientation == Orientation.right:
            self.orientation = Orientation.top

        elif self.orientation == Orientation.top:
            self.orientation = Orientation.left

        elif self.orientation == Orientation.bottom:
            self.orientation = Orientation.right

        else:
            raise OrientationException('Orientation object must be provided to method')

    def turn_right(self):
        if self.orientation == Orientation.left:
            self.orientation = Orientation.top

        elif self.orientation == Orientation.right:
            self.orientation = Orientation.bottom

        elif self.orientation == Orientation.top:
            self.orientation = Orientation.right

        elif self.orientation == Orientation.bottom:
            self.orientation = Orientation.left

        else:
            raise OrientationException('Orientation object must be provided to method')

    def extract_state(self):
        return self.curr_loc, self.successful, self.failure, self.orientation, self.world._world, self._start_loc, self.end_loc

    def get_records(self):
        return self._ast_traverser.get_records()
