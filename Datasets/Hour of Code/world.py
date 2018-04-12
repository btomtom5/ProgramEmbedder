class World:
	def __init__(self, width, height):
		self._world = [[None for i in range(width)] for j in range(height)]
		self.height = height
		self.width = width

	def __repr__(self):
		return self._world

	def set_square(self, x_coord, y_coord):
		self._world[y_coord][x_coord] = Square()

	def set_bomb(self, x_coord, y_coord):
		self._world[y_coord][x_coord] = Square(isBomb=True)

	def set_none(self, x_coord, y_coord):
		self._world[y_coord][x_coord] = None

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