import json


class Record:

    def __init__(self, precond, postcond, ast):
        self.precond_list = precond
        self.postcond_list = postcond
        self.ast = ast
        self._precond = generate_condition(*precond)
        self._postcond = generate_condition(*postcond)

    def write_hoare_triple(self, filepath):
        data = {}
        data["ast"] = self.ast
        data["precond"] = self._precond
        data["postcond"] = self._postcond
        with open(filepath, 'w') as outfile:
            json.dump(data, outfile)


def generate_condition(location, program_state, program_failure, agent_orientation, board, agent_start, agent_end):
    # generate the encoded location
    x, y = location
    encoded_location = [0 for i in range(len(board) * len(board[0]))]
    encoded_location[x * y] = 1

    # generate the encoded numeric program state and numeric program failure
    numeric_program_state = [0 if program_state else 1, ]
    numeric_program_failure = [1 if program_failure else 0, ]

    # generate the encoded program orientation
    encoded_orientation = [0 for i in range(4)]
    encoded_orientation[agent_orientation.value] = 1

    # generate encoded board representation
    encoded_board_representation = [convert_square_to_numeric(square) for row in board for square in row]
    flattened_encoded_board_repr = [entry for square in encoded_board_representation for entry in square]

    # generate encoded agent start
    encoded_agent_start = [0 for i in range(len(board) * len(board[0]))]
    x_start, y_start = agent_start
    encoded_agent_start[x_start * y_start] = 1

    # generate encoded agent end
    encoded_agent_end = [0 for i in range(len(board) * len(board[0]))]
    x_end, y_end = agent_end
    encoded_agent_end[x_end * y_end] = 1

    return encoded_location + numeric_program_state + numeric_program_failure + encoded_orientation + flattened_encoded_board_repr + encoded_agent_start + encoded_agent_end


def convert_square_to_numeric(square):
    encoded_square = [0 for i in range(3)]

    if square is None:
        one_hot_index = 0
    elif square.isBomb:
        one_hot_index = 1
    else:
        one_hot_index = 2

    encoded_square[one_hot_index] = 1

    return encoded_square
