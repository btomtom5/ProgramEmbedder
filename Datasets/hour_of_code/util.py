debug_mode = False


def log(*statement):
    if debug_mode:
        print(*statement)


def is_placeholder(ast):
    return (ast["type"] == "program" or ast["type"] == "statementList" or
            ast["type"] == "maze_turn" or ast["type"] == "DO" or
            ast["type"] == "ELSE")


def is_terminal_statement(ast):
    return (ast["type"] == "turnLeft" or ast["type"] == "turnRight" or
            ast["type"] == "maze_forward")





