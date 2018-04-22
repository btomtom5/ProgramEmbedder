# Given an ast, return a numpy array that is seq_length x statement_dim
# AST is a dictionary of values 
import numpy as np
from simulator.util import is_placeholder, is_terminal_statement
from simulator.tokens import TOKENS_ENUM, NUM_TOKENS

# ast tokenizer is meant to go through the tree dfs
# if else | while loop | for loop |
# The set of numerical transformations?

# TODO set this with the real values or it will break.
MAX_SEQUENCE_LENGTH, STATEMENT_DIMENSION = 100, 20


def ast_tokenizer(ast):
    # If non-terminal
    token_list = []
    if is_placeholder(ast):
        token_list.append("begin_" + ast["type"])
        if "children" in ast:
            for sub_ast in ast["children"]:
                token_list += ast_tokenizer(sub_ast)
        token_list.append("end_" + ast["type"])

    elif is_terminal_statement(ast):
        token_list.append(ast["type"])

    elif ast["type"] == "maze_ifElse":
        conditional_ast = ast["children"][0]
        do_block = ast["children"][1]
        else_block = ast["children"][2]

        token_list.append("begin_" + ast["type"])
        token_list.append(conditional_ast["type"])

        token_list += ast_tokenizer(do_block)
        token_list += ast_tokenizer(else_block)

        token_list.append("end_" + ast["type"])

    elif ast["type"] == "maze_forever":
        sub_ast = ast["children"][0]
        assert len(ast["children"]) == 1, "len of maze_forever has more than one child"

        token_list += ast_tokenizer(sub_ast)
    return token_list


def vectorize_token_list(ast):
    token_list = ast_tokenizer(ast)
    return np.array([vectorize_token(token) for token in token_list])



def vectorize_token(token):
    init_one_hot = [0 for i in range(NUM_TOKENS)]
    init_one_hot[TOKENS_ENUM[token]] = 1
    return init_one_hot


def test_tokenizer(filepath):
    # TODO add some guarantees about the conversion from token to vector.
    with open(filepath, "r") as file:
        ast = json.load(file)
        token_list = ast_tokenizer(ast)

    return token_list


if __name__ == "__main__":
    import json

    test_files = ["Datasets/hour_of_code/data/hoc4/asts/0.json", "Datasets/hour_of_code/data/hoc18/asts/0.json"]

    for test_file in test_files:
        test_tokenizer(test_file)