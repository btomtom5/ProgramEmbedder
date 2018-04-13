debug_mode = False


def log(*statement):
    if debug_mode:
        print(*statement)
