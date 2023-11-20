from termcolor import cprint

def print_red(dummy):
    cprint(dummy, "red")


def print_green(dummy):
    cprint(dummy, "green")


def print_highlight(dummy):
    cprint(dummy, "magenta", "on_white")


def print_blue(dummy):
    cprint(dummy, "blue")
