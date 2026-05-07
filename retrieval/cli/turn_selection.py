import tyro

from ee_retrieval.cli_tools.turn_selection import SelectTurns


if __name__ == '__main__':
    tyro.cli(SelectTurns).run()
