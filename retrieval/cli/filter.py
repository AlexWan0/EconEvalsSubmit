import tyro

from ee_retrieval.cli_tools.filter import main, DWADeficient


if __name__ == '__main__':
    main(tyro.cli(DWADeficient))
