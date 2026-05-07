import tyro

from ee_preds.export import ExportArgs


if __name__ == '__main__':
    tyro.cli(ExportArgs).run()
