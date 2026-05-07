import tyro

from ee_retrieval.cli_tools.misc import (
    LoadIWASummMap,
    LoadClassifiedDWA,
    LoadDWASummMap,
    LoadResult
)

if __name__ == '__main__':
    tyro.cli(LoadIWASummMap | LoadClassifiedDWA | LoadDWASummMap | LoadResult).run()
