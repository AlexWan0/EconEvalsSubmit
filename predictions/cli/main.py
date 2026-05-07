import os
os.environ['FAST_OPENAI_CACHE_DIR'] = '.cache/fast_openai_evals'

from typing import Annotated, Union
import tyro

from ee_preds.evals import RunArgs, RunFromFileArgs


def main(
        run_args: Annotated[Union[
            Annotated[RunArgs, tyro.conf.subcommand('from_cli'), tyro.conf.OmitArgPrefixes],
            Annotated[RunFromFileArgs, tyro.conf.subcommand('from_file'), tyro.conf.OmitArgPrefixes],
        ], tyro.conf.OmitSubcommandPrefixes]
    ) -> None:

    run_args.run()


if __name__ == "__main__":
    tyro.cli(main)
