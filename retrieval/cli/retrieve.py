from ee_retrieval.cli_tools.embed_retrieval import DWA, Sampling, main
import tyro


if __name__ == '__main__':
    main(tyro.cli(DWA | Sampling))
