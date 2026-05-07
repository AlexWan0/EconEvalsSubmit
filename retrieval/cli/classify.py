from ee_retrieval.cli_tools.classify import main, Categorization, Quality
import tyro


if __name__ == '__main__':
    main(tyro.cli(Categorization | Quality))
