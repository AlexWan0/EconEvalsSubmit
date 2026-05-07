from ee_retrieval.utils.checksums import get_hash
import sys

if __name__ == '__main__':
    print(get_hash(sys.argv[1]))
