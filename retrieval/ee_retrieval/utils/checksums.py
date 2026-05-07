from hashlib import sha256
from dataclasses import dataclass
import logging
logger = logging.getLogger(__name__)

from typing import Annotated, get_origin
import inspect


@dataclass
class Checksum:
    """Checksum wrapper to be specified for file-paths in type annotations.

    In conjunction with ``check_vars``, makes sure that the file contents at the path match the annotated checksum.
    """
    hex_digest: str

    def __hash__(self) -> int: # needed for tyro?
        return hash(self.hex_digest)

def get_hash(fp: str) -> str:
    """Get the sha256 hash of a file at a given path.
    
    Args:
        fp (str): file path of file to hash
    
    Returns:
        str: sha256 hash
    """
    with open(fp, 'rb') as f_in:
        read_checksum = sha256(f_in.read()).hexdigest()
    
    return read_checksum

def _check_hash(fp: str, expect_hash: str):
    read_checksum = get_hash(fp)

    if len(read_checksum) < len(expect_hash):
        raise ValueError(f'checksum for {fp}: {read_checksum} ({len(read_checksum)}) shorter than expected; expected {expect_hash} ({len(expect_hash)})')

    if len(read_checksum) > len(expect_hash):
        read_checksum = read_checksum[:len(expect_hash)]

    if read_checksum != expect_hash:
        raise ValueError(f'checksum for {fp} doesn\'t match: {expect_hash} != {read_checksum}')

    logger.info(f'{fp} matches checksum ({expect_hash})')

def check_vars(obj: object):
    """Checks each field object class for `Checksum` annotations and strings (which corresopnd to filepaths). If one is found, checks to make sure that that filepath has the expected checksum.
    
    Args:
        obj (object): dataclass object to check
    
    Raises:
        TypeError: If the filepath has the wrong type.
        ValueError: If the calculated checksum is incorrect or is shorter than the one specified.
    """
    for field_name, field_type in inspect.get_annotations(type(obj)).items():
        if get_origin(field_type) is Annotated:
            for m in field_type.__metadata__:
                if not isinstance(m, Checksum):
                    continue

                fp = getattr(obj, field_name)

                if not isinstance(fp, str):
                    raise TypeError(f'{field_name} for {type(obj)} is not a string, got {type(fp)} instead')

                _check_hash(
                    fp,
                    m.hex_digest
                )
