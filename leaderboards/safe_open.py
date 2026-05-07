from pathlib import Path
from io import BufferedIOBase


SAFE_OPEN_PARENT = Path(__file__).parent / 'data'

def safe_open(
        pth: Path | str,
        *open_args,
        verify_parent: Path = SAFE_OPEN_PARENT,
        max_parts: int | None = None,
        allow_symlinks: bool = False,
        **open_kwargs
    ) -> BufferedIOBase:
    """
    Example: path.open('r', encoding='utf-8') => safe_open(path, 'r', encoding='utf-8')
    Example: open(path_str, 'w') => safe_open(path, 'w')

    Max parts:
        Example: max_parts=1 => reject '../test.txt', 'test/test.txt'; accept 'test.txt'
    """

    if not isinstance(pth, Path):
        pth = Path(pth)

    if '..' in pth.parts or '.' in pth.parts:
        raise ValueError(f"path traversal components ('..' or '.') are not allowed: {pth}")

    if max_parts is not None:
        num_parts = len(pth.parts)

        if num_parts > max_parts:
            raise ValueError(f'too many parts for {pth}: found {num_parts}, max {max_parts}')

    if pth.is_symlink() and not allow_symlinks:
        raise ValueError(f"symlinks are not allowed for {pth}")

    try:
        pth_resolved = pth.resolve(strict=False)
        pth_resolved.relative_to(verify_parent.resolve())

    except ValueError:
        raise ValueError(f"{pth} is outside of {verify_parent}")

    return pth_resolved.open(*open_args, **open_kwargs)
