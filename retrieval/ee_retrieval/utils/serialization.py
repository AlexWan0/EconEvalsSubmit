import dataclasses


def full_repr(obj: object) -> str:
    """Recursively gets the repr of a dataclass instance (i.e., output is a Python expression that produces the original dataclass instance).

    Args:
        obj (object): Input dataclass
    
    Returns:
        str: recursive repr of dataclass
    """

    cls = type(obj)
    if dataclasses.is_dataclass(obj):
        module = cls.__module__
        qual   = cls.__qualname__
        prefix = "" if module == "__main__" else module + "."
        parts = [
            f"{f.name}={full_repr(getattr(obj, f.name))}"
            for f in dataclasses.fields(obj)
        ]
        return f"{prefix}{qual}({', '.join(parts)})"
    
    else:
        return repr(obj)
