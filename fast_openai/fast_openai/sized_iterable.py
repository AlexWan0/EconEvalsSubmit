from typing import Any, TypeAlias, Protocol, Iterator, TypeVar, Iterable, Callable, runtime_checkable


IterType = TypeVar('IterType', covariant=True)

@runtime_checkable
class SizedIterable(Protocol[IterType]):
    def __iter__(self) -> Iterator[IterType]:
        ...

    def __len__(self) -> int:
        ...

def make_sized(iterable: Iterable[IterType], length: int) -> SizedIterable:
    class _Sized:
        def __iter__(self) -> Iterator[IterType]:
            yield from iterable
        
        def __len__(self) -> int:
            return length
    
    return _Sized()

IterTypeNew = TypeVar('IterTypeNew', covariant=True)
def map_sized(
        func: Callable[[IterType], IterTypeNew],
        sized_iterable: SizedIterable[IterType]
    ) -> SizedIterable[IterTypeNew]:

    class _MappedSized:
        def __iter__(self) -> Iterator[IterTypeNew]:
            for orig in sized_iterable:
                yield func(orig)

        def __len__(self) -> int:
            return len(sized_iterable)

    return _MappedSized()

MaybeSizedIter = SizedIterable[IterType] | Iterable[IterType]
def map_maybe_sized(
        func: Callable[[IterType], IterTypeNew],
        maybe_sized_iterable: MaybeSizedIter[IterType]
    ) -> MaybeSizedIter[IterTypeNew]:
    # TODO: do with isinstance instead? or just get rid of the maybesized stuff...

    # if isinstance(maybe_sized_iterable, Iterable):
    if not hasattr(maybe_sized_iterable, '__len__'):
        return map(
            func,
            maybe_sized_iterable
        )
    # elif isinstance(maybe_sized_iterable, SizedIterable):
    else:
        return map_sized(
            func,
            maybe_sized_iterable # type: ignore
        )
    # else:
    #     raise TypeError(f'`maybe_sized_iterable` must be either an Iterable or a SizedIterable')
