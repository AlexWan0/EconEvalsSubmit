class RequestException(Exception):
    pass

class TokenLimitException(Exception):
    pass

class InvalidResponseError(Exception):
    pass

class RetriesExceededException(Exception):
    def __init__(self, seen_exceptions: list[Exception], num_retries: int = 0):
        self.seen_exceptions: list[Exception] = seen_exceptions
        
        message = f'Used up {num_retries} retries. Previous exceptions: {", ".join([str(e) for e in self.seen_exceptions])}'
        
        super().__init__(message)

class OutputValidationException(Exception):
    pass

class CacheMissException(Exception):
    pass
