"""Errors intentionally contain no document content, signed URLs or secrets."""

class KnowledgeError(Exception):
    def __init__(self, code: str, message: str = ""):
        self.code = code
        super().__init__(message or code)


class StrategyError(KnowledgeError):
    pass
