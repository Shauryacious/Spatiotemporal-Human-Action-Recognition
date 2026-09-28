class SharError(Exception):
    def __init__(self, code: str, message: str, retryable: bool = False, status: int = 422):
        super().__init__(message)
        self.code, self.message, self.retryable, self.status = code, message, retryable, status
        self.retry_after: float = 0.0

    def public(self, request_id: str = "") -> dict:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "retryable": self.retryable,
                "request_id": request_id,
            }
        }


class Cancelled(SharError):
    def __init__(self):
        super().__init__("CANCELLED", "Analysis cancelled")
