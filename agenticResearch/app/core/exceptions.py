import groq


NON_RETRYABLE_ERRORS = (
    groq.BadRequestError,          # 400
    groq.AuthenticationError,      # 401
    groq.PermissionDeniedError,    # 403
    groq.NotFoundError,            # 404
    groq.UnprocessableEntityError, # 422
)

RETRYABLE_ERRORS = (
    groq.RateLimitError,      # 429
    groq.InternalServerError, # 500, 502, 503
    groq.APIConnectionError,
    groq.APITimeoutError
)