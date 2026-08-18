from fastapi.responses import JSONResponse
from fastapi import Request
import groq


async def timeout_exception_handler(request: Request, exc: TimeoutError):
    return JSONResponse(
        status_code=504,
        content={"detail": "LLM request timed out"}
    )


async def bad_request_handler(request: Request, exc: groq.BadRequestError):
    return JSONResponse(
        status_code=400,
        content={"detail": "Invalid LLM request"}
    )


async def authentication_handler(request: Request, exc: groq.AuthenticationError):
    return JSONResponse(
        status_code=401,
        content={"detail": "LLM authentication failed"}
    )


async def permission_handler(request: Request, exc: groq.PermissionDeniedError):
    return JSONResponse(
        status_code=403,
        content={"detail": "LLM request forbidden"}
    )


async def not_found_handler(request: Request, exc: groq.NotFoundError):
    return JSONResponse(
        status_code=404,
        content={"detail": "LLM model not found"}
    )


async def rate_limit_handler(request: Request, exc: groq.RateLimitError):
    return JSONResponse(
        status_code=429,
        content={"detail": "LLM rate limit exceeded"}
    )


async def server_error_handler(request: Request, exc: groq.InternalServerError):
    return JSONResponse(
        status_code=502,
        content={"detail": "LLM provider temporarily unavailable"}
    )

async def connection_error_handler(request: Request, exc: groq.APIConnectionError):
    return JSONResponse(
        status_code=503,
        content={"detail": "Unable to connect to LLM provider"}
    )

async def api_timeout_handler(request: Request, exc: groq.APITimeoutError):
    return JSONResponse(
        status_code=504,
        content={"detail": "LLM provider request timed out"}
    )

async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


def register_exception_handlers(app):

    app.add_exception_handler(
        TimeoutError,
        timeout_exception_handler,
    )

    app.add_exception_handler(
        groq.BadRequestError,
        bad_request_handler,
    )

    app.add_exception_handler(
        groq.AuthenticationError,
        authentication_handler,
    )

    app.add_exception_handler(
        groq.PermissionDeniedError,
        permission_handler,
    )

    app.add_exception_handler(
        groq.NotFoundError,
        not_found_handler,
    )

    app.add_exception_handler(
        groq.RateLimitError,
        rate_limit_handler,
    )

    app.add_exception_handler(
        groq.InternalServerError,
        server_error_handler,
    )

    app.add_exception_handler(
        groq.APIConnectionError,
        connection_error_handler,
    )

    app.add_exception_handler(
        groq.APITimeoutError,
        api_timeout_handler,
    )

    app.add_exception_handler(
        Exception,
        global_exception_handler,
    )