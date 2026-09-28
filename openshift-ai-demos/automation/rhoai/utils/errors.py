"""Error formatting utilities.

Converts raw exceptions into short, human-readable strings suitable for
printing on the terminal without SDK tracebacks or multi-line blobs.
"""


def friendly_error(exc: Exception) -> str:
    """Return a concise, single-line error message from any exception.

    kubernetes SDK exceptions (UnauthorizedException, ForbiddenException, …)
    carry a multi-line HTTP body.  This extracts just the HTTP reason phrase
    so the user sees e.g. ``"Unauthorized (401)"`` instead of 30 lines of SDK
    internals embedded in the error message.
    """
    # kubernetes.client.exceptions.ApiException subclasses all expose .status / .reason
    reason = getattr(exc, "reason", None)
    status = getattr(exc, "status", None)
    if reason and status:
        return f"{reason} ({status})"
    if reason:
        return reason
    # Fallback: first line of str(exc) only — avoids multi-line blobs
    return str(exc).splitlines()[0]
