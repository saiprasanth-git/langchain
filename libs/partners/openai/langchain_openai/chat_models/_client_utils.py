"""Helpers for creating OpenAI API clients.

This module allows for the caching of httpx clients to avoid creating new instances
for each instance of ChatOpenAI.

Logic is largely replicated from openai._base_client.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import weakref
from collections.abc import Awaitable, Callable
from functools import lru_cache
from typing import Any, cast

import httpx
import openai
from pydantic import SecretStr


class _SyncHttpxClientWrapper(openai.DefaultHttpxClient):
    """Borrowed from openai._base_client."""

    def __del__(self) -> None:
        if self.is_closed:
            return

        try:
            self.close()
        except Exception:  # noqa: S110
            pass


class _AsyncHttpxClientWrapper(openai.DefaultAsyncHttpxClient):
    """Borrowed from openai._base_client."""

    def __del__(self) -> None:
        if self.is_closed:
            return

        try:
            # TODO(someday): support non asyncio runtimes here
            asyncio.get_running_loop().create_task(self.aclose())
        except Exception:  # noqa: S110
            pass


class _LoopAwareAsyncHttpxClientProxy:
    """A proxy for async httpx clients that maintains per-event-loop instances.

    httpx.AsyncClient connections are bound to the event loop they were created
    on. This proxy is cached at the (base_url, timeout) level via @lru_cache,
    preserving fast O(1) init-time lookup. Internally it stores one
    _AsyncHttpxClientWrapper per event loop in a WeakValueDictionary so that
    clients are automatically garbage-collected when their loop closes.

    This avoids the 'Event loop is closed' RuntimeError that occurs when a
    process-global @lru_cache'd client is reused across different event loops
    (e.g. in multi-threaded environments, Celery workers, or sequential
    asyncio.run() calls).
    """

    def __init__(self, base_url: str | None, timeout: Any) -> None:
        self._base_url = base_url
        self._timeout = timeout
        self._loop_clients: weakref.WeakValueDictionary[
            int, _AsyncHttpxClientWrapper
        ] = weakref.WeakValueDictionary()

    def get_client(self) -> _AsyncHttpxClientWrapper:
        """Return the cached client for the current event loop, creating one if needed."""
        try:
            loop = asyncio.get_running_loop()
            loop_id = id(loop)
        except RuntimeError:
            # No running event loop; return a fresh client (won't be cached).
            return _build_async_httpx_client(self._base_url, self._timeout)

        client = self._loop_clients.get(loop_id)
        if client is None:
            client = _build_async_httpx_client(self._base_url, self._timeout)
            self._loop_clients[loop_id] = client
        return client


def _build_sync_httpx_client(
    base_url: str | None, timeout: Any
) -> _SyncHttpxClientWrapper:
    return _SyncHttpxClientWrapper(
        base_url=base_url
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1",
        timeout=timeout,
    )


def _build_async_httpx_client(
    base_url: str | None, timeout: Any
) -> _AsyncHttpxClientWrapper:
    return _AsyncHttpxClientWrapper(
        base_url=base_url
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1",
        timeout=timeout,
    )


@lru_cache
def _cached_sync_httpx_client(
    base_url: str | None, timeout: Any
) -> _SyncHttpxClientWrapper:
    return _build_sync_httpx_client(base_url, timeout)


@lru_cache
def _cached_async_httpx_client_proxy(
    base_url: str | None, timeout: Any
) -> _LoopAwareAsyncHttpxClientProxy:
    """Return a per-(base_url, timeout) proxy cached by @lru_cache.

    The proxy itself is lightweight and @lru_cache'd, so init-time overhead is
    identical to the original implementation. The proxy internally dispatches
    to a per-event-loop client, preventing cross-loop connection reuse.
    """
    return _LoopAwareAsyncHttpxClientProxy(base_url, timeout)


def _get_default_httpx_client(
    base_url: str | None, timeout: Any
) -> _SyncHttpxClientWrapper:
    """Get default httpx client.

    Uses cached client unless timeout is `httpx.Timeout`, which is not hashable.
    """
    try:
        hash(timeout)
    except TypeError:
        return _build_sync_httpx_client(base_url, timeout)
    else:
        return _cached_sync_httpx_client(base_url, timeout)


def _get_default_async_httpx_client(
    base_url: str | None, timeout: Any
) -> _AsyncHttpxClientWrapper:
    """Get default async httpx client, scoped to the current event loop.

    Async httpx clients are bound to the event loop they were created on, so
    they cannot be safely shared across different event loops. This function
    returns a client that is cached per-loop to avoid 'Event loop is closed'
    errors in multi-threaded or multi-loop environments (e.g. Celery, sequential
    asyncio.run() calls, or multi-threaded FastAPI handlers).

    Uses a fresh (uncached) client when timeout is not hashable.
    """
    try:
        hash(timeout)
    except TypeError:
        return _build_async_httpx_client(base_url, timeout)
    else:
        return _cached_async_httpx_client_proxy(base_url, timeout).get_client()


def _resolve_sync_and_async_api_keys(
    api_key: SecretStr | Callable[[], str] | Callable[[], Awaitable[str]],
) -> tuple[str | None | Callable[[], str], str | Callable[[], Awaitable[str]]]:
    """Resolve sync and async API key values.

    Because OpenAI and AsyncOpenAI clients support either sync or async callables
    for the API key, we need to resolve separate values here.
    """
    if isinstance(api_key, SecretStr):
        sync_api_key_value: str | None | Callable[[], str] = api_key.get_secret_value()
        async_api_key_value: str | Callable[[], Awaitable[str]] = (
            api_key.get_secret_value()
        )
    elif callable(api_key):
        if inspect.iscoroutinefunction(api_key):
            async_api_key_value = api_key
            sync_api_key_value = None
        else:
            sync_api_key_value = cast(Callable, api_key)

            async def async_api_key_wrapper() -> str:
                return await asyncio.get_event_loop().run_in_executor(
                    None, cast(Callable, api_key)
                )

            async_api_key_value = async_api_key_wrapper
    return sync_api_key_value, async_api_key_value
