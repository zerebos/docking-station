import asyncio
import random
import subprocess
from datetime import timedelta
from logging import getLogger

from fastapi_cache import FastAPICache

from ..schemas import RegctlImageInspect
from ..settings import cached, get_app_settings

__all__ = [
    'get_image_inspect',
    'get_image_remote_digest',
]

app_settings = get_app_settings()
logger = getLogger(__name__)

# Upper bound on a single backoff wait, regardless of the configured base.
_MAX_BACKOFF_SECONDS = 60.0

# Substrings that mark a regctl error as transient/rate-limit related and thus
# worth retrying with backoff instead of failing (and poisoning the cache).
_RETRYABLE_ERROR_SIGNATURES = (
    'toomanyrequests',
    'too many requests',
    'rate limit',
    'ratelimit',
    'status 429',
    'status code 429',
    'httpstatus: 429',
    'http status 429',
    'status 500',
    'status 502',
    'status 503',
    'status 504',
    'timeout',
    'timed out',
    'connection reset',
    'connection refused',
    'temporary failure',
    'eof',
)

# Substrings that mark an error as a definitive "this image has no remote"
# result (e.g. local-only images). These are safe to negative-cache briefly so
# we don't re-run regctl for them on every dashboard load.
_NOT_FOUND_ERROR_SIGNATURES = (
    'manifest unknown',
    'no such manifest',
    'no such repository',
    'repository name not known',
    'repository does not exist',
    'name unknown',
    'not found',
    'notfound',
)

# Global limiter shared by every registry-bound regctl invocation. Without this,
# rendering the dashboard fires one subprocess per image (digest + inspect)
# concurrently, which bursts straight into Docker Hub's rate limit. Created
# lazily so it binds to the running event loop rather than import-time state.
_registry_semaphore: asyncio.Semaphore | None = None


def _get_registry_semaphore() -> asyncio.Semaphore:
    global _registry_semaphore
    if _registry_semaphore is None:
        _registry_semaphore = asyncio.Semaphore(
            max(1, app_settings.server.registry_max_concurrency)
        )
    return _registry_semaphore


class RegctlError(Exception):
    """Raised when a regctl command fails after exhausting retries."""


class RegctlNotFoundError(RegctlError):
    """Raised when regctl reports the image definitively has no remote."""


def _is_retryable_error(message: str) -> bool:
    msg = (message or '').lower()
    return any(signature in msg for signature in _RETRYABLE_ERROR_SIGNATURES)


def _is_not_found_error(message: str) -> bool:
    msg = (message or '').lower()
    return any(signature in msg for signature in _NOT_FOUND_ERROR_SIGNATURES)


async def _negative_cache_has(repo_tag: str) -> bool:
    """Return True if repo_tag was recently seen as definitively not-found."""
    if app_settings.server.registry_negative_cache_seconds <= 0:
        return False
    try:
        backend = FastAPICache.get_backend()
        _ttl, data = await backend.get_with_ttl(f'regctl:not-found:{repo_tag}')
        return data is not None
    except Exception:
        logger.warning('Error reading regctl negative cache for %s', repo_tag, exc_info=True)
        return False


async def _negative_cache_add(repo_tag: str) -> None:
    """Remember, for a short TTL, that repo_tag has no remote."""
    ttl = int(app_settings.server.registry_negative_cache_seconds)
    if ttl <= 0:
        return
    try:
        backend = FastAPICache.get_backend()
        await backend.set(f'regctl:not-found:{repo_tag}', '1', ttl)
    except Exception:
        logger.warning('Error writing regctl negative cache for %s', repo_tag, exc_info=True)


async def _run_regctl(args: list[str]) -> bytes:
    """Run a regctl command under the global concurrency limiter, retrying
    rate-limit/transient failures with exponential backoff and jitter.

    Arguments are passed as a list and executed without a shell to avoid any
    command injection from image tags. Returns the command's stdout on success.
    Raises RegctlNotFoundError for a definitive not-found result, or RegctlError
    otherwise.
    """
    max_retries = max(0, app_settings.server.registry_max_retries)
    base_backoff = max(0.0, app_settings.server.registry_retry_backoff_seconds)
    last_error = ''

    for attempt in range(max_retries + 1):
        # Only hold a concurrency slot while the subprocess is actually running;
        # release it before backing off so other calls can make progress.
        async with _get_registry_semaphore():
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                return stdout

            last_error = stderr.decode('utf-8', errors='replace').strip()

        if attempt < max_retries and _is_retryable_error(last_error):
            wait = min(base_backoff * (2 ** attempt), _MAX_BACKOFF_SECONDS)
            wait += random.uniform(0, wait / 2)  # jitter to avoid thundering herd
            logger.warning(
                'regctl rate-limited/transient error (attempt %d/%d), retrying in %.1fs: %s',
                attempt + 1, max_retries, wait, last_error,
            )
            await asyncio.sleep(wait)
            continue

        break

    if _is_not_found_error(last_error) and not _is_retryable_error(last_error):
        raise RegctlNotFoundError(last_error or 'not found')
    raise RegctlError(last_error or 'unknown error')


async def get_image_remote_digest(repo_tag: str, reraise: bool = False, no_cache: bool = False):
    cache_control_max_age_seconds = (timedelta(days=365).total_seconds()
                                     if 'sha256:' in repo_tag
                                     else app_settings.server.cache_control_max_age_seconds)

    @cached(expire=cache_control_max_age_seconds, cache_none=False)
    async def _get_image_remote_digest(repo_tag: str, no_cache: bool = False):
        nonlocal reraise

        try:
            if ':' in repo_tag:
                image_name, _tag = repo_tag.split(':', 1)
            else:
                image_name, _tag = repo_tag, ''

            if not no_cache and await _negative_cache_has(repo_tag):
                logger.debug('regctl image digest skipped (negative cache): %s', repo_tag)
                return None

            logger.debug('regctl image digest request: %s', repo_tag)
            stdout = await _run_regctl(['regctl', 'image', 'digest', repo_tag])

            digest = stdout.decode().strip()
            res = f'{image_name}@{digest}'
            logger.info('regctl image digest response: %s', res)
            return res

        except RegctlNotFoundError as e:
            logger.info('regctl image digest not found (negative-cached): %s (%s)', repo_tag, e)
            await _negative_cache_add(repo_tag)
            if reraise:
                raise Exception(f'Error running regctl command: {e}')
            return None

        except Exception as e:
            logger.error('Error running regctl command: %s', e)
            if reraise:
                raise Exception(f'Error running regctl command: {e}')
            return None

    return await _get_image_remote_digest(
        repo_tag=repo_tag,
        no_cache=no_cache,
    )


async def get_image_inspect(repo_tag: str, reraise: bool = False, no_cache: bool = False):
    is_specific_digest = 'sha256:' in repo_tag
    cache_control_max_age_seconds = (timedelta(days=365).total_seconds()
                                     if is_specific_digest
                                     else app_settings.server.cache_control_max_age_seconds)

    @cached(expire=cache_control_max_age_seconds, cache_none=False)
    async def _get_image_inspect(repo_tag: str, no_cache: bool = False) -> RegctlImageInspect:
        nonlocal reraise

        try:
            if not no_cache and await _negative_cache_has(repo_tag):
                logger.debug('regctl image inspect skipped (negative cache): %s', repo_tag)
                return None

            logger.debug('regctl image inspect request: %s', repo_tag)
            stdout = await _run_regctl(['regctl', 'image', 'inspect', repo_tag])

            res = RegctlImageInspect.model_validate_json(stdout)
            logger.info('regctl image inspect response: %s', res.created)
            return res

        except RegctlNotFoundError as e:
            logger.info('regctl image inspect not found (negative-cached): %s (%s)', repo_tag, e)
            await _negative_cache_add(repo_tag)
            if reraise:
                raise Exception(f'Error running regctl command: {e}')
            return None

        except Exception as e:
            logger.error('Error running regctl command: %s', e)
            if reraise:
                raise Exception(f'Error running regctl command: {e}')
            return None

    return await _get_image_inspect(
        repo_tag=repo_tag,
        no_cache=False if is_specific_digest else no_cache,
    )
