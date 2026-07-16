import asyncio
import random
import subprocess
from datetime import timedelta
from logging import getLogger

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

# Global limiter shared by every registry-bound regctl invocation. Without this,
# rendering the dashboard fires one subprocess per image (digest + inspect)
# concurrently, which bursts straight into Docker Hub's rate limit.
_registry_semaphore = asyncio.Semaphore(
    max(1, app_settings.server.registry_max_concurrency)
)


class RegctlError(Exception):
    """Raised when a regctl command fails after exhausting retries."""


def _is_retryable_error(message: str) -> bool:
    msg = (message or '').lower()
    return any(signature in msg for signature in _RETRYABLE_ERROR_SIGNATURES)


async def _run_regctl(cmd: str) -> bytes:
    """Run a regctl command under the global concurrency limiter, retrying
    rate-limit/transient failures with exponential backoff and jitter.

    Returns the command's stdout on success, raises RegctlError otherwise.
    """
    max_retries = max(0, app_settings.server.registry_max_retries)
    base_backoff = app_settings.server.registry_retry_backoff_seconds
    last_error = ''

    for attempt in range(max_retries + 1):
        # Only hold a concurrency slot while the subprocess is actually running;
        # release it before backing off so other calls can make progress.
        async with _registry_semaphore:
            process = await asyncio.create_subprocess_shell(
                cmd=cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                return stdout

            last_error = stderr.decode().strip()

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

            cmd = f'regctl image digest "{repo_tag}"'
            logger.debug('regctl image digest request: %s', repo_tag)
            stdout = await _run_regctl(cmd)

            digest = stdout.decode().strip()
            res = f'{image_name}@{digest}'
            logger.info('regctl image digest response: %s', res)
            return res

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
            cmd = f'regctl image inspect "{repo_tag}"'
            logger.debug('regctl image inspect request: %s', repo_tag)
            stdout = await _run_regctl(cmd)

            res = RegctlImageInspect.model_validate_json(stdout)
            logger.info('regctl image inspect response: %s', res.created)
            return res

        except Exception as e:
            logger.error('Error running regctl command: %s', e)
            if reraise:
                raise Exception(f'Error running regctl command: {e}')
            return None

    return await _get_image_inspect(
        repo_tag=repo_tag,
        no_cache=False if is_specific_digest else no_cache,
    )
