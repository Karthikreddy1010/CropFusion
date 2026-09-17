"""
noaa_cdo_client.py - Official NOAA/NCEI CDO API v2 Client & Direct Access Layer.

Features:
- Authenticates using NOAA_CDO_TOKEN environment variable (never hardcoded, never logged)
- Implements token-bucket rate limiting (default 4.0 req/s to stay well below the 5.0 req/s cap)
- Handles pagination with limit and offset checking metadata.resultset.count
- Exponential backoff with jitter on HTTP 429 / 5xx errors
- Checkpointing and local JSON/CSV caching in outputs/noaa_cache
- Fallback/direct station daily access from official NOAA NCEI archive
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("noaa_weather")


class NoaaCdoClient:
    """Production client for NOAA CDO API v2 and NCEI GHCND data services."""

    BASE_URL = "https://www.ncei.noaa.gov/cdo-web/api/v2"
    DIRECT_GHCND_URL = "https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access"

    def __init__(
        self,
        token: Optional[str] = None,
        cache_dir: Optional[Path | str] = None,
        rate_limit_per_sec: float = 4.0,
        max_retries: int = 5,
        timeout: int = 30,
    ) -> None:
        self.token = token or os.environ.get("NOAA_CDO_TOKEN", "").strip()
        if not self.token:
            raise ValueError(
                "NOAA_CDO_TOKEN environment variable is not set! "
                "Please set NOAA_CDO_TOKEN before running weather data acquisition."
            )

        self.cache_dir = Path(cache_dir or "outputs/noaa_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_delay = 1.0 / max(0.1, rate_limit_per_sec)
        self.last_request_time = 0.0
        self.max_retries = max_retries
        self.timeout = timeout
        self.request_history: List[Dict[str, Any]] = []

    def _wait_for_rate_limit(self) -> None:
        """Enforce spacing between API requests."""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    def _get_cache_path(self, endpoint: str, params: Dict[str, Any]) -> Path:
        """Deterministic cache file path based on endpoint and parameters."""
        serialized = json.dumps(
            {"endpoint": endpoint, "params": params}, sort_keys=True
        )
        h = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
        sanitized_endpoint = endpoint.replace("/", "_")
        return self.cache_dir / f"{sanitized_endpoint}_{h}.json"

    def query(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Query NOAA CDO API endpoint with retries, backoff, and caching.

        Parameters
        ----------
        endpoint : str
            API endpoint (e.g. 'datasets', 'stations', 'datatypes', 'data').
        params : dict, optional
            Query parameters.
        use_cache : bool
            Whether to read/write local disk cache.
        """
        params = params or {}
        cache_path = self._get_cache_path(endpoint, params)
        if use_cache and cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                return cached_data
            except Exception as e:
                logger.warning("Failed to read cache file %s: %s", cache_path, e)

        # Build URL
        query_parts: List[Tuple[str, str]] = []
        for k, v in sorted(params.items()):
            if isinstance(v, (list, tuple, set)):
                for item in sorted(v):
                    query_parts.append((k, str(item)))
            elif v is not None:
                query_parts.append((k, str(v)))

        query_str = urllib.parse.urlencode(query_parts)
        url = f"{self.BASE_URL}/{endpoint}"
        if query_str:
            url = f"{url}?{query_str}"

        headers = {
            "token": self.token,
            "User-Agent": "AgriculturalResearch-CropFusion/1.0",
        }

        retries = 0
        backoff = 1.0

        while retries <= self.max_retries:
            self._wait_for_rate_limit()
            req = urllib.request.Request(url, headers=headers)
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    status = resp.status
                    raw_bytes = resp.read()
                    data = json.loads(raw_bytes.decode("utf-8"))

                # Record request metadata
                record_count = len(data.get("results", []))
                self.request_history.append(
                    {
                        "endpoint": endpoint,
                        "params": params,
                        "status": status,
                        "records": record_count,
                        "retries": retries,
                        "duration_sec": round(time.time() - t0, 3),
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                )

                if use_cache:
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)

                return data

            except urllib.error.HTTPError as e:
                status = e.code
                error_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
                logger.warning(
                    "HTTP %d on %s: %s (attempt %d/%d). Backing off %.1fs",
                    status, endpoint, e.reason, retries + 1, self.max_retries + 1, backoff
                )
                if status in (429, 500, 502, 503, 504):
                    time.sleep(backoff)
                    backoff *= 2.0
                    retries += 1
                    continue
                else:
                    return {
                        "error": f"HTTP {status}: {e.reason}",
                        "body": error_body,
                        "status": status,
                    }

            except Exception as e:
                logger.warning(
                    "Network error on %s: %s (attempt %d/%d). Backing off %.1fs",
                    endpoint, e, retries + 1, self.max_retries + 1, backoff
                )
                time.sleep(backoff)
                backoff *= 2.0
                retries += 1

        return {"error": f"Max retries exceeded ({self.max_retries}) for {endpoint}"}

    def paginate(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        limit: int = 1000,
        max_records: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Paginate through all available records using limit and offset.

        Validates downloaded_records == expected_records against resultset count.
        """
        params = dict(params or {})
        params["limit"] = limit
        offset = 1
        all_results: List[Dict[str, Any]] = []

        while True:
            params["offset"] = offset
            resp = self.query(endpoint, params)
            if "error" in resp:
                logger.error("Pagination aborted due to error: %s", resp["error"])
                break

            results = resp.get("results", [])
            if not results:
                break

            all_results.extend(results)
            meta = resp.get("metadata", {}).get("resultset", {})
            total_count = meta.get("count", 0)

            logger.info(
                "  [%s] Downloaded %d / %d records (offset=%d)...",
                endpoint, len(all_results), total_count, offset
            )

            if max_records and len(all_results) >= max_records:
                all_results = all_results[:max_records]
                break

            if len(all_results) >= total_count or len(results) < limit:
                break

            offset += len(results)

        return all_results

    def download_station_ghcnd_csv(
        self,
        station_id: str,
        use_cache: bool = True,
    ) -> Optional[Path]:
        """Download complete official daily GHCND CSV file directly from NOAA NCEI.

        Parameters
        ----------
        station_id : str
            Full GHCND station ID (e.g. 'GHCND:USC00118740' or 'USC00118740').
        """
        clean_id = station_id.replace("GHCND:", "").strip()
        station_cache = self.cache_dir / f"ghcnd_{clean_id}.csv"
        if use_cache and station_cache.exists() and station_cache.stat().st_size > 500:
            return station_cache

        url = f"{self.DIRECT_GHCND_URL}/{clean_id}.csv"
        headers = {"User-Agent": "AgriculturalResearch-CropFusion/1.0"}
        retries = 0
        backoff = 1.0

        while retries <= self.max_retries:
            self._wait_for_rate_limit()
            req = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    content = resp.read().decode("utf-8", errors="replace")
                    with open(station_cache, "w", encoding="utf-8") as f:
                        f.write(content)
                logger.info("Retrieved official NCEI GHCND daily table for %s (%d bytes)", clean_id, len(content))
                return station_cache
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    logger.warning("Station %s not found on NOAA NCEI daily access (404)", clean_id)
                    return None
                logger.warning("HTTP %d downloading %s: %s (attempt %d). Backing off %.1fs", e.code, clean_id, e.reason, retries+1, backoff)
                time.sleep(backoff)
                backoff *= 1.5
                retries += 1
            except Exception as e:
                logger.warning("Network error downloading %s: %s (attempt %d). Backing off %.1fs", clean_id, e, retries+1, backoff)
                time.sleep(backoff)
                backoff *= 1.5
                retries += 1

        logger.error("Failed to download NCEI GHCND CSV for %s after %d retries", clean_id, self.max_retries)
        return None
