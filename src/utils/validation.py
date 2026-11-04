import re
from urllib.parse import urlparse

def validate_url(url: str) -> bool:
    if not isinstance(url, str) or not url.strip():
        return False
    url = url.strip()
    # Add scheme if missing
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url_check = "http://" + url
    else:
        url_check = url
    parsed = urlparse(url_check)
    if not parsed.netloc:
        return False
    # Basic checks
    if len(url) > 2048:
        return False
    return True

def is_valid_schema(df, url_col="url", label_col=None):
    if url_col not in df.columns:
        raise ValueError(f"Missing required column: {url_col}")
    if label_col is not None and label_col not in df.columns:
        raise ValueError(f"Missing required column: {label_col}")
    return True
