import re
import math
import ipaddress
from urllib.parse import urlparse, unquote
from collections import Counter
import tldextract

SUSPICIOUS_TLDS = {
    "xyz", "top", "gq", "ml", "cf", "tk", "pw", "cc", "buzz", "info", "icu", "work", "click",
    "link", "loan", "win", "bid", "trade", "review", "country", "stream", "download", "racing"
}
SHORTENERS = {"bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly", "adf.ly", "bit.do"}

def shannon_entropy(s):
    if not s:
        return 0.0
    p, lns = Counter(s), float(len(s))
    return -sum(count/lns * math.log2(count/lns) for count in p.values())

def extract_features_from_url(url: str) -> dict:
    features = {}
    if not isinstance(url, str):
        url = str(url)
    original = url.strip()
    # ensure scheme
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", original):
        url = "http://" + original
    else:
        url = original

    parsed = urlparse(url)
    ext = tldextract.extract(url)
    domain = ext.domain
    suffix = ext.suffix
    subdomain = ext.subdomain
    registered_domain = f"{domain}.{suffix}" if domain and suffix else parsed.netloc

    # URL components
    features["url_length"] = len(original)
    features["hostname_length"] = len(parsed.netloc)
    features["path_length"] = len(parsed.path)
    features["query_length"] = len(parsed.query)
    features["fragment_length"] = len(parsed.fragment)

    # Count
    features["count_dots"] = original.count('.')
    features["count_hyphens"] = original.count('-')
    features["count_underscores"] = original.count('_')
    features["count_slashes"] = original.count('/')
    features["count_question"] = original.count('?')
    features["count_equal"] = original.count('=')
    features["count_at"] = original.count('@')
    features["count_and"] = original.count('&')
    features["count_exclamation"] = original.count('!')
    features["count_space"] = original.count(' ')
    features["count_percent"] = original.count('%')
    features["count_digits"] = sum(c.isdigit() for c in original)
    features["count_letters"] = sum(c.isalpha() for c in original)

    # Ratios
    if features["url_length"] > 0:
        features["ratio_digits"] = features["count_digits"] / features["url_length"]
        features["ratio_letters"] = features["count_letters"] / features["url_length"]
    else:
        features["ratio_digits"] = 0.0
        features["ratio_letters"] = 0.0

    # Special patterns
    features["has_at"] = 1 if '@' in original else 0
    features["has_double_slash_redirect"] = 1 if '//' in parsed.path else 0
    features["has_ip"] = 0
    try:
        # check netloc for IP
        netloc_no_port = parsed.netloc.split(':')[0]
        ipaddress.ip_address(netloc_no_port)
        features["has_ip"] = 1
    except Exception:
        features["has_ip"] = 0
    features["has_https"] = 1 if parsed.scheme == 'https' else 0

    # Subdomain
    features["num_subdomains"] = len([s for s in subdomain.split('.') if s]) if subdomain else 0
    features["subdomain_length"] = len(subdomain) if subdomain else 0

    # TLD
    features["tld"] = suffix.lower() if suffix else ""
    features["is_suspicious_tld"] = 1 if features["tld"] in SUSPICIOUS_TLDS else 0

    # Shortener
    features["is_shortener"] = 1 if any(s in parsed.netloc.lower() for s in SHORTENERS) else 0

    # Entropy
    features["entropy_hostname"] = shannon_entropy(parsed.netloc)
    features["entropy_url"] = shannon_entropy(original)

    # Phishy keywords
    phishy_keywords = ["login", "signin", "verify", "update", "secure", "account", "bank", "paypal", "ebay", "amazon", "free", "bonus", "winner", "confirm", "webscr", "password"]
    lower_url = original.lower()
    features["count_phishy_keywords"] = sum(1 for k in phishy_keywords if k in lower_url)

    # Port
    features["has_port"] = 1 if ':' in parsed.netloc and not features["has_ip"] else (1 if parsed.netloc.count(':') > 1 else 0)
    # simpler
    features["has_nonstandard_port"] = 1 if parsed.port and parsed.port not in (80, 443) else 0

    # Path/query suspicious
    features["path_has_sensitive"] = 1 if any(k in parsed.path.lower() for k in ["admin", "login", ".exe", ".php"]) else 0
    features["query_has_sensitive"] = 1 if any(k in parsed.query.lower() for k in ["id=", "token=", "password=", "session="]) else 0

    # Length of domain words
    features["domain_length"] = len(domain) if domain else 0

    return features

def extract_features_batch(urls):
    import pandas as pd
    rows = []
    for u in urls:
        try:
            rows.append(extract_features_from_url(u))
        except Exception as e:
            # return minimal features on error
            rows.append({
                "url_length": len(str(u)),
                "error": 1
            })
    return pd.DataFrame(rows)
