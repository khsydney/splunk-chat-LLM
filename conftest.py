import os
import ssl
import certifi

_orig_create_default_context = ssl.create_default_context

def _patched_create_default_context(purpose=ssl.Purpose.SERVER_AUTH, *, cafile=None, capath=None, cadata=None):
    if cafile is None and capath is None and cadata is None:
        cafile = certifi.where()
    return _orig_create_default_context(purpose, cafile=cafile, capath=capath, cadata=cadata)

ssl.create_default_context = _patched_create_default_context

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
