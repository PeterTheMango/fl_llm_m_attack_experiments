# master_script/webui/localguard.py
"""Restrict a route to callers on the loopback interface.

The settings page reads and writes the service-account credentials, so it is
the one part of the dashboard that must not follow the tunnel out to the public
internet. Everything else the monitor exposes is read-mostly experiment data;
this is the credential store.

"Local" here means the request was made from the machine running the server --
a browser on the VM, or an SSH port-forward, which terminates on loopback and
is therefore already an authenticated channel. A viewer on the LAN, on the
intranet VPN, or coming down the tunnel is not local.
"""
import ipaddress

from fastapi import HTTPException, Request

# A proxied request carries the original caller's address in one of these. The
# tunnel agent connects to the dashboard over loopback, so the socket address
# alone would say "local" for every remote viewer; the presence of a forwarding
# header is what distinguishes a proxied hop from a genuine local one.
FORWARD_HEADERS = ("x-forwarded-for", "x-real-ip", "x-forwarded-host",
                   "cf-connecting-ip", "forwarded")

DENIED = ("The settings page is available only on the machine running the "
          "dashboard. Open it on the VM itself, or forward the port over SSH "
          "(ssh -L 8080:localhost:8080 <host>).")


def client_host(request: Request) -> str:
    return request.client.host if request.client else ""


def is_loopback(host: str) -> bool:
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    # ::ffff:127.0.0.1 is a loopback caller wearing an IPv6 address; the
    # IPv6Address itself does not report is_loopback, the mapped v4 does.
    mapped = getattr(address, "ipv4_mapped", None)
    if mapped is not None:
        address = mapped
    return address.is_loopback


def is_local(request: Request) -> bool:
    if any(header in request.headers for header in FORWARD_HEADERS):
        return False
    return is_loopback(client_host(request))


def require_local(request: Request) -> None:
    """FastAPI dependency: 403 anything that did not originate on this host."""
    if not is_local(request):
        raise HTTPException(status_code=403, detail=DENIED)
