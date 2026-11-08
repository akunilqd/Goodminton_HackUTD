"""Module 7 – match summary analytics, report generation, and DB upload."""

from .analytics import generate_report


def upload_run(*args, **kwargs):
    from .stdb_upload import upload_run as _upload
    return _upload(*args, **kwargs)


__all__ = ["generate_report", "upload_run"]
