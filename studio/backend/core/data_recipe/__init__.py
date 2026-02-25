"""
Data Recipe core (DataDesigner wrapper + job runner).
"""

from .jobs import JobManager, get_job_manager

__all__ = ["JobManager", "get_job_manager"]
