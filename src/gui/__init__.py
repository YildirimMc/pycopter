"""Panel browser dashboard for PyCopter."""

__all__ = ["create_dashboard", "main"]


def create_dashboard():
    from .dashboard import create_dashboard as dashboard_factory

    return dashboard_factory()


def main() -> None:
    from .dashboard import main as dashboard_main

    dashboard_main()
