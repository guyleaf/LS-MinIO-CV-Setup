import secrets

from rich.console import Console

console = Console()
err_console = Console(stderr=True)


def generate_token(length: int) -> str:
    return secrets.token_hex(length)
