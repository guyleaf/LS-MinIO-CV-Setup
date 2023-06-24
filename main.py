import typer

import src.commands as commands

app = typer.Typer()


app.command(help="Generate config files from templates.")(commands.setup)
app.command()(commands.upload)


@app.command()
def create():
    print("Hello World!")


@app.command()
def evaluate():
    print("Hello World!")


if __name__ == "__main__":
    app()
