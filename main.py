import typer

import src.commands as commands

app = typer.Typer()


app.command(help="Generate config files from templates.")(commands.setup)

app.command(
    help="Upload data to MinIO and generate json files for label studio tasks."
)(commands.upload)

app.command(help="Create a label studio project and import data from MinIO.")(
    commands.create
)


@app.command()
def evaluate():
    print("Hello World!")


if __name__ == "__main__":
    app()
