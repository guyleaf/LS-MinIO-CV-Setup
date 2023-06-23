import typer

app = typer.Typer()


@app.command()
def setup():
    print("Hello World!")


@app.command()
def upload():
    print("Hello World!")


@app.command()
def create():
    print("Hello World!")


@app.command()
def evaluate():
    print("Hello World!")


if __name__ == "__main__":
    app()
