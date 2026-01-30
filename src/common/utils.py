import pathlib

# Source - https://stackoverflow.com/a/28834214
def delete_dir(pth: pathlib.Path):
    for sub in pth.iterdir():
        if sub.is_dir():
            delete_dir(sub)
        else:
            sub.unlink()
    pth.rmdir()
