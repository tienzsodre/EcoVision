import argparse

from recicla.web.server import run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interface web local da EcoVision")
    parser.add_argument("--no-browser", action="store_true", help="Exibe a URL sem abrir o navegador")
    run(open_browser=not parser.parse_args().no_browser)
