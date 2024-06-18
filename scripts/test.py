"""Test script
"""

import argparse


def main(args: argparse.Namespace) -> None:
    """Main function

    Parameters
    ----------
    args : argparse.Namespace
        _description_
    """
    pass


def get_args() -> argparse.Namespace:
    """Parse command-line arguments

    Returns
    -------
    argparse.Namespace
        Arguments
    """
    parser = argparse.ArgumentParser(
        "Test script description",
    )
    return parser.parse_args()


if __name__ == "__main__":
    opts = get_args()
    main(opts)
