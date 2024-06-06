import pathlib
from sysconfig import get_python_version


def is_interactive() -> bool:
    """
    Check if the code is running in a Jupyter Notebook environment.
    """
    try:
        shell = get_python_version().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter Notebook or JupyterLab
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal or IPython console
        else:
            return False  # Other interactive shells
    except NameError:
        return False  # Not in an interactive shell


def get_base_dir(repo_name="CoastSeg") -> pathlib.Path:
    """
    Get the project directory path.

    Returns:
        A `pathlib.Path` object representing the project directory path.
    """

    def resolve_repo_path(cwd: pathlib.Path, proj_name: str) -> pathlib.Path:
        root = cwd.root
        proj_dir = cwd
        # keep moving up the directory tree until the project directory is found or the root is reached
        while proj_dir.name != proj_name:
            proj_dir = proj_dir.parent
            if str(proj_dir) == root:
                msg = "Reached root depth - cannot resolve project path."
                raise ValueError(msg)
        # return the project directory path for example CoastSeg directory
        return proj_dir

    cwd = pathlib.Path().resolve() if is_interactive() else pathlib.Path(__file__)

    proj_dir = resolve_repo_path(cwd, proj_name=repo_name)
    return proj_dir

