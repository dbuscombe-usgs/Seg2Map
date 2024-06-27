import pathlib
from sysconfig import get_python_version
import os

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

    def resolve_repo_path(cwd: pathlib.Path, proj_name: str, max_depth: int = 100) -> pathlib.Path:
        root = cwd.anchor if cwd.anchor else cwd.root  # Handle root differently for Windows
        # print(f"root: {root}")
        proj_dir = cwd
        depth = 0

        # Keep moving up the directory tree until the project directory is found or the root is reached
        while proj_dir.name != proj_name:
            if depth > max_depth:
                raise ValueError(f"Reached maximum depth - cannot resolve project path. Could not find {proj_name} in {cwd}.")
            proj_dir = proj_dir.parent
            depth += 1
            if str(proj_dir) == root:
                raise ValueError(f"Reached root depth - cannot resolve project path. Could not find {proj_name} in {cwd}.")
        
        # Return the project directory path, for example, CoastSeg directory
        return proj_dir


    cwd = pathlib.Path().resolve() if is_interactive() else pathlib.Path(__file__)
    try:
        proj_dir = resolve_repo_path(cwd, proj_name=repo_name)
    except ValueError as e:
        try:
            # see if where the script is running from contains the project directory CoastSeg
            cwd = pathlib.Path(os.getcwd())
            proj_dir = resolve_repo_path(cwd, proj_name=repo_name)
        except ValueError as e:
            # get the current working directory if the project directory is not found
            proj_dir = os.getcwd()
            # convert to a pathlib.Path object
            proj_dir = pathlib.Path(proj_dir)
            
    return proj_dir
