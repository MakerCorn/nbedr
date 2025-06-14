import contextlib
import os

# List of environment variables prefixes that are allowed to be used for configuration.
env_prefix_whitelist = ["OPENAI", "AZURE_OPENAI"]


def read_env_config(use_prefix: str, env: dict = os.environ) -> str:
    """Reads whitelisted environment variables and returns them in a dictionary.

    Overrides the whitelisted environment variable with ones prefixed with the given use_prefix if available.

    Args:
        use_prefix (str): The prefix to filter environment variables.
        env (dict, optional): The environment variables dictionary. Defaults to os.environ.

    Returns:
        str: A dictionary with the whitelisted environment variables.
    """
    config = {}
    for prefix in [None, use_prefix]:
        read_env_config_prefixed(prefix, config, env)
    return config


def read_env_config_prefixed(use_prefix: str, config: dict, env: dict = os.environ) -> str:
    """Reads whitelisted environment variables prefixed with use_prefix and adds them to the dictionary
    with use_prefix stripped.

    Args:
        use_prefix (str): The prefix to filter environment variables.
        config (dict): The dictionary to store the filtered environment variables.
        env (dict, optional): The environment variables dictionary. Defaults to os.environ.
    """
    use_prefix = format_prefix(use_prefix)
    for key in env:
        for env_prefix in env_prefix_whitelist:
            key_prefix = f"{use_prefix}{format_prefix(env_prefix)}"
            if key.startswith(key_prefix):
                striped_key = key.removeprefix(use_prefix)
                config[striped_key] = env[key]


def format_prefix(prefix: str) -> str:
    """Formats the prefix to be used in the environment variable.

    Args:
        prefix (str): The prefix to format.

    Returns:
        str: The formatted prefix.
    """
    if prefix and len(prefix) > 0 and not prefix.endswith("_"):
        prefix = f"{prefix}_"
    if not prefix:
        prefix = ""
    return prefix


@contextlib.contextmanager
def set_env(**environ: dict[str, str]):
    """Temporarily set the process environment variables.

    Warning, this is not thread safe as the environment is updated for the whole process.

    Args:
        environ (dict[str, str]): Environment variables to set.

    Yields:
        None
    """
    old_environ = dict(os.environ)
    os.environ.update(environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)


def get_env_variable(key, default=None):
    """Retrieves the value of an environment variable, returning a default if not set.

    Args:
        key (str): The environment variable key.
        default (Any, optional): The default value to return if the variable is not set. Defaults to None.

    Returns:
        str or Any: The value of the environment variable or the default value.
    """
    return os.environ.get(key, default)


def load_env_file(env_path):
    """Loads environment variables from a .env file.

    Args:
        env_path (str): Path to the .env file.
    """
    from dotenv import load_dotenv

    load_dotenv(env_path)
