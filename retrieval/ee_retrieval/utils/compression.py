import os
import subprocess
import shutil
from shutil import which


def compress_and_remove(
        folder_path: str,
        threads: int = 4,
        level: int = 1
    ) -> str:
    """Compress a folder into a .tar.xz archive and remove the original.

    Args:
        folder_path (str): Path to the folder to compress.
        thread (int): Number of threads to use. Defaults to ``4``.
        level (int): Compression level; from 1-9. Defaults to ``1``.

    Returns:
        str: Path to the created archive.

    Raises:
        FileNotFoundError: If folder or required tools are missing.
        RuntimeError: If compression fails or archive is empty.
    """

    folder_path = folder_path.rstrip(os.sep)
    abs_folder = os.path.abspath(folder_path)
    if not os.path.isdir(abs_folder):
        raise FileNotFoundError(f"No such directory: {folder_path}")

    if which("tar") is None or which("xz") is None:
        raise FileNotFoundError("Both 'tar' and 'xz' must be installed and on your PATH")

    # get compression paths
    parent_dir, folder_name = os.path.split(abs_folder)
    if not folder_name:
        parent_dir, folder_name = os.path.split(parent_dir)

    archive_path = os.path.join(parent_dir, f"{folder_name}.tar.xz")

    compress_prog = f"xz -{level} -T{threads}"

    # do the compression
    cmd = [
        "tar",
        "--use-compress-program", compress_prog,
        "-cf", archive_path,
        "-C", parent_dir, folder_name
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Compression failed: {e}")

    # remove orig.
    if not os.path.isfile(archive_path) or os.path.getsize(archive_path) == 0:
        if os.path.exists(archive_path):
            os.remove(archive_path)
        raise RuntimeError(f"Archive creation failed or resulted in empty file: {archive_path}")

    shutil.rmtree(abs_folder)

    return archive_path

def compressed_path(
        folder_path: str,
        suffix: str = ".tar.xz"
    ) -> str:
    """Return the expected compressed archive path for a folder.

    Args:
        folder_path (str): Path to the folder.
        suffix (str): Compression suffix. Defaults to ``.tar.xz``

    Returns:
        str: Full path to the archive.
    """

    # get paths
    abs_folder = os.path.abspath(folder_path.rstrip(os.sep))
    parent_dir, folder_name = os.path.split(abs_folder)
    if not folder_name:
        parent_dir, folder_name = os.path.split(parent_dir)

    archive_path = os.path.join(parent_dir, folder_name + suffix)

    return archive_path

def decompress_and_remove(
        folder_path: str,
        threads: int = 4,
        suffix: str = ".tar.xz"
    ) -> str:
    """Decompress a .tar.xz archive and remove the archive file.

    Args:
        folder_path (str): Target folder path to restore.
        threads (int): Number of threads to use. Defaults to ``4``.
        suffix (str): Archive suffix. Defaults to ``.tar.xz``

    Returns:
        str: Path to the extracted folder.

    Raises:
        FileNotFoundError: If archive or tools are missing.
        FileExistsError: If output folder already exists.
        RuntimeError: If decompression fails.
    """

    # get paths
    abs_folder = os.path.abspath(folder_path.rstrip(os.sep))
    parent_dir, folder_name = os.path.split(abs_folder)
    if not folder_name:
        parent_dir, folder_name = os.path.split(parent_dir)

    archive_path = os.path.join(parent_dir, folder_name + suffix)

    if os.path.isdir(folder_path):
        raise FileExistsError(f'Output folder {folder_path} already exists')
    if which("tar") is None or which("xz") is None:
        raise FileNotFoundError("Both 'tar' and 'xz' must be installed and on your PATH")
    if not os.path.isfile(archive_path):
        raise FileNotFoundError(f"No such archive: {archive_path}")

    # do decompression
    cmd = [
        "tar",
        "--use-compress-program", f"xz -T{threads} -d",
        "-xf", archive_path,
        "-C", parent_dir
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Decompression failed: {e}")

    # remove archive
    if not os.path.isdir(abs_folder) or not os.listdir(abs_folder):
        raise RuntimeError(f"Extraction failed or produced empty folder: {abs_folder}")

    os.remove(archive_path)

    return abs_folder

def du(path: str) -> float:
    """Return the disk usage of a path in GiB.

    Args:
        path (str): File or directory path.

    Returns:
        float: Size in GiB.

    Raises:
        FileNotFoundError: If path does not exist.
        RuntimeError: If `du` command fails.
        ValueError: If output parsing fails.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"No such file or directory: {path}")

    try:
        proc = subprocess.run(
            ["du", "-s", "--bytes", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"du failed: {e.stderr.strip()}")

    parts = proc.stdout.strip().split()
    if not parts:
        raise ValueError(f"Unexpected du output: {proc.stdout!r}")

    try:
        size_bytes = int(parts[0])
    except ValueError:
        raise ValueError(f"Could not parse size from du output: {parts[0]!r}")

    size_gib = size_bytes / (1024 ** 3)
    return size_gib


def compress_file_and_remove(
        file_path: str,
        threads: int = 4,
        level: int = 1,
        suffix: str = ".zst"
    ) -> str:
    """Compress a file using zstd and remove the original.

    Args:
        file_path (str): Path to the file to compress.
        threads (int): Number of threads. Defaults to ``4``.
        level (int): Compression level. Defaults to ``1``.
        suffix (str): Archive suffix. Defaults to ``.zst``.

    Returns:
        str: Path to the compressed file.

    Raises:
        FileNotFoundError: If file or tool is missing.
        RuntimeError: If compression fails.
    """

    abs_file = os.path.abspath(file_path)
    if not os.path.isfile(abs_file):
        raise FileNotFoundError(f"No such file: {file_path!r}")

    if which("zstd") is None:
        raise FileNotFoundError("'zstd' must be installed and on your PATH")

    archive_path = abs_file + suffix

    cmd = ["zstd", f"-{level}", f"-T{threads}", "--rm", abs_file]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        if os.path.exists(archive_path):
            os.remove(archive_path)
        raise RuntimeError(f"Compression failed: {e}")

    if not os.path.isfile(archive_path) or os.path.getsize(archive_path) == 0:
        raise RuntimeError(f"Archive creation failed or resulted in empty file: {archive_path!r}")

    return archive_path


def compressed_file_path(
        file_path: str,
        suffix: str = ".zst"
    ) -> str:
    """Return the expected compressed file path.

    Args:
        file_path (str): Original file path.
        suffix (str): Compression suffix. Defaults to ``.zst``

    Returns:
        str: File path with suffix appended if missing.
    """

    return file_path if file_path.endswith(suffix) else file_path + suffix


def decompress_file_and_remove(
        file_path: str,
        threads: int = 4,
        suffix: str = ".zst"
    ) -> str:
    """Decompress a zstd file and remove the archive.

    Args:
        file_path (str): Target file path to restore.
        threads (int): Number of threads. Defaults to ``4``.
        suffix (str): Archive suffix. Defaults to ``.zst``

    Returns:
        str: Path to the decompressed file.

    Raises:
        FileExistsError: If output file exists.
        FileNotFoundError: If archive or tool missing.
        RuntimeError: If decompression fails.
    """

    abs_target = os.path.abspath(file_path)
    archive = abs_target if abs_target.endswith(suffix) else abs_target + suffix

    if os.path.isfile(abs_target):
        raise FileExistsError(f'Output file {abs_target} already exists')
    if which("zstd") is None:
        raise FileNotFoundError("'zstd' must be installed and on your PATH")
    if not os.path.isfile(archive):
        raise FileNotFoundError(f"No such archive: {archive!r}")

    cmd = ["zstd", "-d", f"-T{threads}", "--rm", archive]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Decompression failed: {e}")

    if not os.path.isfile(abs_target):
        raise RuntimeError(f"Extraction failed, no file: {abs_target!r}")

    return abs_target
