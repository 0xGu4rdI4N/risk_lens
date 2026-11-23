"""Local SparkSession factory."""
import os
import shutil
import sys
from typing import Optional

_BREW_JDK = "/opt/homebrew/opt/openjdk@17"


def _java_works() -> bool:
    """macOS ships a /usr/bin/java stub that exits non-zero when no JDK is installed."""
    import subprocess
    try:
        return subprocess.run(["java", "-version"], capture_output=True, timeout=15).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _ensure_java_home() -> None:
    if os.environ.get("JAVA_HOME") or _java_works():
        return
    if os.path.exists(os.path.join(_BREW_JDK, "bin", "java")):
        os.environ["JAVA_HOME"] = _BREW_JDK
        os.environ["PATH"] = os.path.join(_BREW_JDK, "bin") + os.pathsep + os.environ.get("PATH", "")


def java_available() -> bool:
    _ensure_java_home()
    return _java_works()


def get_spark(app_name: str = "risklens", master: str = "local[*]", shuffle_partitions: int = 8,
              extra_conf: Optional[dict] = None):
    """Create (or reuse) a local SparkSession whose Python workers use this interpreter."""
    _ensure_java_home()
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    from pyspark.sql import SparkSession

    builder = (SparkSession.builder.appName(app_name).master(master)
               .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
               .config("spark.sql.execution.arrow.pyspark.enabled", "true")
               .config("spark.ui.enabled", "false")
               .config("spark.driver.bindAddress", "127.0.0.1"))
    for k, v in (extra_conf or {}).items():
        builder = builder.config(k, v)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark
