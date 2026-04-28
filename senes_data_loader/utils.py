import json
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from openhexa.sdk import current_run, workspace
from openhexa.sdk.datasets.dataset import DatasetVersion
from rapidfuzz import fuzz, process
from sqlalchemy import create_engine
from unidecode import unidecode


class ColumnMissingError(Exception):
    """Custom exception for missing cols."""


def load_configuration_files(pipeline_path: Path) -> dict:
    """Load all configuration files and return them as a dictionary.

    Returns
    -------
        dict: A dictionary containing all configuration data.
    """
    # Load configuration files
    config_dir = pipeline_path / "config"
    config_specs = [
        ("required_columns", list),
        ("province_list", list),
        ("province_fixes", dict),
        ("espece_list", list),
        ("espece_fixes", dict),
        ("maladie_list", list),
        ("maladie_fixes", dict),
        ("zoonotic_diseases", list),
    ]

    return {name: load_json_key(config_dir / f"{name}.json", name, dtype) for name, dtype in config_specs}


def push_data_to_db_table(
    table_name: str,
    dataframe: pd.DataFrame | None = None,
    file_path: Path | None = None,
    db_url: str | None = None,
) -> None:
    """Push data to a database table.

    Parameters
    ----------
    table_name : str
        The name of the table to update or create.
    dataframe : pd.DataFrame | None
        The DataFrame containing the data to push to the table. If None, data will be read from file_path.
    file_path : Path | None
        The path to the file containing the data to push to the table. If None,
            data will be taken from the 'data' parameter.
    db_url : str | None
        The database URL to connect to. If None, the workspace database URL will be used.
    """
    current_run.log_info(f"Pushing data to table : {table_name}")

    if not table_name:
        raise ValueError("Table name cannot be None")

    if dataframe is None and file_path is None:
        raise ValueError("You must provide either a dataframe (pandas) or a file_path")

    if file_path is not None:
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist")
        df = pd.read_parquet(file_path)
    else:
        df = dataframe.copy()

    if df.empty:
        raise ValueError(f"DataFrame is empty, cannot create DB table '{table_name}'")

    if db_url:
        database_url = db_url
    else:
        # Use the workspace database URL if not provided
        database_url = workspace.database_url

    try:
        # Create engine
        dbengine = create_engine(database_url)
        df.to_sql(table_name, dbengine, index=False, if_exists="replace", chunksize=4096)
    except Exception as e:
        raise Exception(f"Error creating table '{table_name}' with file {file_path}: {e}") from e


def load_json_key(path: str, key: str, expected_type: type) -> object:
    """Load and return a value from a JSON file by key and type.

    Parameters
    ----------
    path : str
        Path to the JSON file.
    key : str
        The key corresponding to the value in the JSON file.
    expected_type : type
        The expected Python type (e.g., dict or list).

    Returns
    -------
        object: The value corresponding to the specified key and type.
    """
    with Path.open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if key in data and isinstance(data[key], expected_type):
            return data[key]
        raise ValueError(f"Key '{key}' not found or is not of type {expected_type.__name__} in {path}.")


def concat_and_format_table(data_list: list[pd.DataFrame], required_cols: list) -> pd.DataFrame:
    """Create the Senes tables.

    Returns
    -------
        pd.DataFrame The concatenated and formatted data.
    """
    dfs = [item["data"][required_cols] for item in data_list]
    table = pd.concat(dfs, axis=0, ignore_index=True)
    table = table.drop_duplicates()
    # Capitalise all string columns to simplify cleaning
    table = table.map(lambda x: x.strip() if isinstance(x, str) else x)
    return convert_month_names(table, "MOIS")


def convert_month_names(data: pd.DataFrame, month_col: str) -> pd.DataFrame:
    """Convert numeric month values to French month names in the specified column.

    Returns
    -------
        pd.DataFrame: The DataFrame with converted month names.
    """
    if month_col not in data.columns:
        raise ColumnMissingError(f"Colonne '{month_col}' est manquante dans les données.")

    # Convertime datetime to month
    data[month_col] = data[month_col].apply(lambda m: m.strftime("%m") if isinstance(m, datetime) else m)

    # Replace numeric month values with French names
    month_mapping = {
        "01": "Janvier",
        "02": "Février",
        "03": "Mars",
        "04": "Avril",
        "05": "Mai",
        "06": "Juin",
        "07": "Juillet",
        "08": "Août",
        "09": "Septembre",
        "10": "Octobre",
        "11": "Novembre",
        "12": "Décembre",
        "Aout": "Août",
    }

    data[month_col] = data[month_col].replace(month_mapping)
    return data


def clean_string(col_name: str) -> str:
    """Clean a string by removing accents, replacing spaces with underscores, and removing non-letter characters.

    Returns
    -------
        str The cleaned string.
    """
    # Remove accents
    col_name = unidecode(col_name)
    col_name = col_name.strip()
    # Replace spaces with underscores
    col_name = col_name.replace("-", "_")
    col_name = col_name.replace(" ", "_")
    # Remove all non-letter characters
    col_name = re.sub(r"[^a-zA-Z0-9/_-]", "", col_name)
    return col_name.upper()


def extract_week_and_year(filename: str) -> tuple:
    """Extract week numbers and year from a filename.

    Returns
    -------
        tuple: A tuple containing a list of week numbers and the year.
    """
    # Find all week numbers after '_S' or '_SE' (with or without underscore)
    weeks = re.findall(r"(?:_|^)S(?:E)?(\d{2})", filename)
    # Find the year after the last underscore
    year_match = re.search(r"_(\d{4})", filename)
    year = year_match.group(1) if year_match else None
    return weeks, year


def validate_columns(df: pd.DataFrame, required_columns: list) -> bool:
    """Validate the Senes data."""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ColumnMissingError(f"Colonnes manquantes : {missing_columns}")


def validate_year(df: pd.DataFrame, filename: str) -> bool:
    """Validate the year in the Senes data."""
    _, year = extract_week_and_year(filename)
    if not year:
        raise ValueError("Année non trouvée dans le nom du fichier.")
    if year is not None and int(year) not in df["ANNEE"].astype(int).unique():
        raise ValueError("Colonne 'ANNEE' ne correspond pas à l'année extraite du nom du fichier.")


def validate_weeks(df: pd.DataFrame, filename: str) -> bool:
    """Validate the week numbers in the Senes data."""
    weeks, _ = extract_week_and_year(filename)

    if not weeks:
        raise ValueError("Semaine non trouvée dans le nom du fichier.")

    se_numeric = pd.to_numeric(df.SE, errors="coerce")
    if not ((se_numeric.dropna() >= 1) & (se_numeric.dropna() <= 53)).all():
        raise ValueError("Colonne 'SE' doit contenir uniquement des numéros de semaine valides (1-53).")

    se_values = set(pd.to_numeric(df.SE, errors="coerce").dropna().astype(int).unique())
    week_values = set(int(x) for x in weeks if pd.notna(x))
    # validate if the weeks are present in the SE column
    if weeks and not week_values.issubset(se_values):
        raise ValueError(
            "Colonne 'SE' contient des numéros de semaine qui ne correspondent pas aux semaines du nom du fichier."
        )


def clean_province_series(province_series: pd.Series, province_fixes: dict) -> pd.Series:
    """Clean province names in a Series.

    Returns
    -------
        pd.Series: The cleaned province names.
    """
    s = province_series.apply(clean_string)
    s = s.replace(province_fixes)
    return s.str.replace("_", " ").str.title()


def fuzzy_correct_column(
    df: pd.DataFrame,
    col: str,
    ref_values: list,
    threshold: int = 85,
) -> tuple[pd.DataFrame, dict, list]:
    """Fuzzy-match and correct a DataFrame column against a reference list of canonical values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (a copy will be modified and returned).
    col : str
        Name of the column to correct (e.g. "ESPECE").
    ref_values : list
        List of canonical/reference values to match against.
    threshold : int, optional
        Minimum fuzzy score (0-100) to accept a match. Default is 85.


    Returns
    -------
        df          : Copy of the DataFrame with `col` corrected in-place.
        unmatched   : List of values that fell below the threshold.
        corrections : Dict mapping original unrecognized values to their corrected matches.
    """
    df = df.copy()
    match_col = f"{col}_match"
    corrections = {}
    unmatched = []

    # --- Normalise the column for matching ---
    df[match_col] = df[col].apply(lambda s: clean_string(s) if isinstance(s, str) else s)

    # --- Build normalised → canonical lookup ---
    ref_lookup = {clean_string(s): s for s in ref_values}
    ref_keys = list(ref_lookup.keys())

    # --- Find values not in the reference list ---
    not_found = df.loc[~df[match_col].isin(ref_keys), match_col].dropna().unique()

    # --- Fuzzy-match each unrecognised value ---
    for value in not_found:
        best = process.extractOne(value, ref_keys, scorer=fuzz.WRatio)
        if best and best[1] >= threshold:
            corrections[value] = best[0]
        else:
            unmatched.append(f"Valeur dans la colonne {col} : '{value}' non identifiée (score: {best[1]})")

    # --- Apply corrections and remap to canonical values ---
    if corrections:
        df[match_col] = df[match_col].replace(corrections)

    df[col] = df[match_col].map(ref_lookup).fillna(df[col])
    df = df.drop(columns=[match_col])

    return df, corrections, unmatched


def swap_column_values_if_needed(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    col1_ref: list,
    col2_ref: list,
) -> pd.DataFrame:
    """Detect and correct potential value swaps between two columns based on reference lists.

    Returns
    -------
        pd.DataFrame: The DataFrame with corrected column values if a swap was detected.
    """
    if not col1 or not col2:
        return df
    if col1 not in df.columns or col2 not in df.columns:
        return df

    df = df.copy()

    # Compute swap mask first so warnings only cover unfixable rows
    mask_swap = df[col1].isin(col2_ref) & df[col2].isin(col1_ref)

    # Auto-swap rows where both columns are clearly reversed

    if mask_swap.any():
        n = mask_swap.sum()
        current_run.log_warning(f"Correction: inversion {col1}<->{col2} sur {n} lignes.")
        for idx in df[mask_swap].index:
            current_run.log_info(f"Correction: '{df.loc[idx, col1]}' <-> '{df.loc[idx, col2]}'")
        df.loc[mask_swap, [col1, col2]] = df.loc[mask_swap, [col2, col1]].to_numpy()

    return df


def send_mail(text: str, mailgun_key: str, email_list: list, email_from: str, email_subject: str) -> None:
    """Send an email notification with the given text."""
    current_run.log_info("Envoi de l'email de notification.")
    return requests.post(
        "https://api.eu.mailgun.net/v3/notifications.openhexa.org/messages",
        auth=("api", f"{mailgun_key}"),
        data={
            "from": f"{email_from} <info@notifications.openhexa.org>",
            "to": ", ".join(email_list),
            "subject": email_subject,
            "text": text,
        },
    )


def add_files_to_dataset(
    dataset_id: str,
    file_paths: list[Path],
    ds_version_prefix: str = "DS",
    ds_desc: str = "Dataset version created by pipeline",
) -> bool:
    """Add files to a new dataset version.

    Parameters
    ----------
    dataset_id : str
        The ID of the dataset to which files will be added.
    file_paths : list[Path]
        A list of file paths to be added to the dataset.
    ds_version_prefix : str, optional
        The prefix for the dataset version name. Default is "DS".
    ds_desc : str, optional
        The description for the dataset version. Default is "Dataset version created by pipeline".

    Returns
    -------
    bool
        True if at least one file was added successfully, False otherwise.

    Raises
    ------
    ValueError
        If the dataset ID is not specified.
    """
    if not dataset_id:
        raise ValueError("Dataset ID is not specified.")

    supported_extensions = {".parquet", ".csv", ".geojson", ".json"}
    added_any = False
    new_version = None

    for src in file_paths:
        if not src.exists():
            current_run.log_warning(f"File not found: {src}")
            continue

        ext = src.suffix.lower()
        if ext not in supported_extensions:
            current_run.log_warning(f"Unsupported file format: {src.name}")
            continue

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp_path = Path(tmp.name)

            shutil.copy2(src, tmp_path)

            if not added_any:
                new_version = get_new_dataset_version(
                    ds_id=dataset_id,
                    prefix=ds_version_prefix,
                    ds_desc=ds_desc,
                )
                current_run.log_info(f"New dataset version created: {new_version.name}")
                added_any = True

            new_version.add_file(str(tmp_path), filename=src.name)
            current_run.log_info(f"File {src.name} added to dataset version: {new_version.name}")

        except Exception as e:
            current_run.log_warning(f"File {src.name} cannot be added: {e}")

        finally:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink()

    if not added_any:
        current_run.log_warning("No valid files found. Dataset version was not created.")
        return False

    return True


def get_new_dataset_version(ds_id: str, prefix: str = "DS", ds_desc: str = "Dataset") -> DatasetVersion:
    """Create and return a new dataset version.

    Parameters
    ----------
    ds_id : str
        The ID of the dataset for which a new version will be created.
    prefix : str, optional
        Prefix for the dataset version name (default is "DS").
    ds_desc : str, optional
        Description for the dataset (default is "Dataset version created by pipeline").

    Returns
    -------
    DatasetVersion
        The newly created dataset version.

    Raises
    ------
    Exception
        If an error occurs while creating the new dataset version.
    """
    try:
        dataset = workspace.get_dataset(ds_id)
    except Exception as e:
        current_run.log_warning(f"Error retrieving dataset: {ds_id}")
        current_run.log_debug(f"Error retrieving dataset {ds_id}: {e}")
        dataset = None

    if dataset is None:
        current_run.log_warning(f"Creating new Dataset with ID: {ds_id}")
        dataset = workspace.create_dataset(name=ds_id.replace("-", "_").upper(), description=ds_desc)

    version_name = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    try:
        return dataset.create_version(version_name)
    except Exception as e:
        raise Exception("An error occurred while creating the new dataset version.") from e
