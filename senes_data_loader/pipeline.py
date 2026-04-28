from pathlib import Path

import pandas as pd
from openhexa.sdk import current_run, pipeline, workspace
from unidecode import unidecode
from utils import (
    ColumnMissingError,
    clean_province_series,
    clean_string,
    concat_and_format_table,
    fuzzy_correct_column,
    load_configuration_files,
    load_json_key,
    push_data_to_db_table,
    send_mail,
    swap_column_values_if_needed,
    validate_columns,
    validate_weeks,
    validate_year,
)


@pipeline("senes_data_loader")
def senes_data_loader():
    """SENES data loader pipeline."""
    run_senes_data_loader()


def run_senes_data_loader():
    """Run the SENES data loader pipeline."""
    pipeline_path = Path(workspace.files_path) / "pipelines" / "senes_data_loader"
    pipeline_path.mkdir(parents=True, exist_ok=True)

    # Process the data
    senes_data = process_senes_data(pipeline_path)

    # Save the processed table as a parquet file (per year)
    last_year = senes_data.ANNEE.max()
    (pipeline_path / "processed").mkdir(parents=True, exist_ok=True)
    senes_data.to_parquet(pipeline_path / "processed" / f"senes_data_{last_year}.parquet", index=False)

    # push the data to the database
    current_run.log_info("Mise à jour de la table COD_SENES dans la base de données")
    push_data_to_db_table(
        table_name="COD_SENES",
        dataframe=senes_data,
        db_url=workspace.database_url,
    )

    # notify by email
    notify_by_email(pipeline_path, senes_data)
    current_run.log_info("Validation du fichier terminée.")


def process_senes_data(pipeline_path: Path) -> pd.DataFrame:
    """Process the Senes data.

    Returns
    -------
        pd.DataFrame The processed data.
    """
    config = load_configuration_files(pipeline_path)

    # Load excel files
    senes_data_list = load_senes_data(Path(workspace.files_path) / "SENES" / "data")

    # validate
    validate_senes_data_format(data_list=senes_data_list, required_columns=config["required_columns"])

    # Format SENES table
    senes_table = concat_and_format_table(
        data_list=senes_data_list, required_cols=config["required_columns"] + ["SOURCE_FILE"]
    )

    # Handle value standardization (let's output the process table each time)
    senes_table = handle_province_names(senes_table, config["province_list"], config["province_fixes"])
    senes_table = handle_espece_and_maladie_names(
        df=senes_table,
        espece_list=config["espece_list"],
        espece_fixes=config["espece_fixes"],
        maladie_list=config["maladie_list"],
        maladie_fixes=config["maladie_fixes"],
    )
    senes_table = handle_zoonotic_diseases(senes_table, config["zoonotic_diseases"])

    # Setting value to NAs if they are not of type int
    cols = [
        "POR",
        "CAS",
        "ABATTU",
        "MORTALITE",
        "DETRUIT",
        "VACCINE",
        "LATITUDE",
        "LONGITUDE",
        "TX_MORTALITE",
        "TX_LETALITE",
    ]
    senes_table[cols] = senes_table[cols].apply(pd.to_numeric, errors="coerce")
    senes_table["SE"] = pd.to_numeric(senes_table["SE"], errors="coerce").astype("Int64")

    # logs
    current_run.log_info(f"Nombre d'enregistrements dans le base de donnees: {senes_table.shape[0]}")
    max_year = senes_table["ANNEE"].max()
    max_week = senes_table[senes_table["ANNEE"] == max_year]["SE"].max()
    current_run.log_info(f"Donnees presentes jusqu'a l'annee: {max_year}, semaine: {int(max_week)}")
    return senes_table


def load_senes_data(data_folder: Path) -> pd.DataFrame:
    """Load the Senes data from the CSV file.

    Returns
    -------
        pd.DataFrame The loaded data.
    """
    data_folder.mkdir(parents=True, exist_ok=True)
    year_folders = sorted([y for y in data_folder.glob("20*")])
    if not year_folders:
        raise FileNotFoundError(f"Erreur : Dossier de données annuelles introuvable dans {data_folder}")

    # collect data per year folder
    senes_data = []
    for year_folder in year_folders:
        senes_files = sorted(list(year_folder.glob("*.xlsx")))
        if not senes_files:
            current_run.log_warning(f"Warning: No .xlsx files found in {year_folder}/")
            continue

        current_run.log_info(f"{len(senes_files)} fichiers trouvés dans {year_folder.name}/")
        try:
            for f in senes_files:
                df = pd.read_excel(f)
                df["SOURCE_FILE"] = f.name  # NOTE: Column added
                senes_data.append(
                    {
                        "filename": f.name,
                        "data": df,
                    }
                )
        except Exception as e:
            current_run.log_warning(f"Erreur sur fichier {f.name}: {e}")
            continue

    current_run.log_info(f"Dernier fichier : {f.name}")
    return senes_data


def validate_senes_data_format(data_list: list[dict], required_columns: list):
    """Validate the Senes data."""
    for item in data_list:
        try:
            f = item["filename"]
            df = item["data"]
            df.columns = [clean_string(col) for col in df.columns]  # standard names for validation
            validate_columns(df, required_columns)
            validate_year(df, f)
            validate_weeks(df, f)
        except ColumnMissingError as e:
            current_run.log_error(f"Colonnes manquantes dans le fichier {f}: {e}")
            raise e
        except ValueError as e:
            current_run.log_warning(f"Erreur de validation dans le fichier {f}: {e}")
            continue


def handle_province_names(df: pd.DataFrame, province_list: list, province_fixes: dict) -> pd.DataFrame:
    """Handle the matching of organizational units.

    Returns
    -------
        pd.DataFrame The data with cleaned province names.
    """
    df = df.copy()
    df["PROVINCE"] = clean_province_series(df["PROVINCE"], province_fixes)

    # Vérification du nom des provinces
    for prov in df["PROVINCE"].unique():
        if prov not in province_list:
            log_msg = f"ATTENTION: La province {prov} n'est pas dans la liste des provinces de RDC"
            current_run.log_warning(log_msg)

    for year in sorted(df["ANNEE"].unique()):
        current_run.log_info(
            f"Nombre de Provinces ayant rapporté en {year}: {len(df[df['ANNEE'] == year]['PROVINCE'].unique())}/26"
        )

    return df


def handle_espece_and_maladie_names(
    df: pd.DataFrame, espece_list: list, espece_fixes: dict, maladie_list: list, maladie_fixes: dict
) -> pd.DataFrame:
    """Handle the matching of species and disease names.

    Returns
    -------
        pd.DataFrame The data with cleaned species and disease names.
    """
    df = df.copy()

    # Clean ESPECE and MALADIE columns with basic formatting and fixes before fuzzy matching
    df["ESPECE"] = df["ESPECE"].apply(lambda x: unidecode(x).strip().capitalize() if isinstance(x, str) else x)
    df["ESPECE"] = df["ESPECE"].replace(espece_fixes)

    df["MALADIE"] = df["MALADIE"].apply(lambda x: unidecode(x).strip().capitalize() if isinstance(x, str) else x)
    df["MALADIE"] = df["MALADIE"].replace(maladie_fixes)

    # Apply inverse fix to correct  misplacements between ESPECE and MALADIE before fuzzy matching
    df["ESPECE"] = df["ESPECE"].replace(maladie_fixes)
    df["MALADIE"] = df["MALADIE"].replace(espece_fixes)

    # Check for potential value swaps between ESPECE and MALADIE columns and correct them
    df = swap_column_values_if_needed(df, "ESPECE", "MALADIE", espece_list, maladie_list)

    df = _handle_espece_names(df, espece_list)
    df = _handle_maladie_names(df, maladie_list)

    return _clean_table(df, espece_list, maladie_list)


def _handle_espece_names(df: pd.DataFrame, espece_list: Path) -> pd.DataFrame:
    """Handle the matching of species names.

    Returns
    -------
        pd.DataFrame The data with cleaned species names.
    """
    df = df.copy()

    # If three first letters are the same, replace with espece_selection otherwise delete
    def match_species(value: str) -> str:
        for species in espece_list:
            if str(value).lower().startswith(species[:3].lower()):
                return species
        return value

    # Apply function to the dataframe
    df["ESPECE"] = df["ESPECE"].apply(match_species)
    # Convert Pou_ to Volaille
    df["ESPECE"] = df["ESPECE"].apply(lambda x: "Volaille" if str(x).lower().startswith("pou") else x)

    df, _, unmatched = fuzzy_correct_column(df, "ESPECE", espece_list, threshold=85)

    if unmatched:
        for msg in unmatched:
            current_run.log_warning(msg)

    return df


def _handle_maladie_names(df: pd.DataFrame, maladie_list: Path) -> pd.DataFrame:
    """Handle the matching of disease names.

    Returns
    -------
        pd.DataFrame The data with cleaned disease names.
    """
    df = df.copy()

    # Automatic formating
    df["MALADIE"] = df["MALADIE"].apply(lambda x: "Dermatose" if str(x).startswith("Derm") else x)
    df["MALADIE"] = df["MALADIE"].apply(lambda x: "Newcastle" if "astle" in str(x) else x)

    df, _, unmatched = fuzzy_correct_column(df, "MALADIE", maladie_list, threshold=85)

    if unmatched:
        for msg in unmatched:
            current_run.log_warning(msg)

    return df


def handle_zoonotic_diseases(df: pd.DataFrame, zoonotic_diseases: list) -> pd.DataFrame:
    """Handle the classification of diseases into zoonotic and non-zoonotic.

    Returns
    -------
        pd.DataFrame: df with new column "MALADIE_TYPE" with diseases as "Zoonotique" or "Non-zoonotique".
    """
    df = df.copy()
    # Les maladies zoonotiques ou zoonoses doivent être ajouté ci dessous;
    # sinon elles seront classifier de maladie non-zoonotique
    df["MALADIE_TYPE"] = df["MALADIE"].apply(lambda x: "Zoonotique" if x in zoonotic_diseases else "Non-zoonotique")
    return df


def _clean_table(df: pd.DataFrame, espece_list: list, maladie_list: list) -> pd.DataFrame:
    """Clean the Senes table by removing rows with ESPECE not in espece_sel and MALADIE not in maladie_sel.

    Returns
    -------
        pd.DataFrame The cleaned data.
    """
    df = df.copy()
    # Check if any unique species is not in espece_sel
    unique_species = df["ESPECE"].unique()
    not_in_espece_sel = [species for species in unique_species if species not in espece_list]
    if not_in_espece_sel:
        current_run.log_info(f"Espèces exclues de la base de données: {not_in_espece_sel}")

    # Check if any unique maladies is not in maladie_sel
    unique_maladies = df["MALADIE"].unique()
    not_in_maladie_sel = [maladie for maladie in unique_maladies if maladie not in maladie_list]
    if not_in_maladie_sel:
        current_run.log_info(f"Maladies exclues de la base de données: {not_in_maladie_sel}")

    # Remove all rows from SENES with ESPECE not in espece_sel and MALADIE not in maladie_sel
    df = df[df["ESPECE"].isin(espece_list)]
    return df[df["MALADIE"].isin(maladie_list)]


def email_report(df: pd.DataFrame) -> str:
    """Send an email report.

    Returns
    -------
        str: The email report text.
    """
    # Filtrer l'année la plus récente
    latest_year = df["ANNEE"].max()
    latest_year_senes = df[df["ANNEE"] == latest_year]

    # Deux dernières semaines
    last_two_weeks = latest_year_senes["SE"].sort_values().unique()[-2:]
    two_last_weeks_senes = latest_year_senes[latest_year_senes["SE"].isin(last_two_weeks)]
    top_3_maladies = two_last_weeks_senes.groupby("MALADIE")["CAS"].sum().sort_values(ascending=False).head(3)

    message = f"""Bonjour,

    La base de données SENES a été mise à jour avec les données les plus récentes.\n
    🦠Alertes épizootiques pour les semaines {int(last_two_weeks[0])} et {int(last_two_weeks[1])} 
    de l'année {latest_year} :
    """

    for mal in top_3_maladies.index:
        total_cas = top_3_maladies[mal]
        message += f"\n🔹 {mal} — {int(total_cas)} cas détectés:\n"

        top_prov_mal = (
            two_last_weeks_senes[two_last_weeks_senes["MALADIE"] == mal]
            .groupby("PROVINCE")["CAS"]
            .sum()
            .sort_values(ascending=False)
            .head(3)
        )

        for prov, cas in top_prov_mal.items():
            message += f"   • {prov} : {int(cas)} cas\n"

    message += "\nCordialement,\nL'équipe Bluesquare."
    return message


def notify_by_email(pipeline_path: Path, senes_data: pd.DataFrame) -> None:
    """Notify by email with the latest data.

    Parameters
    ----------
    pipeline_path : Path
        The path to the pipeline files.
    senes_data : pd.DataFrame
        The processed Senes data.
    """
    email_list = load_json_key(pipeline_path / "config" / "email_list.json", "email_list", list)
    try:
        report = email_report(senes_data)
        mailgun_key = workspace.get_connection("mailgun").key
        send_mail(
            text=report,
            mailgun_key=mailgun_key,
            email_list=email_list,
            email_from="senes update",
            email_subject="Table de données SENES",
        )
        current_run.log_info("Email de notification envoyé avec succès.")
    except Exception as e:
        current_run.log_error(f"Erreur lors de l'envoi de l'email de notification: {e}")
        raise e


if __name__ == "__main__":
    senes_data_loader()
