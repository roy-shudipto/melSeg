import click
import pathlib
import pandas as pd
from loguru import logger

from melseg.defaults import LOG_EXTENSION, LOG_HEADERS


def run_analysis(log_list) -> pd.DataFrame:
    # initiate a pandas->DataFrame for analysis log
    df_analysis_log = pd.DataFrame()

    for log_dict in log_list:
        # get log as pandas->DataFrame
        df_log = log_dict["df"]

        # find the minimum val-loss
        val_loss_min = min(df_log["VAL LOSS"])

        # get pandas->DataFrame for minimum val-loss
        df_val_loss_min = df_log.loc[df_log["VAL LOSS"] == val_loss_min]

        # there is a chance of having multiple rows for minimum val-loss
        # find the row with maximum epoch
        epoch_max = max(df_val_loss_min["EPOCH"])

        # now, get pandas->DataFrame for minimum val-loss and maximum epoch
        df_best_row = df_val_loss_min.loc[df_val_loss_min["EPOCH"] == epoch_max]

        # insert log reference
        df_best_row.insert(0, "LOG REFERENCE", pathlib.Path(log_dict["path"]).stem)

        # append row to analysis log
        df_analysis_log = (
            df_best_row
            if df_analysis_log.empty is True
            else pd.concat([df_analysis_log, df_best_row], axis=0)
        )

    # sort analysis-DataFrame on "LOG REFERENCE"
    df_analysis_log = df_analysis_log.sort_values("LOG REFERENCE")

    # calculate average of analysis-DataFrame
    df_temp = df_analysis_log.drop("LOG REFERENCE", axis=1)
    df_avg = pd.DataFrame(dict(df_temp.mean(axis=0).round(4)), index=[0])
    df_avg.insert(0, "LOG REFERENCE", "Average")

    # append average as a row to analysis-DataFrame
    df_analysis_log = pd.concat([df_analysis_log, df_avg], axis=0)

    return df_analysis_log.reset_index(drop=True)


@click.command()
@click.option(
    "--root",
    type=str,
    required=True,
    help="Path to a directory with .csv logs.",
)
@click.option(
    "--outpath",
    type=str,
    required=True,
    help="Path to an output .csv file.",
)
def analyze(root, outpath) -> None:
    # convert paths from string to pathlib.Path
    root = pathlib.Path(root)
    outpath = pathlib.Path(outpath)

    # check: root exists and is a directory
    if not root.exists() or not root.is_dir():
        logger.error(f"Root: [{root}] is not valid. It is expected to be a directory.")
        exit(1)

    # check: output file has a valid extension
    if outpath.suffix.lower() != LOG_EXTENSION.lower():
        logger.error(f"Output: [{outpath}] needs to have {LOG_EXTENSION} extension.")
        exit(1)

    # read logs in a list as pandas->DataFrame
    log_list = []
    for log_path in root.iterdir():
        # skip log file with invalid extension
        if log_path.suffix.lower() != LOG_EXTENSION.lower():
            logger.debug(
                f"Skipping file: [{log_path}] as it does not have [{LOG_EXTENSION}]"
                " extension."
            )
            continue

        # read log file as pandas->DataFrame. Skip if unable to read.
        try:
            df = pd.read_csv(log_path)
        except UnicodeDecodeError or RuntimeError:
            logger.debug(f"Skipping file: [{log_path}] as it is failed to be read.")
            continue

        # skip log file with invalid column list
        if sorted(df.columns.tolist()) != sorted(LOG_HEADERS):
            logger.debug(
                f"Skipping file: [{log_path}] as it is has invalid columns:"
                f" {df.columns.tolist()}. Expected columns are: {LOG_HEADERS}."
            )
            continue

        # skip log with no data
        if df.empty is True:
            logger.debug(f"Skipping file: [{log_path}] as it has no data.")
            continue

        # append log as pandas->DataFrame
        log_list.append({"df": df, "path": log_path})

    # check: valid number of logs
    if len(log_list) > 0:
        logger.info(f"{len(log_list)} logs are found at [{root}].")
    else:
        logger.error(f"No valid log is found at [{root}].")
        exit(1)

    # run analysis
    analysis_log = run_analysis(log_list)

    # display analysis-DataFame
    logger.info(f"\n{analysis_log}")

    # save analysis-log
    analysis_log.to_csv(outpath, index=False)
    logger.info(f"Analysis-log is saved as: {outpath}")


if __name__ == "__main__":
    logger.info("This pipeline analyzes log files of .csv format.")
    analyze()
