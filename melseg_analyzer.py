"""
This pipeline analyzes log files of .csv format.

Example usage:
python3 melseg_analyzer.py --help
"""

import argparse
import pathlib
import pandas as pd
import yaml
from loguru import logger

from melseg.defaults import ANALYSIS_LOG_NAME, CONFIG_NAME, LOG_EXTENSION, LOG_HEADERS


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


parser = argparse.ArgumentParser()
parser.add_argument(
    "--dir", nargs="+", required=True, help="Path to directory with .csv logs."
)
parser.add_argument(
    "--out",
    nargs="+",
    required=False,
    help=f"Path to output .csv file (default: [...]/{ANALYSIS_LOG_NAME})",
)
args = parser.parse_args()

if __name__ == "__main__":
    logger.info("This pipeline analyzes log files of .csv format.")

    # parse arguments
    dir_args = args.dir
    out_args = args.out

    # check: number of [--dir] arguments doesn't match the number of [--our] arguments
    if out_args and len(dir_args) != len(out_args):
        logger.error(f"Given [--dir]: {dir_args}")
        logger.error(f"Given [--out]: {out_args}")
        logger.error(
            f"If [--out] is given, the number of [--dir] arguments ({len(dir_args)})"
            f" needs to match the number of [--out] arguments ({len(out_args)})."
        )
        exit(1)

    # [--out] is given, and the number of arguments matches the number of [--dir] arguments
    elif out_args and len(dir_args) == len(out_args):
        dir_list = [pathlib.Path(dir_arg) for dir_arg in dir_args]
        out_list = [pathlib.Path(out_arg) for out_arg in out_args]

    # [--out] is not given
    elif out_args is None:
        dir_list = [pathlib.Path(dir_arg) for dir_arg in dir_args]
        out_list = [dir_arg / pathlib.Path(ANALYSIS_LOG_NAME) for dir_arg in dir_args]

    # run analysis
    for dir, out in zip(dir_list, out_list):
        logger.info(f"Working on, Directory path: [{dir}] | Output file path: [{out}]")

        # check: directory-path exists and is a directory
        if not dir.exists() or not dir.is_dir():
            logger.error(
                f"Directory path: [{dir}] is not valid. It is expected to be a"
                " directory."
            )
            exit(1)

        # check: output file has a valid extension
        if out.suffix.lower() != LOG_EXTENSION.lower():
            logger.error(
                f"Output file: [{out}] needs to have {LOG_EXTENSION} extension."
            )
            exit(1)

        # read config
        config_path = dir / CONFIG_NAME
        try:
            config = yaml.safe_load(open(config_path.as_posix()))
        except FileNotFoundError:
            config = None
            logger.debug(f"Config file: {config_path} is not found.")

        # read logs in a list as pandas->DataFrame
        log_list = []
        for log_path in dir.iterdir():
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
            logger.info(f"{len(log_list)} logs are found at [{dir}].")
        else:
            logger.error(f"No valid log is found at [{dir}].")
            exit(1)

        # run analysis
        analysis_log = run_analysis(log_list)

        # display config (if found)
        if config:
            for k, v in config.items():
                logger.info(f"Training Config >> {k}: {v}")

        # display analysis-DataFame
        logger.info(f"Analysis:\n{analysis_log}")

        # save analysis-log
        analysis_log.to_csv(out, index=False)
        logger.info(f"Analysis-log is saved as: {out}\n")
