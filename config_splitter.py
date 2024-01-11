import click
import pathlib
import yaml
from loguru import logger

from melseg.config import parse_config
from melseg.defaults import CONFIG_EXTENSION


@click.command()
@click.option(
    "--config",
    type=str,
    required=True,
    help="Path to the training-config [.yaml] file.",
)
@click.option(
    "--outdir",
    type=str,
    required=True,
    help="Path to the output directory.",
)
def split_config(config, outdir) -> None:
    # convert paths from string to pathlib.Path
    config_path = pathlib.Path(config)
    outdir = pathlib.Path(outdir)

    # check: output-directory is a directory if exists
    if outdir.exists() and not outdir.is_dir():
        logger.error(
            f"outdir: [{outdir}] is not valid. It is expected to be a directory."
        )
        exit(1)
    else:
        outdir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Successfully created directory: {outdir}")

    # convert config from str to pathlib.Path
    config_path = pathlib.Path(config)

    # multiple training-configs will be generated if parameters contain a list
    # of values instead of a single value.
    config_variations = parse_config(config_path)
    logger.info(f"Number of training-config variations: {len(config_variations)}")

    # generate individual sweeps of training-config
    for idx, config_variation in enumerate(config_variations):
        ref = str(idx + 1).zfill(2)  # example: 01, 02, ... 30
        out_path = outdir / pathlib.Path(f"training_config_{ref}").with_suffix(
            CONFIG_EXTENSION
        )

        # include config-reference
        config_variation["CONFIG_REFERENCE"] = f"training_config_{ref}"

        # save
        with open(out_path, "w") as file:
            yaml.dump(config_variation, file, sort_keys=False)

    logger.info(f"Successfully splitted [{config_path}] and saved at [{outdir}].")


if __name__ == "__main__":
    logger.info("This pipeline generates individual sweeps from a training-config.")
    split_config()
