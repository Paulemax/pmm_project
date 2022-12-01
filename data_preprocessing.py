import functools
from typing import Optional
import pandas as pd
from returns import pipeline
from returns.pointfree import bind
from returns.maybe import Maybe, maybe
from pathlib import Path
import requests


# Constants 
_RAW_DATA_PATH: Path = Path("./data/raw/")
_RAW_FILE_NAME = "raw_data.csv"
# TODO save processed data
# _PROCESSED_FILE_NAME = "processed_data.csv"
# PROCESSED_DATA_PATH: Path = Path("./data/processed/")
_URL: str = "http://erddap.marine.ie/erddap/tabledap/IWBNetwork.csv?station_id,longitude,latitude,time,AtmosphericPressure,WindDirection,WindSpeed,Gust,WaveHeight,WavePeriod,MeanWaveDirection,Hmax,AirTemperature,DewPoint,SeaTemperature,RelativeHumidity,QC_Flag"


def _load_data(path: Path) -> pd.DataFrame:
    raw_data = pd.read_csv(path, delimiter=",", skiprows=[1])
    return raw_data


@maybe
def _download_data(path: Path, url: str, file_name: str) -> Optional[Path]:
    file = path / file_name
    if file.exists(): 
        return file

    response = requests.get(url)
    if response.status_code != 200:
        return None 

    path.mkdir(parents=True, exist_ok=True)

    with open(file, "wb") as f:
        f.write(response.content)
    return file


def _set_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """ Sets the columns to the correct type"""
    df_c = df.copy()
    df_c = df.astype({"station_id": "category", "QC_Flag": "bool"})
    df_c["time"] = pd.to_datetime(df_c["time"])
    return df_c


def _handle_nan_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes nan values from the DataFrame in two steps.
    1. Remove columns with to many nan values > 1/2 of dataset
    2. Remove rows with nan values
    """
    # Drop columns with to many nan values 
    df_c = df.drop(columns=["Hmax", "DewPoint", "MeanWaveDirection", "QC_Flag"]) 
    df_c = df_c.dropna()
    return df_c


def prepare_data() -> pd.DataFrame:
    data = pipeline.flow(
        _RAW_DATA_PATH,
        functools.partial(_download_data, url=_URL, file_name=_RAW_FILE_NAME),
        bind(_load_data),
        _set_dtypes,
        _handle_nan_values,
    )

    return data


def main() -> None:
    data = pipeline.flow(
        _RAW_DATA_PATH,
        functools.partial(_download_data, url=_URL, file_name=_RAW_FILE_NAME),
        bind(_load_data),
        _set_dtypes,
        _handle_nan_values,
    )
    print(data)

if __name__ == "__main__":
    main()