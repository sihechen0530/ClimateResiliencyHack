import requests
from locations import locations
from dates import dates
import multiprocessing
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import os

data_dir = "data"

url_template = "https://www.almanac.com/weather/history/WA/{}/{}"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
}

field_order = [
    "Minimum Temperature",
    "Mean Temperature",
    "Maximum Temperature",
    "Mean Sea Level Pressure",
    "Mean Dew Point",
    "Total Precipitation",
    "Visibility",
    "Snow Depth",
    "Mean Wind Speed",
    "Maximum Sustained Wind Speed",
    "Maximum Wind Gust",
]

fields = {
    "Minimum Temperature": ("temp_mn", "°F"),
    "Mean Temperature": ("temp", "°F"),
    "Maximum Temperature": ("temp_mx", "°F"),
    "Mean Sea Level Pressure": ("slp", None),
    "Mean Dew Point": ("dewp", "°F"),
    "Total Precipitation": ("prcp", "IN"),
    "Visibility": ("visib", "MI"),
    "Snow Depth": ("sndp", None),
    "Mean Wind Speed": ("wdsp", "MPH"),
    "Maximum Sustained Wind Speed": ("mxspd", "MPH"),
    "Maximum Wind Gust": ("gust", "MPH"),
}


def get_url(location, date):
    return url_template.format(location, date)


def get_weather_info(location, date):
    print("start processing", location, date)
    url = get_url(location, date)
    response = requests.get(url, headers=headers, timeout=5)

    if response.status_code == 200:
        webpage_content = response.text
        print(f"Page content successfully retrieved! {url}")
    else:
        print(f"Failed to retrieve page. Status code: {response.status_code}")
        return None

    soup = BeautifulSoup(webpage_content, "html.parser")

    data = [location, date]

    # Iterate over fields and find corresponding values
    for label in field_order:
        class_name, unit = fields[label]
        # Find the row by class
        row = soup.find("tr", class_=f"weatherhistory_results_datavalue {class_name}")
        if row:
            # Get the data if available, otherwise note as "No data"
            value_elem = row.find("span", class_="value")
            value = value_elem.text.strip() if value_elem else None
            # Append units if available and data is present
            if unit and value is not None:
                value = float(value)
            data.append(value)

    print("finished processing", location, date)
    return data

def get_batch_weather_info(location, multiple_dates):
    ret = []
    for date in multiple_dates:
        try:
            ret.append(get_weather_info(location, date))
        except Exception as e:
            print("Exception!!", e)
    return ret


processes_num = 22
for location in locations:
    async_res = []
    with multiprocessing.Pool(processes_num) as pool:
        i = 0
        batch_size = len(dates) // processes_num
        for i in range(processes_num):
            async_res.append(
                pool.apply_async(
                    func=get_batch_weather_info,
                    args=(
                        location,
                        dates[i * batch_size:i * batch_size + batch_size],
                    ),
                )
            )
        pool.close()
        pool.join()

    res = []
    for r in async_res:
        try:
            if r.get():
                res.extend(r.get())
        except Exception as e:
            print("Exception", e)


    columns = ["location", "date"] + field_order

    df = pd.DataFrame(
        res,
        columns=columns,
    )

    df.to_csv(f"{data_dir}/{location}.csv", index=False)
