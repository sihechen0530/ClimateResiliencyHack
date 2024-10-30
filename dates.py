from datetime import datetime, timedelta

start_month = 6
end_month = 9

start_year = 2020
end_year = 2024

def get_dates(start_year, end_year, start_month, end_month, start_day=1, end_day=30):
    dates = []
    for year in range(start_year, end_year + 1):
        start_date = datetime(year, start_month, start_day)  # June 1st
        end_date = datetime(year, end_month, end_day)   # September 30th
        current_date = start_date
        
        while current_date <= end_date:
            dates.append(current_date.strftime("%Y-%m-%d"))
            current_date += timedelta(days=1)
    return dates

dates = get_dates(start_year, end_year, start_month, end_month)

