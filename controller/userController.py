from io import BytesIO
import pandas as pd

def convert_cv2(df):  # function to convert dataframe into a csv
    return df.to_csv().encode("utf-8")


def convert_xcel(df):  # fuction to convert a dataframe into an excel file
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer)
    return output

def read_df(data):
    d = {}
    for key in data:
        d[key] = d.get(key, 0) + 1
    df = pd.DataFrame(d.items(), columns=["Name", "Count"])
    return df
