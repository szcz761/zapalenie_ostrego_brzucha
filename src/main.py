import pandas as pd
import xlrd
import os


dir_path = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    df = pd.read_excel(os.path.join(dir_path, '../data/Stany_ostrego_brzucha-dane.xls'))
    print(df)
