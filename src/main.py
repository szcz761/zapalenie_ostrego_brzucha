import pandas as pd
import xlrd

if __name__ == "__main__":
    df = pd.read_excel('/home/szymon/repo/zapalenie_ostrego_brzucha/data/Stany_ostrego_brzucha-dane.xls')
    print (df)