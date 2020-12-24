import numpy as np
import pandas as pd
import os as os
import os
from glob import glob

def read_dataset(path, extension):
    all_files = [file
                     for path, subdir, files in os.walk(path)
                     for file in glob(os.path.join(path, extension))]
    return all_files
    
def traverse_currency_dataset(files, n_rows_first, skip_rows_second, n_rows_second):
    df = pd.DataFrame()
    for file in files:
        df1 = pd.read_csv(file, sep='\t', nrows=n_rows_first, header=1)
        df2 = pd.read_csv(file, sep='\t', skiprows=skip_rows_second, nrows=n_rows_second)
        df1T= df1.T
        df2T= df2.T
        df2T = df2T.iloc[1:]
        df_row = pd.concat([df1T, df2T])
        df = df.append(df_row)
    return df

def prepare_dataset():
    path = os.getcwd()+"/RawData_ IMF Exchange Rates"
    extension = "*2011.xls"
    input_files = read_dataset(path, extension)
    df = traverse_currency_dataset(input_files, 51, 56, 51) 
    extension = "*2012.xls"    
    input_files = read_dataset(path, extension)
    df = df.append(traverse_currency_dataset(input_files, 51, 56, 51))
    extension = "*2013.xls"    
    input_files = read_dataset(path, extension)
    df = df.append(traverse_currency_dataset(input_files, 51, 56, 51))
    extension = "*2014.xls"    
    input_files = read_dataset(path, extension)
    df = df.append(traverse_currency_dataset(input_files, 51, 56, 51))
    extension = "*2015.xls"    
    input_files = read_dataset(path, extension)
    df = df.append(traverse_currency_dataset(input_files, 51, 56, 51))
    extension = "*2016.xls"    
    input_files = read_dataset(path, extension)
    df = df.append(traverse_currency_dataset(input_files, 51, 56, 51))
    extension = "*2017.xls"    
    input_files = read_dataset(path, extension)
    df = df.append(traverse_currency_dataset(input_files, 51, 56, 51))
    extension = "*2018.xls"    
    input_files = read_dataset(path, extension)
    df = df.append(traverse_currency_dataset(input_files, 51, 56, 51))
    extension = "Jan*2019.xls"    
    input_files = read_dataset(path, extension)
    df = df.append(traverse_currency_dataset(input_files, 51, 56, 51))

    extension = "F*2019.xls"    
    input_files = read_dataset(path, extension)
    df = df.append(traverse_currency_dataset(input_files, 39, 44, 39))
    extension = "M*2019.xls"    
    input_files = read_dataset(path, extension)
    df = df.append(traverse_currency_dataset(input_files, 39, 44, 39))
    extension = "A*2019.xls"    
    input_files = read_dataset(path, extension)
    df = df.append(traverse_currency_dataset(input_files, 39, 44, 39))
    extension = "Ju*2019.xls"    
    input_files = read_dataset(path, extension)
    df = df.append(traverse_currency_dataset(input_files, 39, 44, 39))
    extension = "S*2019.xls"    
    input_files = read_dataset(path, extension)
    df = df.append(traverse_currency_dataset(input_files, 39, 44, 39))
    extension = "O*2019.xls"    
    input_files = read_dataset(path, extension)
    df = df.append(traverse_currency_dataset(input_files, 39, 44, 39))
    extension = "N*2019.xls"    
    input_files = read_dataset(path, extension)
    df = df.append(traverse_currency_dataset(input_files, 39, 44, 39))
    extension = "D*2019.xls"    
    input_files = read_dataset(path, extension)
    df = df.append(traverse_currency_dataset(input_files, 39, 44, 39))
    extension = "*2020.xls"    
    input_files = read_dataset(path, extension)
    df = df.append(traverse_currency_dataset(input_files, 39, 44, 39))

    df.to_csv("Test.csv")
    
def main():
    prepare_dataset()
    
if __name__ == "__main__":
    main()