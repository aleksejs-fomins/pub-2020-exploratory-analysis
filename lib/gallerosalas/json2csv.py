import json
import pandas as pd


# Convert settings file from json to csv format
def jsonMice2csv(pwd):
    with open(pwd, 'r') as json_file:
        dataDict = json.load(json_file)

    dataLst = []
    for mousename, mouseDict in dataDict.items():
        for dateKey, sessionLst in mouseDict.items():
            for sessionKey in sessionLst:
                dataLst += [[mousename, dateKey, sessionKey]]

    df = pd.DataFrame(dataLst, columns=['mousename', 'dateKey', 'sessionKey'])
    df.to_csv(
        '/media/aleksejs/DataHDD/work/data/yasir/yasirdata_raw/sessions.csv',
        sep='\t', index=False
    )



jsonMice2csv('/media/aleksejs/DataHDD/work/data/yasir/yasirdata_raw/sessions.json')

