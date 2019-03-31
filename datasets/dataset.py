import pandas as pd, os, numpy as np, json
import xml.etree.ElementTree as etree

# read


def convert_xml_to_csv(panclef_2019_ap, lang='en'):
    # load xml for each language
    data_path = os.path.join(panclef_2019_ap,lang)
    label_path = os.path.join(panclef_2019_ap,lang,'truth.txt')    
    print(label_path)
    label_data = open(label_path)
    label_dict = dict()
    for i in label_data:
        author_id, entity, gender = i.split(":::")
        label_dict[author_id] = [entity, gender.replace("\n","")]

    # print(label_dict)

    # label_path = os.path.join(inputDir,'en/truth.txt')
    # label_file = open(label_path)
    df = pd.DataFrame({'author_id': [], 'text': [], 'gender': [], 'entity': []})

    for filename in os.listdir(data_path):
        
        if not filename.endswith('.xml'): continue
        author_id = os.path.splitext(filename)[0]

        xml_fullname = os.path.join(data_path, filename)
        tree = etree.parse(xml_fullname)
        root = tree.getroot()
        for i in range(len(root[0])):
            text = root[0][i].text
            entity, gender = label_dict[author_id]
            df = df.append({'author_id': author_id, 'text': text, 'gender': gender, 'entity': entity, 'seq': i}, ignore_index=True)			
        
    # df.set_index('author_id')
    #print(df.head())

    # convert xml to text
    # concat text for each author
    # save a dataframe for each language
    df.to_csv("df_"+str(lang)+".csv", index=False, encoding='utf-8')
    return df

def load(lang='en',split_test=0.2, dataframe_path='../'):
    # check if dataframe file exists
    # if not, then create with convert_xml_to_csv
    X_train = []
    y_train = []

    X_test = []
    y_test = []

    return X_train, y_train, X_test, y_test



if __name__ == '__main__':

    #path = 'pan19-author-profiling-training-2019-01-28'
    path = 'pan19-author-profiling-training-2019-02-18'
    convert_xml_to_csv(path, lang='en')
    convert_xml_to_csv(path, lang='es')
