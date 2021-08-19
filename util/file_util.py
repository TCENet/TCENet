import csv
import json


def dict_to_jsonfile(dictdata, dir_path):
    jsonfile_str = json.dumps(dictdata)
    fo = open(dir_path, "w")
    fo.write(jsonfile_str)
    return 0


def jsonfile_to_dict(dir_path):
    fo = open(dir_path, "r")
    dictdata = json.load(fo)
    return dictdata


def list_to_csv(header, list_data, dir_path, if_write_haeder):
    fo = open(dir_path, "w")
    writer = csv.writer(fo)
    if if_write_haeder:
        writer.writerow(header)
    writer.writerows(list_data)
    fo.close()


def csv_to_list(dir_path):
    fo = open(dir_path, "r")
    csv_reader = csv.reader(fo)
    csv_header = next(csv_reader)
    csv_data = [row for row in csv_reader]

    return csv_header, csv_data

def str_to_txt(file_dir,text):
    try:
        fo = open(file_dir,"w")
        fo.write(text)
        fo.close()
    except Exception as e:
        print("File writing exception: ",e )


def txt_to_str(file_dir):
    try:
        fo = open(file_dir,"r")
        file_text = fo.read()
        fo.close()
    except Exception as e:
        print("File open exception: ",e )
        file_text = ""
    return file_text


if __name__ == '__main__':
    root_path = ""
    _, all_data = csv_to_list(root_path + "data/nodes.csv")
    print(len(all_data))
