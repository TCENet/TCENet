#coding: utf-8
import numpy as np
import sys
project_root_path = "your project root path"
sys.path.append(project_root_path)

from copy import deepcopy
import nltk
import numpy as np
from util import ioc_extract, file_util
import data_structure

# data_path = "Data/Bert_Data/"
# model_path = "Model/Bert/model/"


def ioc_info(text, sent_list):
    """
        Extract IOC from target sentence context
        :param text:
        :return: IOC_vector, text_ioc_holder
        """
    try:
        ioc_normalized, results = ioc_extract.extract_observables(text)
        results_remain_dict = {}
        ioc_normalized_remain_dict = {}
        ioc_mapping_dict = deepcopy(data_structure.ioc_type_mapping_dict)
        btr_ioc_type_dict = deepcopy(data_structure.BTR_ioc_type_dict)

        ## origin IOC dict
        for ioc_type in ioc_normalized.keys():
            if ioc_type not in ioc_mapping_dict.keys():
                continue
            if len(results[ioc_type]) != 0:
                mapping_key = ioc_mapping_dict[ioc_type]
                ioc_normalized_remain_dict[mapping_key] = list(ioc_normalized[ioc_type])

        ## normalized IOC dict
        for ioc_type in results.keys():
            if ioc_type not in ioc_mapping_dict.keys():
                continue
            if len(results[ioc_type]) != 0:
                mapping_key = ioc_mapping_dict[ioc_type]
                results_remain_dict[mapping_key] = list(results[ioc_type])

        # IOC placeholder
        if "filepath" in results_remain_dict.keys() and "filename" in results_remain_dict.keys():
            filepath_list = results_remain_dict["filepath"]
            filename_list = results_remain_dict["filename"]
            remove_filename_list = []
            for path in filepath_list:
                for filename in filename_list:
                    if path.find(filename) >= 0:
                        remove_filename_list.append(filename)

            remove_filename_list = list(set(remove_filename_list))
            for filename in remove_filename_list:
                filename_list.pop(filename_list.index(filename))
            for i in range(0, len(sent_list)):
                for filepath in filepath_list:
                    sent_list[i] = sent_list[i].lower().replace(filepath, " $[filepath] ")
                for filename in filename_list:
                    sent_list[i] = sent_list[i].lower().replace(filename, " $[filename] ")

        # IOC placeholder
        elif "filepath" in results_remain_dict.keys():
            filepath_list = results_remain_dict["filepath"]
            for i in range(0, len(sent_list)):
                for temp_ioc in filepath_list:
                    sent_list[i] = sent_list[i].lower().replace(temp_ioc, " $[filepath] ")

        elif "filename" in results_remain_dict.keys():
            filename_list = results_remain_dict["filename"]
            for i in range(0, len(sent_list)):
                for temp_ioc in filename_list:
                    sent_list[i] = sent_list[i].lower().replace(temp_ioc, " $[filename] ")
        # IOC placeholder
        for ioc_type in results_remain_dict:
            if ioc_type == "filename" or ioc_type == "filepath":
                continue
            if ioc_type == "fqdn" and "email" in results_remain_dict.keys():
                email_list_str = " ".join(results_remain_dict["email"])
                fqdn_list = results_remain_dict["fqdn"]
                remove_fqdn_list = []
                for fqdn in fqdn_list:
                    if email_list_str.find(fqdn):
                        remove_fqdn_list.append(fqdn)
                remove_fqdn_list = list(set(remove_fqdn_list))
                for fqdn in remove_fqdn_list:
                    fqdn_list.pop(fqdn_list.index(fqdn))
                results_remain_dict["fqdn"] = fqdn_list
            temp_ioc_list = results_remain_dict[ioc_type]
            for i in range(0, len(sent_list)):
                for temp_ioc in temp_ioc_list:
                    sent_list[i] = sent_list[i].lower().replace(temp_ioc, " $[" + ioc_type + "] ")

        # IOC placeholder
        encode_decode_method_list = deepcopy(data_structure.encode_decode_method_list)
        for method in encode_decode_method_list:
            for i in range(0, len(sent_list)):
                temp_sent = sent_list[i].lower()
                if temp_sent.find(method) >= 0:
                    sent_list[i] = sent_list[i].lower().replace(method, " $[codemethod] ")
                    if "codemethod" not in results_remain_dict.keys():
                        results_remain_dict["codemethod"] = [method]
                    else:
                        if method not in results_remain_dict["codemethod"]:
                            results_remain_dict["codemethod"].append(method)

                    if "codemethod" not in ioc_normalized_remain_dict.keys():
                        ioc_normalized_remain_dict["codemethod"] = [method]
                    else:
                        if method not in ioc_normalized_remain_dict["codemethod"]:
                            ioc_normalized_remain_dict["codemethod"].append(method)

        collection_data_object_list = deepcopy(data_structure.collection_data_object)
        for data_object in collection_data_object_list:
            for i in range(0, len(sent_list)):
                temp_sent = sent_list[i].lower()
                if temp_sent.find(data_object) >= 0:
                    sent_list[i] = sent_list[i].lower().replace(data_object, " $[dataobject] ")
                    if "dataobject" not in results_remain_dict.keys():
                        results_remain_dict["dataobject"] = [data_object]
                    else:
                        if data_object not in results_remain_dict["dataobject"]:
                            results_remain_dict["dataobject"].append(data_object)

                    if "dataobject" not in ioc_normalized_remain_dict.keys():
                        ioc_normalized_remain_dict["dataobject"] = [data_object]
                    else:
                        if data_object not in ioc_normalized_remain_dict["dataobject"]:
                            ioc_normalized_remain_dict["dataobject"].append(data_object)

        protocol_list = data_structure.protocol
        for protocol in protocol_list:
            for i in range(0, len(sent_list)):
                temp_sent = sent_list[i].lower()
                if temp_sent.find(protocol) >= 0:
                    sent_list[i] = sent_list[i].lower().replace(protocol, " $[protocol] ")
                    if "protocol" not in results_remain_dict.keys():
                        results_remain_dict["protocol"] = [protocol]
                    else:
                        if protocol not in results_remain_dict["protocol"]:
                            results_remain_dict["protocol"].append(protocol)

                    if "protocol" not in ioc_normalized_remain_dict.keys():
                        ioc_normalized_remain_dict["protocol"] = [protocol]
                    else:
                        if protocol not in ioc_normalized_remain_dict["protocol"]:
                            ioc_normalized_remain_dict["protocol"].append(protocol)

        # IOC vector
        ioc_vector = np.zeros(len(btr_ioc_type_dict.keys()))
        for ioc_type in results_remain_dict:
            ioc_type_pos = btr_ioc_type_dict[ioc_type]
            ioc_vector[ioc_type_pos] = len(results_remain_dict[ioc_type])
        temp_mean = np.array(ioc_vector).mean()
        temp_max = np.array(ioc_vector).max()
        temp_min = np.array(ioc_vector).min()
        if np.array(ioc_vector).sum() == 0:
            ioc_vector_normalized = ioc_vector
        else:
            ioc_vector_normalized = [(float(i) - temp_mean) / (temp_max - temp_min) for i in np.array(ioc_vector)]
        return sent_list, ioc_vector_normalized, ioc_normalized_remain_dict, results_remain_dict
    except Exception as e:
        print("IOC extract ERROR:", e)
        return sent_list, list(np.zeros(12)), {}, {}

def report_split(text):
    """
    Return text without unnecessary symbols. Split paragraph with $[para].
    :param text:
    :return:
    """

    # text = text.replace("\n", "$[para]. ")
    temp_sent_list = nltk.sent_tokenize(text)

    record_remove_index = []
    for i in range(0, len(temp_sent_list)):
        if i == len(temp_sent_list) - 1:
            continue
        temp_sent = temp_sent_list[i]
        if (temp_sent != '$[para].' and temp_sent.startswith('$[para].') == 1) or (temp_sent != '$[para].' and temp_sent.endswith('$[para].') == 1):
            temp_sent = temp_sent.replace("$[para].", "")
            if not temp_sent.endswith("."):
                temp_sent = temp_sent + "."
            temp_sent_list[i] = temp_sent
        # if temp_sent == "$[para]." and temp_sent_list[i+1] == "$[para].":
        #     record_remove_index.append(i+1)

    paragraph_split_list = []
    for i in range(0, len(temp_sent_list)):
        if i not in record_remove_index:
            temp_sent = temp_sent_list[i]
            if temp_sent.endswith(".."):
                temp_sent = temp_sent[:-1]
            paragraph_split_list.append(temp_sent)

    final_text = " ".join(paragraph_split_list)
    test_list = nltk.sent_tokenize(final_text)
    if len(test_list) != len(paragraph_split_list):
        print("nltk.sent_tokenize split count no same")
        print("post split lenth: ", len(test_list))
        print("before split lenth: ", len(paragraph_split_list))

    return final_text, paragraph_split_list


def report_data_organize(text):
    split_text, sentence_list = report_split(text)
    context_data = []
    for i in range(1, len(sentence_list), 3):
        if i == 0:
            sent_front = "$[para]. "
            sent = sentence_list[i]
            if len(sentence_list) == 1:
                sent_post = "$[para]. "
            else:
                sent_post = sentence_list[i + 1]
        elif i == len(sentence_list) - 1:
            sent_front = sentence_list[i - 1]
            sent = sentence_list[i]
            sent_post = "$[para]. "
        else:
            sent_front = sentence_list[i - 1]
            sent = sentence_list[i]
            sent_post = sentence_list[i + 1]
        sent_list = [sent_front.lower(), sent.lower(), sent_post.lower()]
        temp_text = sent_front.lower() + " " + sent.lower() + " " + sent_post.lower()
        sent_list_ioc_holder, ioc_vector, ioc_normalized, result_remain_dict = ioc_info(temp_text, sent_list)
        context_data.append({
            "origin_list": [sent_front, sent, sent_post],
            "sent_list": sent_list_ioc_holder,
            "ioc_vector": list(ioc_vector),
            "ioc_normalized": ioc_normalized
        })
    # if len(context_data) > 1:
    #     print("BREAK")
    return context_data



def report_data_organize_for_sent(sent_list):
    context_data = []
    for i in range(0, len(sent_list)):
        sent_list[i] = str(sent_list[i]).lower()
    temp_text = " ".join(sent_list)
    sent_list_ioc_holder, ioc_vector, ioc_normalized, result_remain_dict = ioc_info(temp_text, sent_list)
    context_data.append({
        "origin_list": sent_list[i],
        "sent_list": sent_list_ioc_holder,
        "ioc_vector": list(ioc_vector),
        "ioc_normalized": ioc_normalized
    })

    return context_data

def report_data_organize_for_sent_list(sent_list):
    context_data = []
    for i in range(0, len(sent_list)):
        temp_sens = []
        for k in sent_list[i]:
            temp_sens.append(str(k).lower())
        temp_text = " ".join(temp_sens)
        sent_list_ioc_holder, ioc_vector, ioc_normalized, result_remain_dict = ioc_info(temp_text, temp_sens)
        context_data.append({
            "origin_list": sent_list[i],
            "sent_list": sent_list_ioc_holder,
            "ioc_vector": list(ioc_vector),
            "ioc_normalized": ioc_normalized
        })
    return context_data

def context_vector(sent_list, model):
    sent_front = sent_list[0]
    sent_core = sent_list[1]
    sent_post = sent_list[2]
    if sent_front.find("$[para].") >= 0:
        sent_front = sent_core
    if sent_post.find("$[para].") >= 0:
        sent_post = sent_core
    sent_vector = sentence_BERT_Vector(sent_front, sent_core, sent_post, model)
    return sent_vector

def sentence_BERT_Vector(sent_front, sent_core, sent_post, model):
    text = [sent_front, sent_core, sent_post]
    sent_embed = model.encode(text)
    return sent_embed