from time import sleep

import numpy as np
import sys
project_root_path = "your project root path"
sys.path.append(project_root_path)

import torch
from torch import nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from data_process import report_data_organize, context_vector
import math
from util import file_util


model = SentenceTransformer('distilroberta-base-paraphrase-v1')
data_path = "data/test_cases/"
model_path = "trained_model/"

BERT_VECTOR_SIZE = 768
HIDDEN_SIZE = 200
LSTM_Layer = 3


class ThreatContext_enhanced_NN(nn.Module):
    def __init__(self, output_size, filter_num=1, kernel_lst=(2,3), hidden_dim=HIDDEN_SIZE, num_layers=LSTM_Layer, dropout=0.5):
        super(ThreatContext_enhanced_NN, self).__init__()
        self.rnn = nn.LSTM(BERT_VECTOR_SIZE, hidden_dim, num_layers=num_layers, bidirectional=True, dropout=0.5)
        self.dropout = nn.Dropout(0.5)


        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, filter_num, (2, 2)),
                    nn.ReLU(),
                    nn.MaxPool2d((1, 2))),
                nn.Sequential(
                    nn.Conv2d(1, filter_num, (3, 3)),
                    nn.ReLU(),
                    nn.MaxPool2d((2, 1)))
            ]
        )
        self.fc_1 = nn.Linear(hidden_dim * 2, 128)
        self.fc_2 = nn.Linear(132, 128)
        self.fc_3 = nn.Linear(128, output_size)


        self.w_omega = nn.Parameter(torch.Tensor(hidden_dim * 2, hidden_dim * 2))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_dim * 2, 1))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def attention_net_soft(self, x, query, mask=None):

        d_k = query.size(-1)
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)

        p_attn = F.softmax(scores, dim=-1)
        context = torch.matmul(p_attn, x)
        return context, p_attn

    def forward(self, bert_x, ti_vector_x):
        bert_x_seq = bert_x.permute(1, 0, 2)
        bert_x_seq = self.dropout(bert_x_seq)
        bi_lstm_out, (final_hidden_state, final_cell_state) = self.rnn(bert_x_seq)
        bi_lstm_out = bi_lstm_out.permute(1, 0, 2)
        query = self.dropout(bi_lstm_out)
        # soft Attention
        attn_output, attention = self.attention_net_soft(bi_lstm_out, query)


        attn_output = attn_output.permute(1,0,2)
        attn_output = attn_output[1:2]
        attn_output = attn_output.permute(1,0,2)
        fc_out_1 = self.fc_1(attn_output)# 128 * 1
        fc_out_1 = fc_out_1.squeeze(1)

        ti_vector_x = ti_vector_x.unsqueeze(1)
        ti_out = [conv(ti_vector_x) for conv in self.convs]
        ti_out = torch.cat(ti_out, dim=2)# 4 *1
        ti_out = ti_out.view(bert_x.size(0), -1)


        rnn_concat_ti_out = torch.cat((fc_out_1, ti_out), dim=1)
        fc_out_2 = self.fc_2(rnn_concat_ti_out)
        final_out = self.fc_3(fc_out_2)
        return final_out

def prediction(TIeNN, x_test, device):
    TIeNN.eval()
    torch.no_grad()
    text_vector = []
    ioc_vector = []
    for text_ioc_vector in x_test:
        text_vector.append(text_ioc_vector[0])
        ioc_vector.append(text_ioc_vector[1])
    ioc_vector = np.array(ioc_vector)
    text_vector = np.array(text_vector)
    text_batch = torch.Tensor(text_vector)
    ioc_batch = torch.Tensor(ioc_vector)
    text_batch = text_batch.to(device)
    ioc_batch = ioc_batch.to(device)
    pred = TIeNN(text_batch, ioc_batch)
    pred_label = torch.max(pred, dim=1)[1]
    pred_cpu = pred_label.cpu()
    return pred_cpu

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

if __name__ == '__main__':
    T_id = "TA0009"
    device = get_default_device()
    test_text = file_util.txt_to_str(project_root_path + data_path + T_id + ".txt")
    all_text_data = []
    all_ioc_vector_data = []
    context_data = report_data_organize(test_text)

    for temp_dict in context_data:
        ioc_vector = temp_dict["ioc_vector"]
        ioc_vector = np.array(ioc_vector).reshape(len(ioc_vector))
        ioc_vector = ioc_vector.reshape(4, 3)
        all_ioc_vector_data.append(ioc_vector)
        sent_list = temp_dict["sent_list"]
        context_info_vector = context_vector(sent_list, model)
        all_text_data.append(context_info_vector)
    all_text_data_np = np.array(all_text_data)

    feed_in_x = []
    for i in range(0, len(all_text_data_np)):
        feed_in_x.append([all_text_data_np[i], all_ioc_vector_data[i]])
    TCE_NN = ThreatContext_enhanced_NN(2)
    model_state_dict = torch.load(project_root_path + model_path + T_id + ".pth", map_location=torch.device(device))
    TCE_NN.load_state_dict(model_state_dict)

    TCE_NN.to(device)
    result = prediction(TCE_NN, feed_in_x, device)
    label_list_str = file_util.txt_to_str(project_root_path + data_path + T_id + "_label.txt")
    label_list_int = [int(y) for y in label_list_str.split()]
    err = np.abs(np.array(label_list_int) - np.array(result))
    acc = 1 - float(np.abs(err).sum() / result.size())
    for i in range(0, len(context_data)):
        print("=========================")
        print("         Content         ")
        print("-------------------------")
        for sent in context_data[i]["origin_list"]:
            print(sent)
        pred_res = int(result[i])
        if pred_res == 1:
            print("-------------------------")
            print("Predict result: {} Attack      ".format(T_id))
            print("-------------------------")
        else:
            print("-------------------------")
            print("Predict result: Not {} Attack      ".format(T_id))
            print("-------------------------")
        # sleep(2)
    print("True:", label_list_int)
    print("Pred:", [int(res) for res in result])
    print("ACC:", acc)

