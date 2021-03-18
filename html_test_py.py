from sklearn.preprocessing import MinMaxScaler
from csv import reader
import pandas as pd
import argparse

args = argparse.ArgumentParser()
args.add_argument('--data', dest='data', default='abstract', help='specify the dataset')
args = args.parse_args()
orginal_labels = pd.read_csv('./data/SCORE/' + args.data + '_ori/train.csv')

scaler = MinMaxScaler()
# transform data
rows = open(args.data + "_attri_data.csv", "rb")
html = """\
    <table border='10'>
    <tr><th>Pid</th><th>""" + args.data + """</th><th>Ground Truth</th><th>Model Prediction</th></tr>"""
for row in rows:
    all_rows = row.split('@')
    print(len(all_rows))
    ## For every row:
    for row_index in range(len(all_rows) - 1):
        html = html + "<tr>"
        html = html + "<td>" + str(orginal_labels.loc[row_index][0]) + "</td>"
        html = html + "<td>"
        word_list = []
        score_list = []
        ground_truth_list = []
        this_row = all_rows[row_index]
        if args.data == 'claim2':
            # word_score_pairs = this_row.split('[')[0]
            word_score_pairs = this_row.split(';')
            for word_pair in word_score_pairs:
                if word_pair != "":
                    word = word_pair.split('*')[0]
                    try:
                        ground_truth = word_pair.split('%')[1]
                    except:
                        print(word_pair)
                    score = word_pair.split('*')[1]
                    score = score.split('%')[0]
                    word_list.append(word)
                    try:
                        score_list.append(float(score))
                    except:
                        print(score)
                    ground_truth_list.append(ground_truth)
        else:
            word_score_pairs = this_row.split('[')[0]
            word_score_pairs = word_score_pairs.split(';')
            for word_pair in word_score_pairs:
                if word_pair != "":
                    word = word_pair.split('*')[0]
                    try:
                        ground_truth = word_pair.split('%')[1]
                    except:
                        print(word_score_pairs)
                        print(word_pair)
                    score = word_pair.split('*')[1]
                    score = score.split('%')[0]
                    word_list.append(word)
                    try:
                        score_list.append(float(score))
                    except:
                        print(score)
                    ground_truth_list.append(ground_truth)
                    # score_all = sum(score_list)
                    # print('=======sum(score_list)', score_all)
        try:
            # attri_list_norm = [(float(i) - min(score_list)) * 255 / (max(score_list) - min(score_list)) + 0 for i in
            #                    score_list]
            attri_list_norm = [(float(i) - min(score_list)) * 1 / (max(score_list) - min(score_list)) + 0 for i in
                               score_list]
        except:
            print('max(score_list)', max(score_list))
            print('min(score_list)', min(score_list))

        for j in range(len(word_list)):
            try:
                if score_list[j] <= 0:
                    r = str(attri_list_norm[j])
                    # print(r)
                    # col_color = "<span style=\'color:rgb(" + r + ",255,100)\'>" + word_list[j] + " " + "</\span>"
                    col_color = "<span style=\'color:black;background-color:#ff0000;opacity:" + r + "\'>" + word_list[
                        j] + " " + "</\span>"
                    # col_color = "<span style=\'background-color:#ff0000\'>" + word_list[
                    #     j] + " " + "</\span>"
                else:
                    g = str(attri_list_norm[j])
                    # print(g)
                    # col_color = "<span style=\'color:rgb(255," + g + ",100)\'>" + word_list[j] + " " + "</\span>"
                    col_color = "<span style=\'color:black;background-color:#00ff00;opacity:" + g + "\'>" + word_list[
                        j] + " " + "</\span>"
            except:
                print(j)
            html = html + col_color
            # html = html + ground_truth_html+prediction_html
        html = html + "</td>"
        html = html + "<td>" + str(orginal_labels.loc[row_index][-1]) + "</td>"
        html = html + "<td>" + str(orginal_labels.loc[row_index][-2]) + "</td>"
        html = html + "<td>" + "</td>"

    html = html + "</tr>"
    html = html + "</table>"

    f = open('Viz_' + args.data + '.html', 'w')
    f.write(html)
    f.close()
