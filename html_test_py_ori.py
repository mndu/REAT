from sklearn.preprocessing import MinMaxScaler
from csv import reader

scaler = MinMaxScaler()
# transform data

rows = open('attri_data.csv', "rb")
html = """\
    <table border='10'>
    <tr><th>Abstract</th><th>Ground Truth</th><th>Model Prediction</th></tr>"""
csv_reader = reader(rows)
# Iterate over each row in the csv using reader object
for row in csv_reader:
    # row variable is a list that represents a row in csv
    # print(row)
    # for row in rows:
    print(row)
    print('==========')
    html = html + "<tr>"
    # word_attri_pairs = row.split('[')[1]

    # word_attri_pairs = word_attri_pairs.split(']')[0]
    # word_attri_pairs = row.split('[')[1]
    # word_attri_pairs = word_attri_pairs.split(']')[0]
    # word_attri_pairs = row.split(';')
    # print(word_attri_pairs)

    # word_items = row.split('[')[1]
    # word_items = word_items.split(']')[0]
    # attri_items = row.split(']')[1]
    # attri_items = attri_items.split('[')[1]
    # word_list = word_items.split(',')
    # word_list = word_list.remove(" '")
    # attri_list = attri_items.split(',')
    # attri_list = [float(i) for i in attri_list]
    # new_value = ((old_value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
    # print(attri_list)
    # attri_list_norm = [(float(i) - min(attri_list)) * 255 / (max(attri_list) - min(attri_list)) + 0 for i in
    #                    attri_list]
    # print(type(attri_list_norm))
    # print('word list length:', len(word_list))
    # print('attribution list length:', len(attri_list_norm))
    # print('word_list', word_list)
    # html = html + "<td>"
    # for i in range(len(word_list)):
    #     word = word_list[i]
    #     attribution_score = attri_list_norm[i]
    #     if attri_list[i] <= 0:
    #         r = str(attribution_score)
    #         # print(r)
    #         col_color = "<span style=\'color:rgb(" + r + ",0,0)\'>" + word + "</\span>"
    #     else:
    #         g = str(attribution_score)
    #         # print(g)
    #         col_color = "<span style=\'color:rgb(0," + g + ",0)\'>" + word + "</\span>"
    html = html + "<td>"
    word_list = []
    score_list = []
    for index_i in range(len(row)):
        print('row[index_i]', row[index_i])
        word = row[index_i].split(':')[0]
        print('word', word)
        attribution_score = row[index_i].split('%')[0]
        attribution_score = attribution_score.split(':')[1]
        print('attribution_score', attribution_score)
        word_list.append(word)
        ground_truth = row[index_i].split('%')[1]
        print('ground_truth', ground_truth)
        try:
            score_list.append(float(attribution_score))
        except:
            print(attribution_score)
    attri_list_norm = [(float(i) - min(score_list)) * 255 / (max(score_list) - min(score_list)) + 0 for i in score_list]
    for j in range(len(word_list)):
        try:
            if score_list[j] <= 0:
                r = str(attri_list_norm[j])
                col_color = "<span style=\'color:rgb(" + r + ",255,100)\'>" + word_list[j] + " " + "</\span>"
            else:
                g = str(attri_list_norm[j])
                col_color = "<span style=\'color:rgb(255," + g + ",100)\'>" + word_list[j] + " " + "</\span>"
        except:
            print(j)

        html = html + col_color
    html = html + "</td>"
    html = html + "<td>" + "</td>"

    html = html + "</tr>"
    html = html + "</table>"

    f = open('Viz.html', 'w')
    f.write(html)
    f.close()
