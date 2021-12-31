import json
# with open('./dataset_clean2.txt') as sf:
#     # with open('./dataset.txt') as f:
#     #     for line in f:
#     #         if line == '' or line == '\n':
#     #             continue
#     #         line = line.replace('\'', '\"')
#     #         data = json.loads(line)
#     #         loss = data['loss']
#     #         if loss > 10000 or loss == 0:
#     #             continue
#     #         print(line, file=sf)
#     # with open('./dataset_clean2.txt', 'a') as f:
#     #     for l in sf:
#     #         if l != '\n':
#     #             l = l.replace('\n', '')
#     #             print(l, file=f)
#     with open('./dataset_small.txt', 'a') as f:
#         num = 0
#         for l in sf:
#             if l != '\n':
#                 l = l.replace('\n', '')
#                 num+=1
#                 print(l, file=f)
#             if num > 100000:
#                 break

# with open('./dataset_noweight.txt') as sf:
avg_loss = 0
with open('./dataset_clean2.txt') as sf:
    # with open('./dataset.txt') as f:
    #     for line in f:
    #         if line == '' or line == '\n':
    #             continue
    #         line = line.replace('\'', '\"')
    #         data = json.loads(line)
    #         loss = data['loss']
    #         if loss > 10000 or loss == 0:
    #             continue
    #         print(line, file=sf)
    # with open('./dataset_clean2.txt', 'a') as f:
    #     for l in sf:
    #         if l != '\n':
    #             l = l.replace('\n', '')
    #             print(l, file=f)
    with open('./dataset_small_clean2.txt', 'w') as f:
        num = 0
        for line in sf:
            if line == '' or line == '\n':
                continue
            line = line.replace('\'', '\"')
            line = line.replace('\n', '')
            data = json.loads(line)
            loss = data['loss']

            if loss > 10000 or loss == 0:
                continue

            # if loss > 500:
            #     data['loss'] = 500
            avg_loss += loss

            num+=1
            data_str = json.dumps(data)
            f.write(data_str)
            f.write('\n')
            # print(data_str)
            # print(line, file=f)
            if num > 100000:
                break
print(avg_loss / num)