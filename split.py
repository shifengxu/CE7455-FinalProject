import pandas as pd

# with open('C:/Users/annch/Documents/NTU/final_proj/data.txt', 'r', encoding='utf-8') as f:
#     big_file = f.readlines()
#     print(len(big_file))
#     train_list = big_file[:int(0.8*len(big_file))]
#     test_list = big_file[int(0.8 * len(big_file)):]
#
# with open('C:/Users/annch/Documents/NTU/final_proj/data/train.txt', 'w', encoding='utf-8') as f:
#     for train in train_list:
#         f.write(train)
#
# with open('C:/Users/annch/Documents/NTU/final_proj/data/test.txt', 'w', encoding='utf-8') as f:
#     for test in test_list:
#         f.write(test)

with open('C:/Users/annch/Documents/NTU/final_proj/data/adolescent_train.txt', 'r', encoding='utf-8') as f:
    f = f.read()
    print(f[:100])
    words = list(f.split(' '))
    print(len(words))

with open('C:/Users/annch/Documents/NTU/final_proj/data/adolescent_valid.txt', 'r', encoding='utf-8') as f:
    f = f.read()
    print(f[:100])
    words = list(f.split(' '))
    print(len(words))

test_df = pd.read_csv('C:\\Users\\annch\\Documents\\NTU\\final_proj\\BBC_News_Test.csv')
test_txt = test_df['Text']
combined_text = []
word_counter = 0
for excerpt in test_txt[100:250]:
    combined_text.append(excerpt)
    words = len(list(excerpt.split(' ')))
    word_counter+=words

print(excerpt)
print(word_counter)

with open('C:/Users/annch/Documents/NTU/final_proj/data/adult_test.txt', 'w', encoding='utf-8') as f:
    for txt in combined_text:
        f.write(txt)