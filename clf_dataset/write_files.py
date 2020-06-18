import csv
import tarfile
# from transformers import cached_path
import io

# empd_file = "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz"
EMOTION = 2
CONTEXT = 3
UTTERANCE = 5

def replace_comma(text):
    p = {'_comma_':' ,', '..':'.', '...':'.'}
    for k, v in p.items():
        text = text.replace(k,v)
    return text

def read_from_csv(csv_file):
    data = []
    i = 0
    last_emotion = None
    last_context = None
    # csv_file.seek(0)
    reader = csv.reader(csv_file)
    
    for row in reader:
        if i == 0:
            i += 1
            continue
        # print(row)
        if row[EMOTION] == last_emotion:
            history.append(replace_comma(row[UTTERANCE]))            
        else:
            # append dictionary to data and reinitialize it
            if last_emotion != None:
                entry = {"emotion": last_emotion, "context": last_context, "history": history}
                data.append(entry)
            
            # create data for a new entry in dataset
            last_emotion = row[EMOTION]
            last_context = replace_comma(row[CONTEXT])
            history = [replace_comma(row[UTTERANCE])]
            
    return data

            
csv_name = './dataset/curated_test.csv'
with open(csv_name, 'r') as csv_file:
    data = read_from_csv(csv_file)

with open('test.txt', 'w') as f:
    for d in data:
        line = '__label__' + d["emotion"] + ' ' + d['context'] + ' '
        for h in d['history']:
            if '\n' in h:
                print(h)
            line += h + ' '
        line += '\n'
        f.write(line) 