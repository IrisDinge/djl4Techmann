import pandas as pd
import json
import csv
import os


def main(csv_path, new_path, json_path):
    df = pd.read_csv(csv_path, index_col=None, dtype=str)
    df['imageHeight'] = 96
    df['imageWidth'] = 96
    a = [df['num'][0:1495].replace(r'\t', '', regex=True) + '.jpg',
         '00' + df['num'][1495:4927] + '.jpg']
    b = pd.concat(a)
    df.insert(0, "imagePath", b)
    c = df.drop(['num', ], axis=1)
    c.to_csv(new_path, index=False)

    csvfile = open(new_path, 'r')
    jsonfie = open(json_path, 'w')
    fieldnames = ('imagePath', 'snout_x', 'snout_y', 'body_x', 'body_y',
                  'tail_x', 'tail_y', 'imageHeight', 'imageWidth')
    reader = csv.DictReader(csvfile, fieldnames)

    for row in reader:
        json.dump(row, jsonfie)
        jsonfie.write('\n')


def multiplejson(json_path, annotation):
    with open(json_path, 'r') as f:
        for line in f.readlines()[1:]:
            data = json.loads(line)
            name = data['imagePath'].replace(r'.jpg', '')
            dicta = {'label': 'snout',
                     'points': [[int(data['snout_x']), int(data['snout_y'])]],
                     'group_id': None,
                     'shape_type': "points",
                     "flags": {}}
            dictb = {'label': 'body',
                     'points': [[int(data['body_x']), int(data['body_y'])]],
                     'group_id': None,
                     'shape_type': "points",
                     "flags": {}
                     }
            dictc = {'label': 'tail',
                     'points': [[int(data['tail_x']), int(data['tail_y'])]],
                     'group_id': None,
                     'shape_type': "points",
                     "flags": {}
                     }
            dictshape = {'version': "4.5.6",
                         'flags': {},
                         'shape': [dicta, dictb, dictc],
                         'imagePath': data['imagePath'],
                         'imageHeight': int(data['imageHeight']),
                         'imageWidth': int(data['imageWidth'])}
            annotation_path = os.path.join(annotation, name + '.json')
            with open(annotation_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(dictshape, indent=1))
                f.close()


main("C:\\Users\\TME-DJ\\Desktop\\total1\\part.txt",
     "C:\\Users\\TME-DJ\\Desktop\\total1\\new.csv",
     "C:\\Users\\TME-DJ\\Desktop\\total1\\total.json")
multiplejson("C:\\Users\\TME-DJ\\Desktop\\total1\\total.json", "C:\\Users\\TME-DJ\\Desktop\\total1\\annotation")

